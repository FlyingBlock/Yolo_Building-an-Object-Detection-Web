"""
FastAPI backend for YOLO object detection.

Run it with:

    uvicorn api:app --reload --port 8000

Interactive docs are then at http://127.0.0.1:8000/docs and the main endpoint is
POST /detect. On the first request the model weights are downloaded (~6 MB for
yolov8n) and the first CUDA inference costs a one-off ~16 s of kernel setup; hit
POST /warmup once to pay that cost up front instead of during a user's upload.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

import detector as det

__version__ = "1.0.0"

logger = logging.getLogger("yolo.api")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally pay the model/cache warmup cost at boot rather than on demand."""
    if det.PRELOAD_MODEL:
        try:
            info = det.warmup(det.DEFAULT_MODEL)
            logger.info("Preloaded %s on %s in %s ms", info["model"], info["device"], info["total_ms"])
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("Preload failed: %s", exc)
    yield


app = FastAPI(
    title="YOLO 目标检测 API",
    description=(
        "通过 HTTP 提供 YOLOv8 / YOLO11 推理服务。向 /detect 提交一张图片，即可获得"
        "边界框、类别标签、置信度分数以及标注后的图片。"
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ERROR HANDLING
# ============================================================

@app.exception_handler(det.ModelError)
async def model_error_handler(request: Request, exc: det.ModelError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})


# ============================================================
# HEALTH / METADATA
# ============================================================

@app.get("/", tags=["元信息"], summary="服务信息")
def root() -> dict[str, Any]:
    return {
        "service": "YOLO 目标检测 API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "GET /health": "服务与运行设备状态",
            "GET /models": "可用模型与已缓存模型",
            "POST /warmup": "加载权重并完成首次推理预热",
            "POST /models/upload": "安装自定义 .pt 模型",
            "POST /detect": "目标检测，返回 JSON",
            "POST /detect/stream": "目标检测，以 NDJSON 流返回推理进度与结果",
            "POST /detect/image": "目标检测，返回标注后的 PNG 图片",
        },
    }


@app.get("/health", tags=["元信息"], summary="服务与运行设备状态")
def health() -> dict[str, Any]:
    device = det.resolve_device()
    gpu: dict[str, Any] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            gpu = {
                "name": torch.cuda.get_device_name(index),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
            }
    except Exception:  # pragma: no cover
        gpu = None

    return {
        "status": "ok",
        "version": __version__,
        "device": device,
        "gpu": gpu,
        "default_model": det.DEFAULT_MODEL,
        "confidence_threshold": det.CONFIDENCE_THRESHOLD,
        "iou_threshold": det.IOU_THRESHOLD,
        "max_image_size": det.MAX_IMAGE_SIZE,
        "max_upload_mb": det.MAX_UPLOAD_MB,
        "custom_model_upload": det.ALLOW_CUSTOM_MODEL_UPLOAD,
        "models_cached": det.cached_models(),
    }


@app.get("/models", tags=["元信息"], summary="列出内置模型、自定义模型与已缓存模型")
def models() -> dict[str, Any]:
    return {"success": True, **det.available_models()}


@app.post("/warmup", tags=["元信息"], summary="加载权重并执行一次预热推理")
def warmup(
    model: str | None = Query(None, description="模型别名，例如 yolov8n"),
    device: str | None = Query(None, description="运行设备：auto | cpu | cuda:0 | mps"),
) -> dict[str, Any]:
    return {"success": True, **det.warmup(model, device)}


@app.post("/models/upload", tags=["元信息"], summary="安装自定义 .pt 模型")
async def upload_model(file: UploadFile = File(..., description="一个 .pt 权重文件")):
    if not det.ALLOW_CUSTOM_MODEL_UPLOAD:
        raise HTTPException(
            status_code=403,
            detail="自定义模型上传功能已关闭（ALLOW_CUSTOM_MODEL_UPLOAD=0）。",
        )
    data = await file.read()
    if not det.is_allowed_upload(len(data)):
        raise HTTPException(status_code=413, detail=f"文件超过 {det.MAX_UPLOAD_MB} MB 上限。")
    try:
        name = det.save_custom_model(file.filename or "", data)
    except det.ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "success": True,
        "model": name,
        "bytes": len(data),
        "warning": ".pt 文件本质上是 pickle，请只加载你信任的模型。",
    }


# ============================================================
# DETECTION
# ============================================================

async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的文件为空。")
    if not det.is_allowed_upload(len(data)):
        raise HTTPException(
            status_code=413,
            detail=f"图片超过 {det.MAX_UPLOAD_MB} MB 上限（MAX_UPLOAD_MB）。",
        )
    return data


async def _run_detection(
    file: UploadFile,
    confidence: float | None,
    iou: float | None,
    model: str | None,
    device: str | None,
    max_size: int | None,
    classes: str | None,
    annotate: bool,
    image_format: str,
) -> dict[str, Any]:
    data = await _read_upload(file)
    try:
        image = det.decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    class_filter: list[int] | None = None
    if classes:
        try:
            class_filter = [int(part) for part in classes.replace(" ", "").split(",") if part]
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="classes 必须是逗号分隔的类别 id 列表。"
            ) from exc

    started = time.perf_counter()
    try:
        payload = det.detect(
            image,
            model=model,
            confidence=confidence,
            iou=iou,
            device=device,
            max_size=max_size,
            annotate=annotate,
            classes=class_filter,
            image_format=image_format,
        )
    except det.ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"检测失败：{exc}") from exc

    payload["success"] = True
    payload["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    payload["summary"] = det.summarise(payload["detections"])
    return payload


@app.post("/detect", tags=["检测"], summary="执行目标检测并返回 JSON")
async def detect(
    file: UploadFile = File(..., description="图片文件：JPG、PNG、WebP 或 BMP"),
    confidence: float | None = Form(None, description="最低置信度，取值 0-1"),
    iou: float | None = Form(None, description="NMS IoU 阈值，取值 0-1"),
    model: str | None = Form(None, description="模型别名，例如 yolov8n"),
    device: str | None = Form(None, description="运行设备：auto | cpu | cuda:0 | mps"),
    max_size: int | None = Form(None, description="推理前把长边缩放到该尺寸"),
    classes: str | None = Form(None, description="要保留的类别 id，以逗号分隔"),
    include_image: bool = Form(True, description="以 data URI 形式内嵌标注后的图片"),
    image_format: str = Form("jpeg", description="jpeg（体积小）或 png（无损）"),
) -> dict[str, Any]:
    """
    检测上传图片中的目标。

    与文档中给出的 curl 调用一致：

        curl -X POST http://localhost:8000/detect \\
          -F 'file=@test_image.jpg' -F 'confidence=0.5' -F 'model=yolov8n'
    """
    if image_format.lower() not in {"jpeg", "jpg", "png"}:
        raise HTTPException(status_code=400, detail="image_format 只能是 'jpeg' 或 'png'。")
    return await _run_detection(
        file, confidence, iou, model, device, max_size, classes,
        annotate=include_image, image_format=image_format,
    )


@app.post(
    "/detect/image",
    tags=["检测"],
    summary="执行目标检测并返回标注后的 PNG 图片",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def detect_image(
    file: UploadFile = File(..., description="图片文件：JPG、PNG、WebP 或 BMP"),
    confidence: float | None = Form(None),
    iou: float | None = Form(None),
    model: str | None = Form(None),
    device: str | None = Form(None),
    max_size: int | None = Form(None),
    classes: str | None = Form(None),
) -> Response:
    """检测逻辑与 /detect 相同，只是响应体本身就是标注后的 PNG 图片。"""
    payload = await _run_detection(
        file, confidence, iou, model, device, max_size, classes,
        annotate=True, image_format="png",
    )
    raw = base64.b64decode(payload["annotated_image"].split(",", 1)[1])
    headers = {
        "X-Detection-Count": payload["count"],
        "X-Inference-Ms": payload["inference_ms"],
        "X-Model": payload["model"],
        "X-Device": payload["device"],
    }
    return Response(content=raw, media_type="image/png", headers=_ascii_headers(headers))


def _ascii_headers(headers: dict[str, Any]) -> dict[str, str]:
    """HTTP headers must be latin-1 encodable; keep them strictly ASCII."""
    return {k: str(v).encode("ascii", "replace").decode("ascii") for k, v in headers.items()}


# ============================================================
# STREAMING DETECTION (progress feed)
# ============================================================

def _progress_line(event: dict[str, Any]) -> bytes:
    """One NDJSON record. `ensure_ascii=False` keeps the Chinese stage labels readable."""
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@app.post(
    "/detect/stream",
    tags=["检测"],
    summary="执行目标检测并以 NDJSON 流返回推理进度与结果",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/x-ndjson": {}}}},
)
async def detect_stream(
    file: UploadFile = File(..., description="图片文件：JPG、PNG、WebP 或 BMP"),
    confidence: float | None = Form(None, description="最低置信度，取值 0-1"),
    iou: float | None = Form(None, description="NMS IoU 阈值，取值 0-1"),
    model: str | None = Form(None, description="模型别名，例如 yolov8n"),
    device: str | None = Form(None, description="运行设备：auto | cpu | cuda:0 | mps"),
    max_size: int | None = Form(None, description="推理前把长边缩放到该尺寸"),
    classes: str | None = Form(None, description="要保留的类别 id，以逗号分隔"),
    include_image: bool = Form(True, description="以 data URI 形式内嵌标注后的图片"),
    image_format: str = Form("jpeg", description="jpeg（体积小）或 png（无损）"),
) -> StreamingResponse:
    """
    与 `POST /detect` 结果完全相同，但推理期间会把每个流水线阶段作为一行 JSON
    实时推回，客户端据此绘制进度条与耗时曲线：

    ```
    {"type":"progress","stage":"resized","label":"缩放预处理","elapsed_ms":12.4,...}
    ...
    {"type":"result","count":3,"inference_ms":41.2,...}
    ```

    事件类型只有两种：`progress`（可重复）与 `result`（最后一条）。出错时推一条
    `{"type":"error","message":...}`；HTTP 状态码在流开始前就已确定，因此客户端
    必须读错误事件，而不能只看状态码。
    """
    image_format = (image_format or "jpeg").lower()
    if image_format not in {"jpeg", "jpg", "png"}:
        raise HTTPException(status_code=400, detail="image_format 只能是 'jpeg' 或 'png'。")

    # Validated before the response starts, so bad input still yields a normal 4xx.
    data = await _read_upload(file)
    try:
        image = det.decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    class_filter: list[int] | None = None
    if classes:
        try:
            class_filter = [int(part) for part in classes.replace(" ", "").split(",") if part]
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="classes 必须是逗号分隔的类别 id 列表。"
            ) from exc

    return StreamingResponse(
        _detect_event_stream(
            image=image,
            data_size=len(data),
            filename=file.filename or "image",
            confidence=confidence,
            iou=iou,
            model=model,
            device=device,
            max_size=max_size,
            classes=class_filter,
            annotate=include_image,
            image_format=image_format,
        ),
        media_type="application/x-ndjson",
        headers={
            # Progress is worthless if a proxy buffers the whole body first.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _detect_event_stream(**kwargs: Any) -> AsyncIterator[bytes]:
    """
    Bridge the synchronous, generator-based detector into an async byte stream.

    Inference is blocking and single-threaded (one shared GPU), so the generator is
    advanced inside a worker thread while this coroutine only awaits the events it
    produces. Events go into a plain list guarded by an asyncio.Event rather than a
    bounded Queue: the producer can then always make progress, so a slow or stalled
    reader can never wedge the worker holding the global inference lock (the event
    count is bounded by the pipeline stages, so the buffer cannot grow without
    limit either).
    """
    loop = asyncio.get_running_loop()
    buffer: list[tuple[str, Any]] = []
    wakeup = asyncio.Event()
    cancelled = threading.Event()

    def publish(kind: str, payload: Any) -> None:
        # Called from the worker thread; call_soon_threadsafe is the only safe hop.
        def deliver() -> None:
            buffer.append((kind, payload))
            wakeup.set()

        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:
            # The event loop is gone (server shutting down mid-request).
            cancelled.set()

    def produce() -> None:
        started = time.perf_counter()
        try:
            for event in det.detect_stream(
                kwargs["image"],
                model=kwargs["model"],
                confidence=kwargs["confidence"],
                iou=kwargs["iou"],
                device=kwargs["device"],
                max_size=kwargs["max_size"],
                annotate=kwargs["annotate"],
                classes=kwargs["classes"],
                image_format=kwargs["image_format"],
            ):
                if cancelled.is_set():
                    return
                if event.get("type") == "result":
                    event = {
                        **event,
                        "success": True,
                        "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
                        "summary": det.summarise(event["detections"]),
                        "source": {"filename": kwargs["filename"], "bytes": kwargs["data_size"]},
                    }
                publish("event", event)
        except (det.ModelError, ValueError) as exc:
            publish("error", str(exc))
        except Exception as exc:  # pragma: no cover - unexpected runtime failure
            logger.exception("Streaming detection failed")
            publish("error", f"检测失败：{exc}")
        finally:
            publish("end", None)

    worker = threading.Thread(target=produce, name="yolo-detect-stream", daemon=True)
    worker.start()

    try:
        while True:
            if not buffer:
                wakeup.clear()
                await wakeup.wait()
                continue
            while buffer:
                kind, payload = buffer.pop(0)
                if kind == "end":
                    return
                if kind == "error":
                    # The status line is already sent, so the failure must travel
                    # in-band; the UI reads this event instead of an HTTP code.
                    yield _progress_line(
                        {"type": "error", "success": False, "message": payload}
                    )
                    continue
                yield _progress_line(payload)
    finally:
        # A client disconnect lands here. Setting the flag makes the producer stop
        # at the next event boundary and drop out of _infer_lock, instead of
        # holding the single-GPU inference lock for a reader that has gone away.
        cancelled.set()


if __name__ == "__main__":  # pragma: no cover - convenience only
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
