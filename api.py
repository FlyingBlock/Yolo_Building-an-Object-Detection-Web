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

import base64
import contextlib
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

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


if __name__ == "__main__":  # pragma: no cover - convenience only
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
