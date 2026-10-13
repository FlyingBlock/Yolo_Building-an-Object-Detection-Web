"""
Shared YOLO detection core for the Object Detection Web App.

This module is the single source of truth for model loading, inference and
annotation. It is imported by every entry point so that behaviour is identical
whether you call the REST API, the Streamlit UI, or a CLI script:

    api.py            FastAPI backend   ->  uvicorn api:app --reload --port 8000
    app.py            Streamlit frontend ->  streamlit run app.py
    detect.py         single image CLI
    batch_detect.py   folder batch CLI

Nothing in here performs I/O on import beyond reading the optional .env file, so
importing this module is cheap.
"""

from __future__ import annotations

import base64
import colorsys
import io
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# CONFIGURATION  (see .env.example); a tiny .env loader keeps the app
# dependency-free -- python-dotenv is not required.
# ============================================================

def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv(PROJECT_ROOT / ".env")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.5)
IOU_THRESHOLD = _env_float("IOU_THRESHOLD", 0.45)
DEVICE = os.environ.get("DEVICE", "auto")
MAX_IMAGE_SIZE = _env_int("MAX_IMAGE_SIZE", 1280)
MAX_UPLOAD_MB = _env_float("MAX_UPLOAD_MB", 25.0)
MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 64_000_000)
PRELOAD_MODEL = _env_flag("PRELOAD_MODEL", False)

# Loading a .pt file executes pickle data, so an upload endpoint that accepts
# .pt files is a remote-code-execution surface. It is enabled by default because
# the API binds to loopback and custom models are an advertised feature, but it
# must be switched off before exposing this API to a network you do not control.
ALLOW_CUSTOM_MODEL_UPLOAD = _env_flag("ALLOW_CUSTOM_MODEL_UPLOAD", True)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))

# Only these short names trigger an automatic Ultralytics weight download.
BUILTIN_MODELS: dict[str, str] = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
}

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.pt$")
_DOWNLOADABLE = re.compile(r"^yolo(?:v8|11|12)[nsmlx]\.pt$")

# COCO class names, used only as a fallback if a model exposes none.
_FALLBACK_NAMES = "person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors teddy bear hair drier toothbrush".split()


class ModelError(ValueError):
    """Raised for an unknown, unsafe or missing model reference."""


# ============================================================
# DEVICE + MODEL CACHE
# ============================================================

def resolve_device(device: str | None = None) -> str:
    """Turn 'auto' into a concrete torch device string."""
    requested = (device or DEVICE or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch always present in practice
        pass
    return "cpu"


def resolve_model_ref(ref: str | None) -> str:
    """
    Map a user/model reference to something Ultralytics can load.

    Accepts a builtin alias ('yolov8n'), a builtin filename ('yolov8s.pt'), or
    the name of a file inside MODELS_DIR. Path separators and absolute paths are
    rejected so that an API caller cannot point the loader at arbitrary files.
    """
    raw = (ref or "").strip() or DEFAULT_MODEL
    if raw in BUILTIN_MODELS:
        return BUILTIN_MODELS[raw]

    name = Path(raw).name
    if name != raw or not _SAFE_FILENAME.match(name):
        raise ModelError(f"不支持的模型引用：{raw!r}")

    candidate = MODELS_DIR / name
    if candidate.is_file():
        return str(candidate)

    if name in BUILTIN_MODELS.values() or _DOWNLOADABLE.match(name):
        return name

    raise ModelError(
        f"未知模型 {raw!r}。请使用以下之一：{sorted(BUILTIN_MODELS)}，"
        f"或先把 .pt 文件上传到 models 目录。"
    )


_models: dict[str, Any] = {}
_models_lock = threading.RLock()
# Ultralytics keeps a reusable predictor on the model object, so two threads
# calling predict() on one instance can race. Inference is serialised; on a
# single GPU that is the correct behaviour anyway.
_infer_lock = threading.RLock()


def load_model(ref: str | None = None, device: str | None = None):
    """
    Return a cached Ultralytics YOLO model (thread-safe, one instance per path).

    The first call for a given model downloads weights if needed and can take a
    few seconds; later calls are free.
    """
    from ultralytics import YOLO

    resolved = resolve_model_ref(ref)
    key = f"{resolved}::{resolve_device(device)}"
    with _models_lock:
        model = _models.get(key)
        if model is None:
            model = YOLO(resolved)
            _models[key] = model
        return model


def warmup(ref: str | None = None, device: str | None = None) -> dict[str, Any]:
    """
    Load the model and run one throwaway inference.

    The first CUDA inference on a fresh process pays for context creation and
    cuDNN kernel selection (measured at ~16 s on an RTX 5060 Ti versus ~20 ms
    afterwards). Doing it here keeps that cost out of the user's first request.
    """
    import numpy as np

    started = time.perf_counter()
    model = load_model(ref, device)
    load_ms = (time.perf_counter() - started) * 1000.0

    probe = np.zeros((320, 320, 3), dtype=np.uint8)
    infer_started = time.perf_counter()
    model.predict(source=probe, conf=0.9, device=resolve_device(device), verbose=False)
    infer_ms = (time.perf_counter() - infer_started) * 1000.0

    names = getattr(model, "names", None) or {}
    return {
        "model": resolve_model_ref(ref),
        "device": resolve_device(device),
        "classes": len(names) or len(_FALLBACK_NAMES),
        "load_ms": round(load_ms, 1),
        "first_inference_ms": round(infer_ms, 1),
        "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def cached_models() -> list[dict[str, Any]]:
    with _models_lock:
        return [
            {"ref": key.split("::", 1)[0], "device": key.split("::", 1)[1]}
            for key in _models
        ]


def available_models() -> dict[str, Any]:
    """Builtin aliases plus any user-supplied weights already on disk."""
    custom: list[str] = []
    if MODELS_DIR.is_dir():
        custom = sorted(
            p.name for p in MODELS_DIR.glob("*.pt") if _SAFE_FILENAME.match(p.name)
        )
    return {
        "builtin": sorted(BUILTIN_MODELS),
        "custom": custom,
        "cached": cached_models(),
        "default": DEFAULT_MODEL,
    }


def save_custom_model(filename: str, data: bytes) -> str:
    """Persist an uploaded .pt into MODELS_DIR and return the safe basename."""
    if not ALLOW_CUSTOM_MODEL_UPLOAD:
        raise ModelError(
            "自定义模型上传功能已关闭。若要启用，请设置 ALLOW_CUSTOM_MODEL_UPLOAD=1。"
        )
    name = Path(filename or "").name
    if not _SAFE_FILENAME.match(name):
        raise ModelError(
            "模型文件必须是 .pt 文件，且文件名只能由字母、数字、点、下划线或连字符组成。"
        )
    if not data:
        raise ModelError("上传的模型文件为空。")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / name
    target.write_bytes(data)
    return name


# ============================================================
# IMAGE HANDLING
# ============================================================

def decode_image(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into a contiguous RGB uint8 array."""
    if not data:
        raise ValueError("图片数据为空。")
    try:
        with Image.open(io.BytesIO(data)) as img:
            # A small compressed file can still decode to an enormous bitmap, so
            # reject by pixel count before allocating anything.
            pixels = img.size[0] * img.size[1]
            if pixels > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"图片过大，无法处理：{pixels / 1e6:.1f} MP 超过上限 "
                    f"{MAX_IMAGE_PIXELS / 1e6:.0f} MP（MAX_IMAGE_PIXELS）。"
                )
            img = img.convert("RGB")
            array = np.asarray(img, dtype=np.uint8)
    except ValueError:
        raise
    except Exception as exc:
        detail = str(exc).strip()
        if (
            isinstance(exc, ModuleNotFoundError)
            or not detail
            # Pillow's own sniffing failure, which would otherwise leak its English
            # text (plus a BytesIO repr) into the message the UI displays.
            or detail.startswith("cannot identify image file")
        ):
            # Pillow probes optional plugins during format sniffing and surfaces a
            # missing optional dependency instead of "unrecognised file".
            detail = "无法将其识别为图片"
        raise ValueError(
            f"图片格式不受支持或文件已损坏（{detail}）。"
            "请上传 JPG、PNG、WebP 或 BMP 文件。"
        ) from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("图片必须包含三个颜色通道。")
    return np.ascontiguousarray(array)


def encode_image(array: np.ndarray, fmt: str = "png", quality: int = 90) -> bytes:
    """
    Encode an RGB uint8 array as image bytes.

    PNG is lossless but heavy for a photo (a 1080p frame is ~2.6 MB), which
    matters when the image travels inline in a JSON response. JPEG at quality 90
    is roughly a tenth of that with no visible penalty on annotation overlays,
    so the API uses JPEG for its data URI and PNG for the raw-bytes endpoint.
    """
    buffer = io.BytesIO()
    image = Image.fromarray(array)
    if fmt.lower() in {"jpg", "jpeg"}:
        image.save(buffer, format="JPEG", quality=int(quality), optimize=True, progressive=True)
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_png(array: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG bytes."""
    return encode_image(array, "png")


def to_data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def class_color_rgb(class_id: int) -> tuple[int, int, int]:
    """Stable, well-separated colour per class id (golden-ratio hue rotation)."""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_detections(rgb: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
    """
    Draw colour-coded boxes, class labels and confidence scores onto a copy of
    the image. Drawn in RGB channel order, so the result displays correctly.
    """
    canvas = np.ascontiguousarray(rgb.copy())
    height, width = canvas.shape[:2]
    scale = max(1.0, min(width, height) / 640.0)
    thickness = max(1, int(round(1.8 * scale)))
    font_scale = 0.55 * scale
    font_thickness = max(1, int(round(1.1 * scale)))

    for det in detections:
        box = det["box"]
        x1, y1 = int(round(box["x1"])), int(round(box["y1"]))
        x2, y2 = int(round(box["x2"])), int(round(box["y2"]))
        colour = class_color_rgb(int(det["class_id"]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, lineType=cv2.LINE_AA)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        # Keep the label inside the frame.
        top = y1 - text_h - baseline - 4
        if top < 0:
            top = min(y1 + 2, max(0, height - text_h - baseline - 4))
        right = min(x1 + text_w + 6, width)
        cv2.rectangle(canvas, (x1, top), (right, top + text_h + baseline + 4), colour, -1)
        # Pick black or white text depending on the label background luminance.
        luma = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
        text_colour = (0, 0, 0) if luma > 140 else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_colour,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas


# ============================================================
# DETECTION
# ============================================================

# ---- progress reporting -------------------------------------------------
#
# Inference itself is a single opaque call into Ultralytics, so there is no
# per-epoch loss curve to show for a single image. What *can* be reported honestly
# is the pipeline: decode -> resize -> load -> infer -> annotate, with wall-clock
# time and (when CUDA is in play) utilisation/VRAM sampled at each boundary, plus
# the detector's own view of the image it is about to look at. detect_stream()
# emits one event per boundary; detect() is the same code with the reporting
# switched off, which is why both share _detect_locked().

# Order matters: it is the real execution order, so a consumer can trust
# `stage_index` to advance monotonically, and it is the order stages appear in the
# progress strip and in the timing curve. Keys are wire values; labels are Chinese
# because they are rendered directly in the UI and in the JSON event stream.
STAGE_LABELS: dict[str, str] = {
    "queued": "排队",
    "received": "接收图片",
    "resized": "缩放预处理",
    "model_loaded": "加载权重",
    "inferred": "模型推理",
    "decoded": "解析检测框",
    "annotated": "绘制标注",
    "encoded": "编码输出",
    "done": "完成",
}
STAGE_ORDER: tuple[str, ...] = tuple(STAGE_LABELS)

ProgressCallback = Callable[[dict[str, Any]], None]

# The sink the pipeline reports to. It is a module global rather than a parameter
# because _detect_locked() is reached through two public entry points that must
# behave identically; _set_emit() swaps it for the duration of a stream. The
# default swallows both positional and keyword extras so a plain detect() call
# reports to nobody without paying for any bookkeeping.
_active_emit: ProgressCallback = lambda *args, **kwargs: None  # noqa: E731


def _set_emit(emit: ProgressCallback) -> ProgressCallback:
    """Install a progress sink; returns the previous one so callers can restore it."""
    global _active_emit
    previous, _active_emit = _active_emit, emit
    return previous


def _emit_stage(emit: ProgressCallback, stage: str, **extra: Any) -> None:
    """Thin indirection so the pipeline has exactly one reporting call site shape."""
    emit(stage, **extra)


def _new_progress_callback(sink: ProgressCallback, sync_gpu: bool = False) -> ProgressCallback:
    """
    Wrap a user sink so every emitted event also carries timing and hardware data.

    `time_ms` is measured from the first event; `elapsed_ms` is the same clock, so
    a consumer can plot either the per-stage delta or the cumulative curve.
    GPU numbers are sampled on demand and skipped entirely when CUDA is absent.
    """
    started = time.perf_counter()

    def _emit(stage: str, **extra: Any) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        event: dict[str, Any] = {
            "stage": stage,
            "label": STAGE_LABELS.get(stage, stage),
            "stage_index": STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1,
            "stage_total": len(STAGE_ORDER),
            "elapsed_ms": round(elapsed_ms, 1),
            "time_ms": round(elapsed_ms, 1),
        }
        if sync_gpu:
            event.update(_gpu_snapshot())
        if extra:
            event.update(extra)
        try:
            sink(event)
        except Exception:  # a broken observer inside a worker thread would
            pass          # otherwise take the whole detection down with it

    return _emit


def _gpu_snapshot() -> dict[str, Any]:
    """Best-effort CUDA utilisation/VRAM sample; empty dict on failure or no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        snapshot: dict[str, Any] = {
            "vram_used_mb": round((total - free) / 1024 / 1024, 1),
            "vram_total_mb": round(total / 1024 / 1024, 1),
            "vram_util_pct": round(100.0 * (total - free) / total, 1) if total else 0.0,
        }
        try:  # nvidia-ml-py is optional; recent torch ships it, older ones do not
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snapshot["gpu_util_pct"] = int(rates.gpu)
            snapshot["gpu_mem_util_pct"] = int(rates.memory)
        except Exception:
            pass
        return snapshot
    except Exception:  # pragma: no cover - depends on the local CUDA stack
        return {}


def _image_metrics(rgb: np.ndarray) -> dict[str, Any]:
    """
    Cheap statistics describing the frame the detector is about to look at.

    These are what the UI plots live while inference runs, and they are measured on
    the (possibly downscaled) working image, so they double as a sanity check that
    preprocessing did what the caller asked for.
    """
    height, width = rgb.shape[:2]
    try:
        # A 64x64 thumbnail keeps this in the sub-millisecond range on any input.
        small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        luma = (
            0.299 * small[..., 0].astype(np.float32)
            + 0.587 * small[..., 1].astype(np.float32)
            + 0.114 * small[..., 2].astype(np.float32)
        )
        brightness = float(luma.mean())
        contrast = float(luma.std())
        sharpness = float(cv2.Laplacian(luma, cv2.CV_32F).var())
        # Mean colour cast: how far each channel sits from neutral grey.
        cast = [round(float(channel), 1) for channel in small.reshape(-1, 3).mean(axis=0)]
    except Exception:  # pragma: no cover - defensive only
        brightness = contrast = sharpness = 0.0
        cast = [0.0, 0.0, 0.0]

    megapixels = (width * height) / 1e6
    return {
        "width": int(width),
        "height": int(height),
        "megapixels": round(megapixels, 3),
        "aspect": round(width / height, 3) if height else 0.0,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "rgb_mean": cast,
    }


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)) and names:
        return {i: str(v) for i, v in enumerate(names)}
    return {i: name for i, name in enumerate(_FALLBACK_NAMES)}


def detect(
    image: np.ndarray,
    model: str | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    device: str | None = None,
    max_size: int | None = None,
    annotate: bool = True,
    classes: Iterable[int] | None = None,
    image_format: str = "jpeg",
    image_quality: int = 90,
) -> dict[str, Any]:
    """
    Run detection on an RGB uint8 array.

    A large input is downscaled for inference only; reported and drawn boxes are
    always in original-image pixel coordinates. Returns a plain dict so it can be
    JSON-serialised directly by the API.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("期望传入形状为 (H, W, 3) 的 RGB 图像数组。")

    conf = _check_unit_interval("confidence", confidence, CONFIDENCE_THRESHOLD)
    iou_threshold = _check_unit_interval("iou", iou, IOU_THRESHOLD)

    work, scale = _fit_for_inference(image, max_size)
    # Same lock, same span as before this module gained a streaming twin: load plus
    # predict, so a stream and a plain call can never interleave on one GPU.
    with _infer_lock:
        return _detect_locked(
            image=image,
            work=work,
            scale=scale,
            model=model,
            conf=conf,
            iou_threshold=iou_threshold,
            device=device,
            annotate=annotate,
            classes=classes,
            image_format=image_format,
            image_quality=image_quality,
        )


def detect_stream(
    image: np.ndarray,
    model: str | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    device: str | None = None,
    max_size: int | None = None,
    annotate: bool = True,
    classes: Iterable[int] | None = None,
    image_format: str = "jpeg",
    image_quality: int = 90,
) -> Iterator[dict[str, Any]]:
    """
    Same detection as detect(), but reporting each pipeline stage as it completes.

    Yields progress events shaped like::

        {"type": "progress", "stage": "resized", "label": "缩放预处理",
         "stage_index": 3, "stage_total": 9,
         "elapsed_ms": 12.4, "time_ms": 12.4,
         "width": 1280, "height": 853, "megapixels": 1.09,
         "brightness": 118.2, "contrast": 47.1, "sharpness": 903.4,
         "gpu_util_pct": 37, "vram_used_mb": 812.5, ...}

    and finishes with a single ``{"type": "result", ...}`` event carrying exactly
    the dict detect() would have returned. The caller is expected to consume the
    generator to exhaustion (or close it, which stops the pipeline at the next
    stage boundary and frees the inference lock).

    The whole pipeline -- including the model load -- runs inside the same
    ``_infer_lock`` detect() takes, so a stream and a plain call can never overlap
    on one GPU.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("期望传入形状为 (H, W, 3) 的 RGB 图像数组。")
    conf = _check_unit_interval("confidence", confidence, CONFIDENCE_THRESHOLD)
    iou_threshold = _check_unit_interval("iou", iou, IOU_THRESHOLD)

    queue: list[dict[str, Any]] = []
    # GPU sampling only makes sense when we are about to run on a CUDA device, and
    # resolve_device(None) is cheap relative to everything that follows.
    sync_gpu = resolve_device(device).startswith("cuda")

    with _infer_lock:
        emit = _new_progress_callback(queue.append, sync_gpu=sync_gpu)
        restore_emit = _set_emit(emit)

        def drain() -> Iterator[dict[str, Any]]:
            while queue:
                event = dict(queue.pop(0))
                event.setdefault("type", "progress")
                yield event

        try:
            emit("queued")

            work, scale = _fit_for_inference(image, max_size)
            yield from drain()

            payload = _detect_locked(
                image=image,
                work=work,
                scale=scale,
                model=model,
                conf=conf,
                iou_threshold=iou_threshold,
                device=device,
                annotate=annotate,
                classes=classes,
                image_format=image_format,
                image_quality=image_quality,
            )
            yield from drain()
            yield {"type": "result", **payload}
        finally:
            _set_emit(restore_emit)


def _check_unit_interval(name: str, value: float | None, default: float) -> float:
    resolved = default if value is None else float(value)
    if not 0.0 < resolved <= 1.0:
        raise ValueError(f"{name} 必须位于 (0, 1] 区间内。")
    return resolved


def _fit_for_inference(image: np.ndarray, max_size: int | None) -> tuple[np.ndarray, float]:
    """Downscale the long edge to `max_size`; return (working image, scale factor)."""
    limit = MAX_IMAGE_SIZE if max_size is None else int(max_size)
    original_h, original_w = image.shape[:2]
    if not limit or max(original_h, original_w) <= limit:
        return image, 1.0
    scale = limit / float(max(original_h, original_w))
    work = cv2.resize(
        image,
        (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return work, scale


def _detect_locked(
    *,
    image: np.ndarray,
    work: np.ndarray,
    scale: float,
    model: str | None,
    conf: float,
    iou_threshold: float,
    device: str | None,
    annotate: bool,
    classes: Iterable[int] | None,
    image_format: str,
    image_quality: int,
) -> dict[str, Any]:
    """
    The detection pipeline itself, reporting through the active progress sink.

    detect() and detect_stream() both land here, so the events a caller sees and
    the payload it receives can never drift apart. Preprocessing decisions (the
    resize, the thresholds) are resolved by the caller; this function only runs the
    stages and reports them.

    The caller must hold ``_infer_lock``.
    """
    emit = _active_emit
    original_h, original_w = image.shape[:2]
    _emit_stage(emit, "received", source_bytes=None)

    _emit_stage(
        emit, "resized",
        **_image_metrics(work),
        source_width=int(original_w),
        source_height=int(original_h),
        scale=round(scale, 4),
        resized=work is not image,
    )

    active_device = resolve_device(device)
    load_started = time.perf_counter()
    yolo = load_model(model, active_device)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    _emit_stage(emit, "model_loaded", load_ms=round(load_ms, 1))

    infer_started = time.perf_counter()
    results = yolo.predict(
        source=work,
        conf=conf,
        iou=iou_threshold,
        device=active_device,
        verbose=False,
        classes=list(classes) if classes is not None else None,
    )
    inference_ms = (time.perf_counter() - infer_started) * 1000.0
    _emit_stage(emit, "inferred", inference_ms=round(inference_ms, 1))

    result = results[0]
    names = _normalise_names(getattr(result, "names", None) or getattr(yolo, "names", None))
    boxes = getattr(result, "boxes", None)

    detections: list[dict[str, Any]] = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clses = boxes.cls.detach().cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, class_id in zip(xyxy, confs, clses):
            # Undo the inference-time downscale so boxes match the original image.
            ox1, oy1, ox2, oy2 = (float(x1) / scale, float(y1) / scale,
                                  float(x2) / scale, float(y2) / scale)
            ox1 = max(0.0, min(ox1, float(original_w)))
            oy1 = max(0.0, min(oy1, float(original_h)))
            ox2 = max(0.0, min(ox2, float(original_w)))
            oy2 = max(0.0, min(oy2, float(original_h)))
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), str(class_id)),
                    "confidence": round(float(score), 4),
                    "box": {
                        "x1": round(ox1, 2),
                        "y1": round(oy1, 2),
                        "x2": round(ox2, 2),
                        "y2": round(oy2, 2),
                        "width": round(ox2 - ox1, 2),
                        "height": round(oy2 - oy1, 2),
                    },
                }
            )
            # Report every so often (and once at the end) so that a crowded frame --
            # where this walk is the slowest post-processing step -- still animates
            # the progress bar without paying for an event per box.
            if len(detections) % 16 == 0:
                _emit_stage(emit, "decoded", parsed=len(detections), total=int(len(boxes)))
    _emit_stage(emit, "decoded", parsed=len(detections), total=len(detections))

    counts: dict[str, int] = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

    scores = [det["confidence"] for det in detections]
    payload: dict[str, Any] = {
        "model": resolve_model_ref(model),
        "device": active_device,
        "confidence_threshold": conf,
        "iou_threshold": iou_threshold,
        "image": {"width": int(original_w), "height": int(original_h), "inferred_at_scale": round(scale, 4)},
        "inference_ms": round(inference_ms, 1),
        "load_ms": round(load_ms, 1),
        "preprocess": {
            "scale": round(scale, 4),
            "resized": work is not image,
            "inference_width": int(work.shape[1]),
            "inference_height": int(work.shape[0]),
        },
        "count": len(detections),
        "counts_by_class": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "detections": detections,
        "scores": {
            "mean_confidence": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "max_confidence": round(max(scores), 4) if scores else 0.0,
            "min_confidence": round(min(scores), 4) if scores else 0.0,
        },
    }
    if annotate:
        draw_started = time.perf_counter()
        annotated = draw_detections(image, detections)
        draw_ms = (time.perf_counter() - draw_started) * 1000.0
        _emit_stage(emit, "annotated", draw_ms=round(draw_ms, 1), count=len(detections))

        encode_started = time.perf_counter()
        fmt = (image_format or "jpeg").lower()
        mime = "image/png" if fmt == "png" else "image/jpeg"
        encoded = encode_image(annotated, fmt, image_quality)
        payload["annotated_image"] = to_data_uri(encoded, mime)
        payload["annotated_bytes"] = len(encoded)
        _emit_stage(
            emit, "encoded",
            draw_ms=round(draw_ms, 1),
            encode_ms=round((time.perf_counter() - encode_started) * 1000.0, 1),
            annotated_bytes=len(encoded),
        )
    _emit_stage(emit, "done", count=len(detections), inference_ms=round(inference_ms, 1))
    return payload


def summarise(detections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Per-class breakdown mirroring the README's detection summary table:
    class name, count, mean confidence and the bounding box coordinates.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(det["class_name"], []).append(det)

    rows: list[dict[str, Any]] = []
    for name, group in grouped.items():
        scores = [d["confidence"] for d in group]
        rows.append(
            {
                "class": name,
                "count": len(group),
                "mean_confidence": round(sum(scores) / len(scores), 4),
                "max_confidence": round(max(scores), 4),
                "boxes": "; ".join(
                    "{:.0f},{:.0f},{:.0f},{:.0f}".format(
                        d["box"]["x1"], d["box"]["y1"], d["box"]["x2"], d["box"]["y2"]
                    )
                    for d in group
                ),
            }
        )
    return sorted(rows, key=lambda row: (-row["count"], row["class"]))


def detections_csv(detections: Sequence[dict[str, Any]]) -> str:
    """Flat per-detection CSV (one row per object)."""
    import csv

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height"]
    )
    for det in detections:
        box = det["box"]
        writer.writerow(
            [
                det["class_id"],
                det["class_name"],
                det["confidence"],
                box["x1"],
                box["y1"],
                box["x2"],
                box["y2"],
                box["width"],
                box["height"],
            ]
        )
    return buffer.getvalue()


def is_allowed_upload(size_bytes: int) -> bool:
    return size_bytes <= MAX_UPLOAD_MB * 1024 * 1024
