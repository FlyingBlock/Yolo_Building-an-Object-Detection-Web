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
from typing import Any, Iterable, Sequence

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

    conf = CONFIDENCE_THRESHOLD if confidence is None else float(confidence)
    iou_threshold = IOU_THRESHOLD if iou is None else float(iou)
    if not 0.0 < conf <= 1.0:
        raise ValueError("confidence 必须位于 (0, 1] 区间内。")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou 必须位于 (0, 1] 区间内。")

    limit = MAX_IMAGE_SIZE if max_size is None else int(max_size)
    original_h, original_w = image.shape[:2]
    scale = 1.0
    work = image
    if limit and max(original_h, original_w) > limit:
        scale = limit / float(max(original_h, original_w))
        work = cv2.resize(
            image,
            (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    active_device = resolve_device(device)
    yolo = load_model(model, active_device)

    started = time.perf_counter()
    with _infer_lock:
        results = yolo.predict(
            source=work,
            conf=conf,
            iou=iou_threshold,
            device=active_device,
            verbose=False,
            classes=list(classes) if classes is not None else None,
        )
    inference_ms = (time.perf_counter() - started) * 1000.0

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

    counts: dict[str, int] = {}
    for det in detections:
        counts[det["class_name"]] = counts.get(det["class_name"], 0) + 1

    payload: dict[str, Any] = {
        "model": resolve_model_ref(model),
        "device": active_device,
        "confidence_threshold": conf,
        "iou_threshold": iou_threshold,
        "image": {"width": int(original_w), "height": int(original_h), "inferred_at_scale": round(scale, 4)},
        "inference_ms": round(inference_ms, 1),
        "count": len(detections),
        "counts_by_class": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "detections": detections,
    }
    if annotate:
        annotated = draw_detections(image, detections)
        fmt = (image_format or "jpeg").lower()
        mime = "image/png" if fmt == "png" else "image/jpeg"
        payload["annotated_image"] = to_data_uri(
            encode_image(annotated, fmt, image_quality), mime
        )
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
