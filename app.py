"""
Streamlit frontend for YOLO object detection.

Run it with:

    streamlit run app.py

It talks to the FastAPI backend from api.py over HTTP (start that first with
`uvicorn api:app --reload --port 8000`), and can also fall back to running the
model in-process if you would rather not start the backend -- both paths call the
same code in detector.py, so results are identical.

On Windows the easiest way to start both halves at once is the one-click launcher
at the repository root: double-click `Launch.bat`,
which sets up the environment, starts the API and this UI, and opens the browser.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import queue
import threading
import time
from typing import Any, Callable

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

APP_TITLE = "YOLO 目标检测"
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
URL_MAX_MB = 25.0
# Rough payload size assumed when the server does not send Content-Length, used
# only to keep the upload bar moving instead of snapping straight to 100%.
ASSUMED_UPLOAD_MB = 8.0

# Fallback used only if neither the API nor detector.py can be reached. The real
# list is discovered at runtime -- see load_model_catalogue() -- so weights dropped
# into models/ show up here without editing this file.
FALLBACK_MODEL_ALIASES = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l",
    "yolo11n", "yolo11s", "yolo11m",
]
MODEL_SOURCE_API = "API（GET /models）"
MODEL_SOURCE_LOCAL = "本地目录探测"

# The backend labels are UI text as well as branch conditions, so they live in one
# place: run_detection(), the sidebar radio and the API errors all key off them.
REST_BACKEND_LABEL = "REST API（uvicorn 服务）"
LOCAL_BACKEND_LABEL = "本地推理（进程内）"
BACKEND_OPTIONS = [REST_BACKEND_LABEL, LOCAL_BACKEND_LABEL]

# Inline CSS. Streamlit styles its own widgets, so only the small custom pieces
# (header row, stage chips, metric grid) need rules; the chart gridlines are the
# one Streamlit-owned element worth softening for a denser look. Defined up here
# because st.set_page_config() and its styling must run before anything renders.
CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; }

  .yolo-head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap;
               margin-bottom: 0.35rem; }
  .yolo-head .idx { display: inline-flex; align-items: center; justify-content: center;
                    min-width: 1.6rem; height: 1.6rem; padding: 0 0.3rem; border-radius: 0.5rem;
                    background: #00d4aa1f; color: #00d4aa; font-weight: 700; font-size: 0.8rem; }
  .yolo-head .name { font-weight: 600; font-size: 1rem; }
  .yolo-head .meta { color: #8892a4; font-size: 0.8rem; }
  .yolo-live { display: flex; justify-content: space-between; gap: 0.5rem; margin: 0.5rem 0 0.15rem;
               font-size: 0.82rem; }
  .yolo-live .stage { font-weight: 600; color: #00d4aa; }
  .yolo-live .clock { color: #8892a4; font-variant-numeric: tabular-nums; }
  .yolo-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.45rem; }
  .yolo-chip { font-size: 0.7rem; line-height: 1.5; padding: 0.05rem 0.45rem; border-radius: 1rem;
               border: 1px solid #8892a44d; color: #8892a4; white-space: nowrap; }
  .yolo-chip.active { border-color: #00d4aa; color: #00d4aa; background: #00d4aa1a; font-weight: 600; }
  .yolo-chip.done { border-color: #00d4aa80; color: #9fe8d8; }
  .yolo-state { font-size: 0.78rem; line-height: 1.7; color: #8892a4; margin-top: 0.2rem; }
  .yolo-state-note { display: block; margin: 0.4rem 0 0.5rem; }
  .yolo-gpu { font-size: 0.75rem; color: #8892a4; font-variant-numeric: tabular-nums;
              margin-top: 0.3rem; }
  .vega-embed .role-axis-label text { font-size: 10px; }
</style>
"""

st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# PROGRESS TRACKING
#
# The API reports the detection pipeline as a stream of stage events (see
# detector.detect_stream). Everything below turns that stream into the progress
# bar, the stage strip and the live curves shown inside each image's container.
# Progress events are plain dicts, so a job can run on a worker thread and still
# be inspected by the UI thread; nothing here touches st.* off the main thread.
# ============================================================

# Pipeline stages a single image passes through, in execution order, with the
# fraction of the bar each one completes. Shared by both backends.
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
STAGE_FRACTIONS: dict[str, float] = {
    "queued": 0.02,
    "received": 0.07,
    "resized": 0.13,
    "model_loaded": 0.20,
    "inferred": 0.80,
    "decoded": 0.87,
    "annotated": 0.94,
    "encoded": 0.98,
    "done": 1.00,
}
# Curve colours: source image vs. detected content, kept distinct from the green
# accent the app already uses for bars.
CHART_BLUE = "#4f8cff"
CHART_AMBER = "#ffb02e"
CHART_TEAL = "#00d4aa"
CHART_SLATE = "#8892a4"


def progress_reset(source: dict[str, Any] | None = None) -> dict[str, Any]:
    """A fresh, JSON-friendly progress record for one image."""
    return {
        "source": source or {},
        "events": [],
        "stages": {},
        "percent": 0.0,
        "stage": None,
        "label": "等待开始",
        "elapsed_ms": 0.0,
        "started": None,
        "done": False,
        "kind": "idle",
        "error": None,
        "result": None,
    }


def progress_capture(kind: str, record: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    """
    The sink handed to a job: one callable per (job, image) that folds each stage
    event into `record`.

    A "kind" event only rewrites the record's identity (label + coarse percent),
    which is how the upload phase reports bytes without polluting the stage data.
    Monotonic percents stop the bar from jumping backwards when a stage reports
    several sub-events.
    """
    def capture(event: dict[str, Any]) -> None:
        now = time.perf_counter()
        if record["started"] is None:
            record["started"] = now
        if event.get("kind"):
            record["kind"] = "upload"
            record["label"] = str(event.get("stage") or record["label"])
            record["percent"] = max(record["percent"], float(event.get("percent") or 0.0))
            record["elapsed_ms"] = round((now - record["started"]) * 1000.0, 1)
            if event.get("source"):
                record["source"].update(event["source"])
            return

        stage = str(event.get("stage") or "")
        events = record["events"]
        previous = events[-1]["time_ms"] if events else 0.0
        event = dict(event)
        # The server clock starts before the upload finishes, so seed the first
        # local sample from the client clock to keep the curve gap-free.
        event.setdefault("time_ms", (now - record["started"]) * 1000.0)
        event["dt_ms"] = round(max(0.0, float(event["time_ms"]) - previous), 1)
        events.append(event)

        record["kind"] = "inference"
        record["stage"] = stage or record["stage"]
        record["label"] = str(event.get("label") or STAGE_LABELS.get(stage, stage))
        record["elapsed_ms"] = round(float(event["time_ms"]), 1)
        event_percent = 100.0 * STAGE_FRACTIONS.get(stage, record["percent"] / 100.0)
        if stage == "done":
            event_percent = 100.0
        record["percent"] = max(record["percent"], event_percent)
        record["stages"][stage] = {
            key: value
            for key, value in event.items()
            if key not in {"stage", "label", "stage_index", "stage_total", "type", "dt_ms"}
        }

    return capture


def progress_schema(progress: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a progress record into the handful of scalars the UI reads."""
    record = progress or {}
    source = record.get("source") or {}
    result = record.get("result") or {}
    event = record.get("events") or [{}]
    latest = event[-1] if event else {}
    scores = result.get("scores") or {}
    stages_seen = set(record.get("stages") or {})
    return {
        "done": bool(record.get("done")),
        "kind": record.get("kind") or "idle",
        "label": record.get("label") or "等待开始",
        "stage": record.get("stage"),
        "percent": float(record.get("percent") or 0.0),
        "elapsed_ms": float(record.get("elapsed_ms") or 0.0),
        "event_count": len(record.get("events") or []),
        "source_name": source.get("name") or "",
        "source_kb": float(source.get("kb") or 0.0),
        "attempted": source.get("bytes"),
        "count": result.get("count"),
        "class_count": len(result.get("counts_by_class") or {}),
        "inference_ms": result.get("inference_ms"),
        "total_ms": result.get("total_ms"),
        "mean_conf": scores.get("mean_confidence"),
        "max_conf": scores.get("max_confidence"),
        "width": latest.get("width"),
        "height": latest.get("height"),
        "megapixels": latest.get("megapixels"),
        "brightness": latest.get("brightness"),
        "contrast": latest.get("contrast"),
        "sharpness": latest.get("sharpness"),
        "gpu": latest.get("gpu_util_pct"),
        "vram_used": latest.get("vram_used_mb"),
        "vram_total": latest.get("vram_total_mb"),
        "stages_seen": stages_seen,
        "has_gpu": any(
            item.get("gpu_util_pct") is not None for item in record.get("events") or []
        ),
        "has_result": bool(result),
        "result": result,
        "error": record.get("error"),
    }


# ============================================================
# BACKENDS
# ============================================================

class BackendError(RuntimeError):
    """A failure the user can act on: backend down, bad image, bad settings."""


def api_models(api_url: str) -> dict[str, Any]:
    """Ask the backend which models it can actually load (builtin + models/)."""
    response = requests.get(f"{api_url.rstrip('/')}/models", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("GET /models 返回了非预期的响应。")
    return payload


def api_upload_model(api_url: str, filename: str, data: bytes) -> dict[str, Any]:
    """Install a .pt into the backend's models/ directory via POST /models/upload."""
    response = requests.post(
        f"{api_url.rstrip('/')}/models/upload",
        files={"file": (filename, data, "application/octet-stream")},
        timeout=300,
    )
    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise BackendError(f"上传失败（HTTP {response.status_code}）：{detail}")
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def _cached_remote_catalogue(api_url: str) -> dict[str, Any]:
    """Ask the backend for its model list (cached ~15 s; see the sidebar refresh)."""
    return api_models(api_url)


def local_model_catalogue() -> tuple[dict[str, Any] | None, str | None]:
    """
    Ask detector.py directly what it can load: (catalogue, note).

    Imported lazily: the module pulls in torch/ultralytics/cv2, which is slow and
    pointless when the user is running against the REST backend. A broken install
    returns (None, reason) instead of taking the whole sidebar down with it.
    """
    try:
        import detector as det
    except ImportError as exc:
        return None, f"无法导入 detector.py：{exc}"
    try:
        return det.available_models(), None
    except Exception as exc:  # corrupt config, unreadable models dir, ...
        return None, f"本地模型列表读取失败：{exc}"


def load_model_catalogue(backend: str, api_url: str) -> tuple[dict[str, Any], str, str | None]:
    """
    Return (catalogue, source, note) for the model drop-down.

    The REST backend is authoritative: it lists what *it* can actually load, which
    is the process that will run the inference. If it is unreachable we fall back
    to scanning the local models/ directory so the picker still works.
    """
    if backend == REST_BACKEND_LABEL:
        try:
            return _cached_remote_catalogue(api_url), MODEL_SOURCE_API, None
        except Exception as exc:  # backend down, non-JSON reply, ...
            local, local_note = local_model_catalogue()
            if local is None:
                return {}, MODEL_SOURCE_LOCAL, (
                    f"读取模型列表失败：API 不可用（{exc}）；{local_note}。"
                )
            return local, MODEL_SOURCE_LOCAL, (
                f"API 不可用（{exc}），已改为本地扫描 `models/` 目录。"
            )

    local, local_note = local_model_catalogue()
    if local is None:
        return {}, MODEL_SOURCE_LOCAL, f"读取模型列表失败：{local_note}。"
    return local, MODEL_SOURCE_LOCAL, None


def order_models(catalogue: dict[str, Any]) -> list[str]:
    """
    Build the drop-down contents: custom weights first, then builtin aliases.

    Custom names win on collision with an alias, because resolve_model_ref() gives
    a file in models/ priority too -- so what the user picks is what gets loaded.
    """
    custom = [name for name in catalogue.get("custom") or [] if name]
    if not custom:
        custom = sorted(
            name
            for name in (catalogue.get("downloaded") or [])
            if isinstance(name, str) and name.endswith(".pt")
        )
    builtin = [name for name in catalogue.get("builtin") or [] if name]

    ordered: list[str] = []
    for name in [*custom, *builtin]:
        if name not in ordered:
            ordered.append(name)
    return ordered or list(FALLBACK_MODEL_ALIASES)


def api_health(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def api_warmup(api_url: str, model: str, device: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/warmup",
        params={"model": model, "device": device},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def detect_via_api(
    api_url: str,
    data: bytes,
    filename: str,
    model: str,
    confidence: float,
    iou: float,
    device: str,
    image_format: str,
    max_size: int,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run one image through POST /detect/stream.

    The request is read as NDJSON so every pipeline stage the backend reaches is
    handed to `on_progress` the moment it happens; the `result` event is returned.
    Falls back transparently to the plain /detect endpoint if the backend is an
    older build that does not expose the streaming route.
    """
    files = {"file": (filename, data, "application/octet-stream")}
    form = {
        "model": model,
        "confidence": str(confidence),
        "iou": str(iou),
        "device": device,
        "image_format": image_format,
        "max_size": str(max_size),
        "include_image": "true",
    }

    def report(event: dict[str, Any]) -> None:
        if on_progress is not None:
            on_progress(event)

    def upload_progress(sent: int, total: int | None) -> None:
        expected = total if total else int(ASSUMED_UPLOAD_MB * 1024 * 1024)
        percent = 1.5 * min(1.0, sent / max(1, expected))
        report(
            {
                "kind": "upload",
                "stage": f"上传图片 {sent / 1024 / 1024:.1f} MB",
                "percent": percent,
            }
        )

    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/detect/stream",
            files=files,
            data=form,
            timeout=900,
            stream=True,
        )
    except requests.exceptions.ConnectionError as exc:
        raise BackendError(
            f"无法连接位于 {api_url} 的 API 服务。请先启动它：\n\n"
            f"    Windows 一键启动：双击项目根目录的「Launch.bat」\n"
            f"    手动启动：uvicorn api:app --reload --port 8000\n\n"
            f"或在侧边栏把“推理后端”切换为“{LOCAL_BACKEND_LABEL}”。"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendError("连接 API 超时。") from exc

    if response.status_code in (404, 405):
        # Older backend without /detect/stream: keep working, just without curves.
        response.close()
        report({"kind": "upload", "stage": "后端不支持流式进度，改用普通接口", "percent": 2.0})
        return _detect_via_api_plain(api_url, data, filename, form)

    if response.status_code != 200:
        detail = response.text
        response.close()
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise BackendError(f"API 返回状态码 {response.status_code}：{detail}")

    payload: dict[str, Any] | None = None
    failure: str | None = None
    sent = 0
    total = response.headers.get("Content-Length")
    expected = int(total) if total and total.isdigit() else None

    try:
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            sent += len(line)
            upload_progress(sent, expected)
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if event.get("type") == "result":
                payload = event
            elif event.get("type") == "error":
                failure = str(event.get("message") or "推理失败。")
            else:
                report(event)
    except requests.exceptions.Timeout as exc:
        raise BackendError("API 处理这张图片时超时。") from exc
    finally:
        response.close()

    if failure:
        raise BackendError(failure)
    if payload is None:
        raise BackendError("API 流已结束，但没有返回检测结果。")
    return payload


def _detect_via_api_plain(
    api_url: str, data: bytes, filename: str, form: dict[str, str]
) -> dict[str, Any]:
    """Non-streaming fallback used when the backend predates /detect/stream."""
    response = requests.post(
        f"{api_url.rstrip('/')}/detect",
        files={"file": (filename, data, "application/octet-stream")},
        data=form,
        timeout=600,
    )
    if response.status_code != 200:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise BackendError(f"API 返回状态码 {response.status_code}：{detail}")
    return response.json()


def detect_locally(
    data: bytes,
    model: str,
    confidence: float,
    iou: float,
    device: str,
    max_size: int,
    image_format: str = "jpeg",
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the same pipeline in-process, reporting the same stage events."""
    try:
        import detector as det
    except ImportError as exc:  # ultralytics / torch not installed
        raise BackendError(f"本地推理需要先安装检测相关依赖（{exc}）。") from exc

    def report(event: dict[str, Any]) -> None:
        if on_progress is not None:
            on_progress(event)

    report(
        {
            "kind": "upload",
            "stage": f"读取本地图片 {len(data) / 1024:.0f} KB",
            "percent": 2.0,
        }
    )
    try:
        image = det.decode_image(data)
    except ValueError as exc:
        raise BackendError(str(exc)) from exc

    started = time.perf_counter()
    payload: dict[str, Any] | None = None
    try:
        for event in det.detect_stream(
            image,
            model=model,
            confidence=confidence,
            iou=iou,
            device=device,
            max_size=max_size,
            annotate=True,
            classes=None,
            image_format=image_format,
        ):
            if event.get("type") == "result":
                payload = event
            else:
                report(event)
    except (det.ModelError, ValueError) as exc:
        raise BackendError(str(exc)) from exc

    if payload is None:
        raise BackendError("本地推理没有返回结果。")
    payload["success"] = True
    payload["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    payload["summary"] = det.summarise(payload["detections"])
    return payload


# ============================================================
# HELPERS
# ============================================================

def decoded_size(data: bytes) -> tuple[int, int] | None:
    """Pixel size of encoded image bytes without fully decoding them (header only)."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


def avg_box_ratio(result: dict[str, Any]) -> float:
    """
    Mean bounding-box area as a fraction of the source frame (0..1).

    A single number that says "how much of the picture is object", which is a more
    useful glance-value than the raw pixel areas when images arrive at mixed sizes.
    """
    image = result.get("image") or {}
    area = float(image.get("width") or 0) * float(image.get("height") or 0)
    boxes = result.get("detections") or []
    if not area or not boxes:
        return 0.0
    return sum(
        float(det["box"]["width"]) * float(det["box"]["height"]) for det in boxes
    ) / (area * len(boxes))


def fetch_image_url(url: str) -> tuple[bytes, str]:
    """Download a user-supplied image URL, with scheme and size guards."""
    url = url.strip()
    if not url:
        raise BackendError("请先输入图片 URL。")
    if not url.lower().startswith(("http://", "https://")):
        raise BackendError("仅支持以 http:// 或 https:// 开头的 URL。")
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BackendError(f"无法下载该 URL：{exc}") from exc

    declared = response.headers.get("Content-Type", "")
    if declared and not declared.lower().startswith("image/"):
        raise BackendError(f"该 URL 指向的不是图片（Content-Type：{declared}）。")

    payload = response.content
    if not payload:
        raise BackendError("该 URL 返回了空响应。")
    if len(payload) > URL_MAX_MB * 1024 * 1024:
        raise BackendError(f"该图片超过 {URL_MAX_MB:.0f} MB 上限。")
    filename = url.split("?")[0].rstrip("/").split("/")[-1] or "downloaded-image"
    return payload, filename


def data_uri_to_bytes(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1])


def summary_dataframe(summary: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(summary)
    return frame.rename(
        columns={
            "class": "类别",
            "count": "数量",
            "mean_confidence": "平均置信度",
            "max_confidence": "最高置信度",
            "boxes": "检测框 (x1,y1,x2,y2)",
        }
    )


def detections_dataframe(detections: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, det in enumerate(detections, start=1):
        box = det["box"]
        rows.append(
            {
                "#": index,
                "类别": det["class_name"],
                "置信度": det["confidence"],
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "宽度": box["width"],
                "高度": box["height"],
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# PER-IMAGE CONTAINERS
#
# One container per image, stacked top to bottom. Inside a container the input and
# the annotated result sit side by side in a three-column row (each therefore about
# a third of the page wide), the third column carries that image's numbers, and the
# progress bar plus the inference curves run full width underneath -- so a whole
# image's run, live or finished, stays inside a single bordered box.
# ============================================================

# Inline CSS lives near the top of this file (see CUSTOM_CSS): it has to be
# registered before st.set_page_config() renders anything.


def render_chips(seen: set[str], active: str | None) -> str:
    """The stage strip: every pipeline step, marked done / active / pending."""
    reached = False
    chips = []
    for stage in STAGE_ORDER:
        if stage == active:
            mark, reached = "active", True
        elif stage in seen or reached:
            mark = "done"
        else:
            mark = "pending"
        chips.append(f'<span class="yolo-chip {mark}">{STAGE_LABELS[stage]}</span>')
    return f'<div class="yolo-chips">{"".join(chips)}</div>'


def metric_tiles(items: list[tuple[str, Any]]) -> str:
    """Rows of label/value HTML; values are already formatted by the caller."""
    rows = []
    for label, value in items:
        shown = "—" if value is None or value == "" else value
        rows.append(
            f'<div><span style="display:inline-block;min-width:7.5rem;color:#8892a4;'
            f'font-size:0.8rem;">{label}</span>'
            f'<span style="font-variant-numeric:tabular-nums;">{shown}</span></div>'
        )
    return "".join(rows)


def timing_frame(progress: dict[str, Any]) -> pd.DataFrame:
    """Per-stage time deltas for the timing bars (measured between consecutive events)."""
    rows = []
    for event in progress.get("events") or []:
        rows.append(
            {
                "阶段": event.get("label") or STAGE_LABELS.get(str(event.get("stage")), "?"),
                "耗时(ms)": round(float(event.get("dt_ms") or 0.0), 1),
                "累计(ms)": round(float(event.get("time_ms") or 0.0), 1),
            }
        )
    return pd.DataFrame(rows)


def hardware_frame(progress: dict[str, Any]) -> pd.DataFrame:
    """GPU/VRAM samples over the run, long-form so one line mark can draw them all."""
    rows = []
    for event in progress.get("events") or []:
        if event.get("gpu_util_pct") is None and event.get("vram_used_mb") is None:
            continue
        moment = round(float(event.get("time_ms") or 0.0), 1)
        for key, label, unit in (
            ("gpu_util_pct", "GPU 使用率", "%"),
            ("gpu_mem_util_pct", "显存带宽占用", "%"),
            ("vram_used_mb", "显存占用", "MB"),
            ("vram_total_mb", "显存总量", "MB"),
        ):
            value = event.get(key)
            if value is not None:
                rows.append(
                    {"运行时刻(ms)": moment, "指标": label, "数值": float(value), "单位": unit}
                )
    return pd.DataFrame(rows)


def timing_chart(frame: pd.DataFrame) -> alt.Chart:
    """Horizontal bars: where the milliseconds of this image actually went."""
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=2, size=14)
        .encode(
            y=alt.Y("阶段:N", sort=list(frame["阶段"]), title=None,
                    axis=alt.Axis(labelLimit=110)),
            x=alt.X("耗时(ms):Q", title="耗时 (ms)"),
            color=alt.Color("耗时(ms):Q", scale=alt.Scale(scheme="teals"), legend=None),
            tooltip=["阶段", "耗时(ms)", "累计(ms)"],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=3, fontSize=10, color="#b6bdc9")
        .encode(
            y=alt.Y("阶段:N", sort=list(frame["阶段"]), title=None),
            x=alt.X("耗时(ms):Q"),
            text=alt.Text("耗时(ms):Q", format=".0f"),
        )
    )
    return (
        (bars + labels)
        .properties(height=max(150, 26 * len(frame) + 40))
        .configure_view(stroke=None)
        .configure_axis(grid=False, domainColor="#8892a466", tickColor="#8892a466")
    )


def hardware_chart(frame: pd.DataFrame) -> alt.Chart:
    """GPU utilisation, memory-bandwidth use and VRAM over the life of one inference."""
    percentage = frame[frame["单位"] == "%"]
    megabytes = frame[frame["单位"] == "MB"]
    layers = []
    if not percentage.empty:
        layers.append(
            alt.Chart(percentage)
            .mark_line(point=alt.OverlayMarkDef(size=24), strokeWidth=2)
            .encode(
                x=alt.X("运行时刻(ms):Q", title="运行时刻 (ms)"),
                y=alt.Y("数值:Q", title="占用 (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "指标:N",
                    scale=alt.Scale(domain=["GPU 使用率", "显存带宽占用"],
                                    range=[CHART_BLUE, CHART_TEAL]),
                    legend=alt.Legend(orient="top", title=None, columns=2),
                ),
                tooltip=["运行时刻(ms)", "指标", alt.Tooltip("数值:Q", format=".1f")],
            )
        )
    if not megabytes.empty:
        layers.append(
            alt.Chart(megabytes)
            .mark_line(strokeDash=[4, 3], strokeWidth=1.6)
            .encode(
                x=alt.X("运行时刻(ms):Q"),
                y=alt.Y("数值:Q", title="显存 (MB)"),
                color=alt.Color(
                    "指标:N",
                    scale=alt.Scale(domain=["显存占用", "显存总量"],
                                    range=[CHART_AMBER, CHART_SLATE]),
                    legend=alt.Legend(orient="top", title=None, columns=2),
                ),
                tooltip=["运行时刻(ms)", "指标", alt.Tooltip("数值:Q", format=".0f")],
            )
        )
    return (
        alt.layer(*layers)
        .resolve_scale(y="independent")
        .properties(height=240)
        .configure_view(stroke=None)
        .configure_axis(grid=True, gridColor="#8892a433", domainColor="#8892a466")
    )


def confidence_chart(result: dict[str, Any], threshold: float) -> alt.Chart | None:
    """Confidence distribution of the kept detections, with the cut-off marked."""
    scores = [float(det["confidence"]) for det in result.get("detections") or []]
    if not scores:
        return None
    frame = pd.DataFrame({"置信度": scores})
    return (
        alt.Chart(frame)
        .mark_bar(color=CHART_TEAL, binSpacing=2, opacity=0.85)
        .encode(
            x=alt.X("置信度:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="置信度"),
            y=alt.Y("count():Q", title="目标数量"),
            tooltip=[alt.Tooltip("置信度:Q", bin=alt.Bin(step=0.05, extent=[0, 1])), "count()"],
        )
        .properties(height=170)
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )


def render_card(
    index: int,
    source_bytes: bytes,
    source_name: str,
    progress: dict[str, Any],
) -> None:
    """Draw one image's whole world: inputs, numbers, progress and curves."""
    schema = progress_schema(progress)
    result = schema["result"] or {}
    download_name = st.session_state.get("download_names", {}).get(index, source_name)
    size = decoded_size(source_bytes)

    with st.container(border=True):
        st.markdown(
            f'<div class="yolo-head"><span class="idx">#{index + 1}</span>'
            f'<span class="name">{html.escape(source_name)}</span>'
            f'<span class="meta">{len(source_bytes) / 1024:.0f} KB'
            + (f' · {size[0]}×{size[1]} px' if size else "")
            + "</span></div>",
            unsafe_allow_html=True,
        )

        # Three equal columns put each picture at roughly a third of the page width,
        # which is what keeps the pair on one row on a normal desktop window.
        left, middle, right = st.columns(3, gap="small")

        with left:
            st.image(
                source_bytes,
                caption=f"输入图片 · {source_name}",
                width="stretch",
            )

        with middle:
            # The placeholder is created empty and filled in later in this same run,
            # so the annotated image lands in the row that was already laid out.
            result_slot = st.empty()

        with right:
            live = st.empty()
            extra_slot = st.empty()

        if schema["done"]:
            if result.get("annotated_image"):
                result_slot.image(
                    data_uri_to_bytes(result["annotated_image"]),
                    caption=f"检测结果 · {download_name}",
                    width="stretch",
                )
            else:
                result_slot.info("未返回标注图片。")

        threshold = float(result.get("confidence_threshold", 0.5))

        # ---- third column: live read-out while running, full stats once done ----
        if not schema["done"]:
            live.markdown(
                metric_tiles(
                    [
                        ("状态", f'<span style="color:#00d4aa;font-weight:600;">'
                                 f'{html.escape(schema["label"])}</span>'),
                        ("已用时间", f"{schema['elapsed_ms'] / 1000:.1f} s"),
                        ("GPU 使用率", None if schema["gpu"] is None else f"{schema['gpu']} %"),
                        ("显存占用", None if schema["vram_used"] is None
                         else f"{schema['vram_used']:.0f} MB"),
                        ("图像亮度", None if schema["brightness"] is None
                         else f"{schema['brightness']:.0f} / 255"),
                        ("图像清晰度", None if schema["sharpness"] is None
                         else f"{schema['sharpness']:.0f}"),
                        ("推理尺寸", None if schema["width"] is None
                         else f"{schema['width']}×{schema['height']} px"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            extra_slot.markdown(
                '<div class="yolo-state">推理进行中，结果图片与统计会实时出现在本容器内。'
                '<span class="yolo-state-note">进度条表示当前所处的流水线阶段，'
                '右侧曲线为该图片的实测数据。</span></div>',
                unsafe_allow_html=True,
            )
        elif schema["error"]:
            live.error(schema["error"])
        elif result:
            live.markdown(
                metric_tiles(
                    [
                        ("目标数量", f'<b style="font-size:1.25rem;">{result.get("count", 0)}</b>'),
                        ("类别数量", len(result.get("counts_by_class") or {})),
                        ("推理耗时", f"{float(result.get('inference_ms') or 0):.0f} ms"),
                        ("端到端耗时", f"{float(result.get('total_ms') or 0):.0f} ms"),
                        ("平均置信度", None if not result.get("scores")
                         else f"{result['scores']['mean_confidence']:.3f}"),
                        ("最高置信度", None if not result.get("scores")
                         else f"{result['scores']['max_confidence']:.3f}"),
                        ("平均框面积", None if not result.get("detections")
                         else f"{avg_box_ratio(result) * 100:.1f} % 画面"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                f"模型 `{result.get('model', '')}` · 设备 `{result.get('device', '')}` · "
                f"阈值 conf ≥ {threshold} / IoU {result.get('iou_threshold', '')}"
            )
            if not result.get("detections"):
                st.warning("没有目标达到置信度阈值，可尝试调低侧边栏的置信度。")
            else:
                with st.expander(f"按类别汇总（{len(result.get('summary') or [])} 类）"):
                    st.dataframe(
                        summary_dataframe(result["summary"]), width="stretch", hide_index=True
                    )
                with st.expander(f"全部检测结果（{result.get('count', 0)} 个）"):
                    st.dataframe(
                        detections_dataframe(result["detections"]),
                        width="stretch",
                        hide_index=True,
                    )

            extra_slot.markdown("**导出**")
            grid = extra_slot.columns(2)
            stem = os.path.splitext(download_name)[0]
            grid[0].download_button(
                "检测明细 CSV",
                detections_csv_bytes(result["detections"]),
                file_name=f"{stem}_detections.csv",
                mime="text/csv",
                key=f"csv_det_{index}",
            )
            grid[1].download_button(
                "类别汇总 CSV",
                summary_dataframe(result["summary"]).to_csv(index=False).encode("utf-8"),
                file_name=f"{stem}_summary.csv",
                mime="text/csv",
                key=f"csv_sum_{index}",
            )
            grid[0].download_button(
                "完整 JSON",
                json.dumps(
                    {k: v for k, v in result.items() if k != "annotated_image"}, indent=2
                ).encode("utf-8"),
                file_name=f"{stem}_detections.json",
                mime="application/json",
                key=f"json_{index}",
            )
            if result.get("annotated_image"):
                extension = "png" if result["annotated_image"].startswith("data:image/png") else "jpg"
                grid[1].download_button(
                    "标注图片",
                    data_uri_to_bytes(result["annotated_image"]),
                    file_name=f"{stem}_annotated.{extension}",
                    mime="image/png" if extension == "png" else "image/jpeg",
                    key=f"img_{index}",
                )

        # ---- full-width progress bar, stage strip and inference curves ----
        percent = schema["percent"] / 100.0
        if schema["done"]:
            status = "推理完成"
        elif schema["percent"] <= 0.0:
            status = "准备中…"
        else:
            status = f"推理中 · {percent * 100:.0f}%"
        st.markdown(
            f'<div class="yolo-live"><span class="stage">{html.escape(status)} — '
            f'{html.escape(schema["label"])}</span>'
            f'<span class="clock">已用 {schema["elapsed_ms"] / 1000:.1f} s</span></div>',
            unsafe_allow_html=True,
        )
        bar_slot = st.empty()
        if percent > 0.0:
            bar_slot.progress(
                min(1.0, max(0.0, percent)),
                text=f"{percent * 100:.0f}% · {schema['label']}",
            )
        st.markdown(
            render_chips(schema["stages_seen"], schema["stage"]),
            unsafe_allow_html=True,
        )

        timing_slot = st.empty()
        frame = timing_frame(progress)
        if len(frame) >= 1:
            timing_slot.altair_chart(timing_chart(frame), width="stretch")

        hardware = hardware_frame(progress)
        if not hardware.empty and schema["has_gpu"]:
            with st.expander("推理过程硬件曲线（GPU 使用率 / 显存）", expanded=not schema["done"]):
                st.altair_chart(hardware_chart(hardware), width="stretch")
            st.markdown(
                f'<div class="yolo-gpu">GPU 使用率采样自推理流水线的各阶段边界'
                f'（{schema["event_count"]} 个采样点）。</div>',
                unsafe_allow_html=True,
            )

        if schema["done"] and schema["has_result"]:
            confidence = confidence_chart(result, threshold)
            if confidence is not None:
                st.altair_chart(confidence, width="stretch")


def render_gallery(records: list[dict[str, Any]]) -> None:
    """Every image's container, one above the next."""
    if not records:
        return
    for index, record in enumerate(records):
        source = record.get("source") or {}
        render_card(
            index,
            record.get("source_bytes") or b"",
            source.get("name") or f"image-{index + 1}",
            record["progress"],
        )


# ============================================================
# JOB RUNNER (worker thread + Streamlit-driven polling)
# ============================================================

def run_job(pages: list[tuple[bytes, str]], settings: dict[str, Any]) -> None:
    """
    Detect every queued image, one container each, top to bottom.

    The detection itself runs on a worker thread because the progress UI has to be
    repainted while it is busy; the worker only ever touches plain dicts, and this
    (main) thread does all the Streamlit calls. The queue carries events from one
    to the other. In-flight records are stashed in session state *before* the loop
    starts, so an interrupted run still shows what got done rather than vanishing.
    """
    records: list[dict[str, Any]] = []
    for payload, name in pages:
        records.append(
            {
                "source_bytes": payload,
                "progress": progress_reset(
                    {"name": name, "kb": round(len(payload) / 1024, 1), "bytes": len(payload)}
                ),
            }
        )
    events: "queue.Queue[tuple[int, dict[str, Any]]]" = queue.Queue()
    state = {"index": 0}
    token = str(time.time())
    st.session_state["detections"] = {"records": records, "meta": {"token": token}}
    st.session_state["download_names"] = {
        index: os.path.basename(name) for index, (_, name) in enumerate(pages)
    }

    def capture_for(index: int) -> Callable[[dict[str, Any]], None]:
        sink = progress_capture("detect", records[index]["progress"])

        def capture(event: dict[str, Any]) -> None:
            sink(event)
            events.put((index, event))

        return capture

    def worker() -> None:
        for position, (payload, name) in enumerate(pages):
            state["index"] = position
            record = records[position]["progress"]
            try:
                if settings["backend"] == REST_BACKEND_LABEL:
                    result = detect_via_api(
                        settings["api_url"], payload, name,
                        settings["model"], settings["confidence"], settings["iou"],
                        settings["device"], settings["image_format"], settings["max_size"],
                        on_progress=capture_for(position),
                    )
                else:
                    result = detect_locally(
                        payload, settings["model"], settings["confidence"], settings["iou"],
                        settings["device"], settings["max_size"],
                        image_format=settings["image_format"],
                        on_progress=capture_for(position),
                    )
            except BackendError as exc:
                record["error"] = str(exc)
                record["label"] = "失败"
            except Exception as exc:  # pragma: no cover - unexpected runtime failure
                record["error"] = f"推理失败：{exc}"
                record["label"] = "失败"
            else:
                record["result"] = result
                record["percent"] = 100.0
                record["label"] = "完成"
            finally:
                record["done"] = True

    render_gallery(records)

    thread = threading.Thread(target=worker, name="yolo-ui-detect", daemon=True)
    thread.start()
    while thread.is_alive() or not events.empty():
        drained: set[int] = set()
        while True:
            try:
                index, _ = events.get(timeout=0.05)
            except queue.Empty:
                break
            drained.add(index)
        for index in drained:
            render_card(
                index,
                records[index]["source_bytes"],
                records[index]["progress"]["source"]["name"],
                records[index]["progress"],
            )

    for record in records:
        # A worker that died without reporting anything would otherwise leave a
        # container spinning forever.
        if not record["progress"].get("done"):
            record["progress"]["done"] = True
            record["progress"].setdefault("error", None)
            if not record["progress"].get("result") and not record["progress"]["error"]:
                record["progress"]["label"] = "完成"

    st.session_state["detections"] = {
        "records": records,
        "meta": {
            "token": token,
            "model": settings["model"],
            "device": settings["device"],
            "backend": settings["backend"],
            "finished": time.strftime("%H:%M:%S"),
        },
    }
    # One clean repaint from the finished records, so the container layout does not
    # depend on how many progress repaints this run happened to make.
    st.rerun()


def detections_csv_bytes(detections: list[dict[str, Any]]) -> bytes:
    try:
        import detector as det

        return det.detections_csv(detections).encode("utf-8")
    except ImportError:
        return detections_dataframe(detections).to_csv(index=False).encode("utf-8")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ 设置")

backend = st.sidebar.radio(
    "推理后端",
    BACKEND_OPTIONS,
    help="REST API 通过 HTTP 调用 api.py；本地推理则在 Streamlit 进程内直接运行模型。",
)
api_url = DEFAULT_API_URL
if backend == REST_BACKEND_LABEL:
    api_url = st.sidebar.text_input("API 地址", value=DEFAULT_API_URL)

# The model list is discovered, never hardcoded: builtin aliases come from the
# backend, and every *.pt in its models/ directory is appended. A click on
# "刷新模型列表" drops the cache so a freshly copied weight appears immediately.
if st.session_state.pop("refresh_models", False):
    _cached_remote_catalogue.clear()

catalogue, catalogue_source, catalogue_note = load_model_catalogue(backend, api_url)
model_options = order_models(catalogue)
custom_models = [name for name in (catalogue.get("custom") or []) if name]

header_left, header_right = st.sidebar.columns([3, 2])
header_left.markdown("**模型**")
with header_right:
    # A zero-height spacer nudges the button down so it lines up with the label.
    st.write("")
    if st.button(
        "🔄 刷新",
        key="refresh_models_btn",
        help="重新扫描 models/ 目录并重新读取后端模型列表",
    ):
        st.session_state["refresh_models"] = True
        st.rerun()

model = st.sidebar.selectbox(
    "模型",
    model_options,
    index=0,
    label_visibility="collapsed",
    help=(
        "带 `.pt` 后缀的是 `models/` 目录里的自定义权重（已排在列表最前面）；"
        "其余为内置别名，首次使用时自动下载。n 速度最快、l/x 精度最高。"
    ),
)

model_status = f"共 {len(model_options)} 个可选模型"
if custom_models:
    model_status += f"，其中 {len(custom_models)} 个来自 `models/`"
model_status += f"（来源：{catalogue_source}）"
st.sidebar.caption(model_status)

if backend == REST_BACKEND_LABEL:
    st.sidebar.caption("📂 自定义权重的存放目录：`models/`（可用 `MODELS_DIR` 环境变量修改）。")
else:
    try:
        import detector as _det

        st.sidebar.caption(f"📂 自定义权重的存放目录：`{_det.MODELS_DIR}`")
    except ImportError:
        pass

if catalogue_note:
    st.sidebar.warning(catalogue_note)
elif not custom_models:
    if backend == REST_BACKEND_LABEL:
        st.sidebar.info(
            "还没有自定义权重。把 `.pt` 复制到后端的 `models/` 目录，"
            "或在这里上传，然后点「🔄 刷新」即可选用。"
        )
    else:
        st.sidebar.info(
            "还没有自定义权重。把 `.pt` 复制到上面的 `models/` 目录，然后点「🔄 刷新」即可选用。"
        )

if backend == REST_BACKEND_LABEL:
    with st.sidebar.expander("📤 上传自定义权重（.pt）", expanded=False):
        uploaded_model = st.file_uploader("选择 .pt 权重文件", type=["pt"], key="model_upload")
        if uploaded_model is not None and st.button("安装到 models/", key="model_upload_btn"):
            try:
                info = api_upload_model(api_url, uploaded_model.name, uploaded_model.getvalue())
            except BackendError as exc:
                st.error(str(exc))
            except requests.exceptions.RequestException as exc:
                st.error(f"无法连接 API：{exc}")
            else:
                _cached_remote_catalogue.clear()
                st.session_state["refresh_models"] = True
                # Drop the uploaded bytes from the widget before rerunning, or the
                # file would still be staged and the install would look repeatable.
                st.session_state.pop("model_upload", None)
                st.success(f"已安装 `{info.get('model', uploaded_model.name)}`，正在刷新列表…")
                st.rerun()
        st.caption(
            "上传的权重会写入后端的 `models/` 目录并出现在下拉框中。"
            "`.pt` 文件本质上是 pickle，请只安装你信任的模型。"
        )

confidence = st.sidebar.slider("置信度阈值", 0.10, 0.90, 0.50, 0.05)
iou = st.sidebar.slider("NMS IoU 阈值", 0.10, 0.90, 0.45, 0.05)
device = st.sidebar.selectbox("运行设备", ["auto", "cpu", "cuda:0", "mps"], index=0)
max_size = st.sidebar.select_slider(
    "最大推理尺寸（像素）", options=[640, 960, 1280, 1600, 1920, 2560], value=1280,
    help="推理前会把长边缩放到该尺寸；检测框仍按原图坐标输出。",
)
image_format = st.sidebar.radio(
    "标注图片格式", ["jpeg", "png"], index=0,
    help="JPEG 响应体积更小；PNG 无损，但体积约为前者的 10 倍。",
)

st.sidebar.divider()

if backend == REST_BACKEND_LABEL:
    if st.sidebar.button("🔌 测试连接", width="stretch"):
        try:
            info = api_health(api_url)
            gpu = info.get("gpu") or {}
            st.sidebar.success(
                f"连接成功 · v{info['version']}\n\n"
                f"运行设备：`{info['device']}`" + (f"\n\nGPU：{gpu.get('name')}" if gpu else "")
            )
        except requests.RequestException as exc:
            st.sidebar.error(
                f"无法连接：{exc}\n\n请先启动后端（Windows 可双击项目根目录的"
                f"「Launch.bat」）：\n\n"
                f"`uvicorn api:app --reload --port 8000`"
            )
    if st.sidebar.button("🔥 预热模型", width="stretch"):
        with st.sidebar.status("正在加载权重并预热…", expanded=True) as status:
            try:
                info = api_warmup(api_url, model, device)
                status.update(label="模型已就绪", state="complete")
                st.write(
                    f"已在 `{info['device']}` 上加载 `{info['model']}` "
                    f"（{info['classes']} 个类别），耗时 {info['total_ms']:.0f} ms。"
                )
            except requests.RequestException as exc:
                status.update(label="预热失败", state="error")
                st.write(str(exc))

settings = {
    "backend": backend,
    "api_url": api_url,
    "model": model,
    "confidence": confidence,
    "iou": iou,
    "device": device,
    "image_format": image_format,
    "max_size": max_size,
}

st.sidebar.divider()
st.sidebar.caption(
    "模型列表会自动读取：内置别名来自后端，`models/` 目录里每个 `.pt` 都会出现在下拉框中"
    f"（当前 {len(model_options)} 个）。把新权重放进去后点「🔄 刷新」即可。"
)
if backend == REST_BACKEND_LABEL:
    st.sidebar.caption(
        "⚠️ `ALLOW_CUSTOM_MODEL_UPLOAD` 默认开启。`.pt` 文件本质上是 pickle，"
        "因此请只上传你信任的模型，并让 API 保持在本机监听。"
    )


# ============================================================
# MAIN
# ============================================================

st.title("🎯 " + APP_TITLE)
st.caption(
    "在浏览器中使用 YOLOv8 / YOLO11 —— 一次上传多张图片、调用摄像头或通过 URL 加载图片。"
    "每张图片占一个容器，容器内左侧为输入图、中间为检测结果图（各约三分之一宽，因此同排显示），"
    "右侧是该图的统计与导出，底部是推理进度与实测曲线。"
)

upload_tab, webcam_tab, url_tab = st.tabs(["📁 上传图片", "📷 摄像头", "🔗 图片 URL"])

with upload_tab:
    uploads = st.file_uploader(
        "选择图片（可一次选中多张，也可以把多张拖进来）",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        help="支持 JPG、PNG、WebP、BMP。每张图片会生成一个独立容器，按上传顺序从上到下排列。",
    )
    if uploads:
        names = [item.name for item in uploads]
        st.caption(
            f"已选择 {len(uploads)} 张图片：" + "、".join(f"`{name}`" for name in names[:6])
            + ("…" if len(names) > 6 else "")
        )
        if st.button("🚀 开始检测", type="primary", key="detect_upload", width="stretch"):
            run_job([(item.getvalue(), item.name) for item in uploads], settings)
    else:
        st.info("上传一张或多张 JPG、PNG、WebP、BMP 图片即可开始。")

with webcam_tab:
    shot = st.camera_input("拍摄照片")
    if shot is not None:
        if st.button("🚀 开始检测", type="primary", key="detect_webcam", width="stretch"):
            run_job([(shot.getvalue(), "webcam.jpg")], settings)
    else:
        st.info("请允许浏览器访问摄像头，然后拍摄一帧画面。")

with url_tab:
    url = st.text_input("图片 URL", placeholder="https://example.com/photo.jpg")
    if st.button("🚀 开始检测", type="primary", key="detect_url", width="stretch"):
        try:
            payload, filename = fetch_image_url(url)
        except BackendError as exc:
            st.error(str(exc))
        else:
            run_job([(payload, filename)], settings)

st.divider()
detections = st.session_state.get("detections") or {}
records = detections.get("records") or []
meta = detections.get("meta") or {}
if records:
    done = sum(1 for record in records if (record.get("progress") or {}).get("done"))
    header = f"### 检测结果（{len(records)} 个容器）"
    if meta.get("finished"):
        header += f" · 模型 `{meta.get('model', '')}` · 设备 `{meta.get('device', '')}`"
    st.markdown(header)
    st.caption(
        f"已完成 {done} / {len(records)} 张 · 每行一个容器，容器内输入图与结果图并排"
        "（各约三分之一宽），推理进度与数据曲线位于容器底部。"
    )
    render_gallery(records)
else:
    st.info("检测结果会以「每张图片一个容器」的形式显示在这里，容器内同时展示输入图、结果图和推理曲线。")

with st.expander("API 参考", expanded=False):
    st.markdown(
        """
前端调用的接口与你可以手动调用的接口完全一致：

```bash
curl -X POST http://localhost:8000/detect \\
  -F 'file=@test_image.jpg' \\
  -F 'confidence=0.5' \\
  -F 'model=yolov8n'
```

需要推理进度和过程曲线时改用流式接口，它按行（NDJSON）推送每个流水线阶段：

```bash
curl -N -X POST http://localhost:8000/detect/stream \\
  -F 'file=@test_image.jpg' -F 'model=yolov8n'
```

```json
{"type":"progress","stage":"resized","label":"缩放预处理","elapsed_ms":12.4,"width":1280,"height":853,"brightness":118.2}
{"type":"result","count":3,"inference_ms":41.2,"detections":[...],"summary":[...]}
```

其他接口：`GET /health`、`GET /models`、`POST /warmup`、
`POST /models/upload`、`POST /detect/image`（返回标注后的 PNG 图片）。
交互式接口文档：<http://127.0.0.1:8000/docs>
        """
    )
