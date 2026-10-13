"""
Streamlit frontend for YOLO object detection.

Run it with:

    streamlit run app.py

It talks to the FastAPI backend from api.py over HTTP (start that first with
`uvicorn api:app --reload --port 8000`), and can also fall back to running the
model in-process if you would rather not start the backend -- both paths call the
same code in detector.py, so results are identical.

On Windows the easiest way to start both halves at once is the one-click launcher
at the repository root: double-click `一键启动.bat` (or `OneClick-Start.bat`),
which sets up the environment, starts the API and this UI, and opens the browser.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

APP_TITLE = "YOLO 目标检测"
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
URL_MAX_MB = 25.0

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

st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
) -> dict[str, Any]:
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
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/detect", files=files, data=form, timeout=600
        )
    except requests.exceptions.ConnectionError as exc:
        raise BackendError(
            f"无法连接位于 {api_url} 的 API 服务。请先启动它：\n\n"
            f"    Windows 一键启动：双击项目根目录的「一键启动.bat」\n"
            f"    手动启动：uvicorn api:app --reload --port 8000\n\n"
            f"或在侧边栏把“推理后端”切换为“{LOCAL_BACKEND_LABEL}”。"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendError("API 处理这张图片时超时。") from exc

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
) -> dict[str, Any]:
    try:
        import detector as det
    except ImportError as exc:  # ultralytics / torch not installed
        raise BackendError(
            f"本地推理需要先安装检测相关依赖（{exc}）。"
        ) from exc

    try:
        image = det.decode_image(data)
    except ValueError as exc:
        raise BackendError(str(exc)) from exc

    started = time.perf_counter()
    try:
        payload = det.detect(
            image,
            model=model,
            confidence=confidence,
            iou=iou,
            device=device,
            max_size=max_size,
            annotate=True,
            image_format="jpeg",
        )
    except (det.ModelError, ValueError) as exc:
        raise BackendError(str(exc)) from exc

    payload["success"] = True
    payload["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    payload["summary"] = det.summarise(payload["detections"])
    return payload


# ============================================================
# HELPERS
# ============================================================

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


def run_detection(data: bytes, filename: str, settings: dict[str, Any]) -> None:
    """Execute detection with the configured backend and store the result."""
    try:
        if settings["backend"] == REST_BACKEND_LABEL:
            result = detect_via_api(
                settings["api_url"], data, filename,
                settings["model"], settings["confidence"], settings["iou"],
                settings["device"], settings["image_format"], settings["max_size"],
            )
        else:
            result = detect_locally(
                data, settings["model"], settings["confidence"], settings["iou"],
                settings["device"], settings["max_size"],
            )
    except BackendError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
        return

    st.session_state["result"] = result
    st.session_state["source_bytes"] = data
    st.session_state["source_name"] = filename
    st.session_state["settings_used"] = settings


def render_results() -> None:
    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    st.subheader("检测结果")

    metrics = st.columns(4)
    metrics[0].metric("目标数量", result["count"])
    metrics[1].metric("类别数量", len(result["counts_by_class"]))
    metrics[2].metric("推理耗时", f"{result['inference_ms']:.0f} ms")
    metrics[3].metric("总耗时", f"{result.get('total_ms', 0):.0f} ms")

    st.caption(
        f"模型 **{result['model']}**，运行于 **{result['device']}** · "
        f"置信度 ≥ {result['confidence_threshold']} · IoU {result['iou_threshold']} · "
        f"源图尺寸 {result['image']['width']}×{result['image']['height']} 像素"
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        if result.get("annotated_image"):
            st.image(
                data_uri_to_bytes(result["annotated_image"]),
                caption=f"标注结果 · {st.session_state.get('source_name', 'image')}",
                width="stretch",
            )
        else:
            st.info("未返回标注图片。")
    with right:
        if not result["detections"]:
            st.warning(
                "没有目标达到置信度阈值。可以尝试调低置信度滑块，"
                "或改用更大的模型。"
            )
        else:
            summary = summary_dataframe(result["summary"])
            st.markdown("**按类别汇总**")
            st.dataframe(summary, width="stretch", hide_index=True)
            st.markdown("**各类别目标数量**")
            st.bar_chart(
                summary[["类别", "数量"]],
                x="类别",
                y="数量",
                horizontal=True,
                color="#00d4aa",
            )

    if result["detections"]:
        with st.expander(f"全部检测结果（{result['count']} 个）", expanded=False):
            st.dataframe(
                detections_dataframe(result["detections"]), width="stretch", hide_index=True
            )

    st.markdown("**导出**")
    downloads = st.columns(4)
    stem = os.path.splitext(st.session_state.get("source_name", "image"))[0]
    downloads[0].download_button(
        "检测明细 CSV",
        detections_csv_bytes(result["detections"]),
        file_name=f"{stem}_detections.csv",
        mime="text/csv",
    )
    downloads[1].download_button(
        "类别汇总 CSV",
        summary_dataframe(result["summary"]).to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_summary.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "完整 JSON",
        json.dumps(
            {k: v for k, v in result.items() if k != "annotated_image"}, indent=2
        ).encode("utf-8"),
        file_name=f"{stem}_detections.json",
        mime="application/json",
    )
    if result.get("annotated_image"):
        extension = "png" if result["annotated_image"].startswith("data:image/png") else "jpg"
        downloads[3].download_button(
            "标注图片",
            data_uri_to_bytes(result["annotated_image"]),
            file_name=f"{stem}_annotated.{extension}",
            mime=f"image/{'png' if extension == 'png' else 'jpeg'}",
        )


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
                f"「一键启动.bat」）：\n\n"
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
    "在浏览器中使用 YOLOv8 / YOLO11 —— 上传图片、调用摄像头或通过 URL 加载图片。"
    "检测框、类别标签和置信度会绘制在图片上，并在下方汇总展示。"
)

upload_tab, webcam_tab, url_tab = st.tabs(["📁 上传图片", "📷 摄像头", "🔗 图片 URL"])

with upload_tab:
    uploaded = st.file_uploader(
        "选择图片", type=IMAGE_TYPES, label_visibility="collapsed"
    )
    if uploaded is not None:
        st.image(uploaded, caption=f"{uploaded.name} · {uploaded.size / 1024:.0f} KB", width="stretch")
        if st.button("🚀 开始检测", type="primary", key="detect_upload"):
            run_detection(uploaded.getvalue(), uploaded.name, settings)
    else:
        st.info("上传一张 JPG、PNG、WebP 或 BMP 图片即可开始。")

with webcam_tab:
    shot = st.camera_input("拍摄照片")
    if shot is not None:
        if st.button("🚀 开始检测", type="primary", key="detect_webcam"):
            run_detection(shot.getvalue(), "webcam.jpg", settings)
    else:
        st.info("请允许浏览器访问摄像头，然后拍摄一帧画面。")

with url_tab:
    url = st.text_input("图片 URL", placeholder="https://example.com/photo.jpg")
    if st.button("🚀 开始检测", type="primary", key="detect_url"):
        try:
            payload, filename = fetch_image_url(url)
        except BackendError as exc:
            st.error(str(exc))
        else:
            st.image(payload, caption=f"{filename} · {len(payload) / 1024:.0f} KB", width="stretch")
            run_detection(payload, filename, settings)

render_results()

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

其他接口：`GET /health`、`GET /models`、`POST /warmup`、
`POST /models/upload`、`POST /detect/image`（返回标注后的 PNG 图片）。
交互式接口文档：<http://127.0.0.1:8000/docs>
        """
    )
