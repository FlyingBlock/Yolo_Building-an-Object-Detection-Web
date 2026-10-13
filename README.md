# YOLO Building — Object Detection Web App

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square) ![Stars](https://img.shields.io/github/stars/FlyingBlock/Yolo_Building-an-Object-Detection-Web?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/FlyingBlock/Yolo_Building-an-Object-Detection-Web?style=flat-square&color=blue) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Upload an image, point a camera at something, or paste a URL — and watch YOLOv8
> draw its boxes in the browser. One detection core, several front doors: a FastAPI
> service, a Streamlit UI, and two command-line tools. On Windows, a single
> double-click starts everything.

**Topics:** `computer-vision` · `deep-learning` · `object-detection` · `yolov8` · `ultralytics` · `fastapi` · `streamlit` · `opencv` · `python-web-app`

---

## Overview

This project wraps [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) in a web
application you can actually run without assembling a stack yourself.

Everything routes through one module, `detector.py`, which owns model loading, inference
and annotation. The REST API, the Streamlit UI and both CLI scripts import it, so a
detection made from the browser, from `curl` and from the command line is produced by
exactly the same code — same boxes, same labels, same confidence values.

The Streamlit frontend offers three ways in: **upload** an image file, capture a frame
with the **webcam**, or paste an **image URL**. Detected objects are returned with
bounding boxes, class labels and confidence scores, alongside a summary table and a
per-class count chart. The UI can talk to the FastAPI backend over HTTP, or run the model
in-process if you would rather not start a second service.

The interface and the generated API documentation are written in Chinese.

---

## Features

**Three input modes** — file upload (JPG / PNG / WebP / BMP), a live webcam capture, or a
remote image URL.

**A shared detection core** — `detector.py` is the single source of truth for inference
and annotation. The API, the UI and the CLIs cannot drift apart, because they call the
same functions.

**FastAPI backend** — a documented service (`/docs`) with:

| Endpoint | Purpose |
|---|---|
| `POST /detect` | Run detection, return JSON (optionally with the annotated image embedded as a data URI) |
| `POST /detect/image` | Run detection, return the annotated image as PNG |
| `GET /models` | List built-in, custom and already-cached models |
| `POST /models/upload` | Install a custom `.pt` weight file |
| `POST /warmup` | Load the weights and pay the one-off warm-up inference cost up front |
| `GET /health` | Service and inference-device status |
| `GET /` | Service metadata |

**Pluggable inference backend in the UI** — switch between the REST service and in-process
inference from the sidebar. If the API is unreachable, the UI says so and scans the local
`models/` directory instead of failing.

**Configurable thresholds** — confidence and NMS IoU are adjustable per request and from
the UI; a class filter and a maximum input size are available too.

**Custom weights** — drop a `.pt` file into `models/`, or upload one through the UI. The
model picker then offers it next to the built-in YOLOv8 variants.

**Command-line tools** — `detect.py` for a single image and `batch_detect.py` for a whole
folder, the latter producing annotated images, per-image JSON, plus combined `results.json`
and `results.csv`.

**One-click Windows launcher** — `start.ps1` finds a suitable interpreter, installs what is
missing, starts the API and the UI as one process tree, and opens the browser.

**Request guards** — upload size and pixel-count limits, a configurable CORS origin list,
and custom-model loading kept behind an explicit opt-in, because loading a `.pt` file
executes pickle data.

---

## Architecture

```
                 ┌──────────────────────────────┐
   Browser  ───► │  Streamlit UI   (app.py)     │
                 │  upload · webcam · URL       │
                 └───────────┬──────────────────┘
                             │  HTTP (or in-process)
                             ▼
                 ┌──────────────────────────────┐
   curl / app ─► │  FastAPI service (api.py)    │
                 │  /detect · /detect/image     │
                 └───────────┬──────────────────┘
                             │
                             ▼
                 ┌──────────────────────────────┐
                 │  detector.py                 │
                 │  load · infer · annotate     │
                 └───────────┬──────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      yolov8n/s/m/… (.pt)            models/  ← custom weights
```

---

## Requirements

- **Python 3.9+** (3.11 or 3.13 recommended)
- Windows for the one-click launcher; the API, UI and CLIs are cross-platform
- No GPU required — CUDA is used automatically when available

Model weights are **not** stored in this repository. On first use, Ultralytics downloads the
selected variant (~6 MB for `yolov8n`). The first inference on a CUDA device also pays a
one-off kernel warm-up cost, which `POST /warmup` exists to absorb.

---

## Quick start

### Windows — one double-click

Clone the repository, then double-click **`一键启动.bat`** (Chinese messages) or
**`OneClick-Start.bat`** (ASCII-only). The launcher sets up the environment, starts both
halves of the app, and opens the browser at the Streamlit UI.

It looks for a Python interpreter in this order:

1. `-PythonPath <python.exe>` if you pass one
2. a conda environment named `yoloweb` (Miniforge / Anaconda / Miniconda)
3. a project-local `.venv`
4. any system Python 3.9+

Useful switches:

```powershell
# start with different ports, or without opening a browser
.\OneClick-Start.bat -ApiPort 8010 -UiPort 8511 -NoBrowser

# install dependencies and exit, without starting anything
.\OneClick-Start.bat -SetupOnly

# run the launcher directly for the full option list
powershell -ExecutionPolicy Bypass -File start.ps1 -?
```

### 中文快速开始

1. 克隆仓库后，双击项目根目录的 **`一键启动.bat`**。
2. 脚本会自动准备 Python 环境、安装缺失的依赖、启动后端与前端，并打开浏览器。
3. 首次运行会下载 YOLOv8 权重（`yolov8n` 约 6 MB），请保持网络畅通。
4. 只想安装环境而不启动服务，使用 `OneClick-Start.bat -SetupOnly`。

前端页面与接口文档均为中文。

### Manual setup

```bash
git clone https://github.com/FlyingBlock/Yolo_Building-an-Object-Detection-Web.git
cd Yolo_Building-an-Object-Detection-Web

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt

# terminal 1 — API
uvicorn api:app --reload --port 8000

# terminal 2 — UI
streamlit run app.py
```

Then open <http://localhost:8501> for the UI and <http://127.0.0.1:8000/docs> for the
interactive API documentation.

---

## Usage

### API

```bash
# JSON result, with the annotated image embedded as a data URI
curl -X POST http://localhost:8000/detect \
  -F 'file=@samples/bus.jpg' \
  -F 'confidence=0.5' \
  -F 'model=yolov8n'

# Annotated PNG instead of JSON
curl -X POST http://localhost:8000/detect/image \
  -F 'file=@samples/bus.jpg' -o annotated.png

# Warm the model up so the first user upload is fast
curl -X POST http://localhost:8000/warmup
```

### Single image

```bash
python detect.py --image samples/bus.jpg --model yolov8m --save-json detections.json
```

### Folder batch

```bash
python batch_detect.py --input ./images/ --output ./results/ --model yolov8s --recursive
```

Both CLIs accept `--confidence`, `--iou`, `--device`, `--max-size`, `--classes` and
output-related flags; run either with `--help` for the full list.

### Custom weights

Put a `.pt` file in `models/`, or upload it from the UI sidebar, and select it in the model
picker. Loading a `.pt` file executes pickle data, so keep
`ALLOW_CUSTOM_MODEL_UPLOAD` enabled only while the API is bound to localhost.

---

## Configuration

Every setting has a built-in default, so a `.env` file is optional. Copy `.env.example` to
`.env` to override anything.

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `yolov8n.pt` | Default model variant |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | Non-maximum suppression IoU threshold |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda:0` or `mps` |
| `MAX_IMAGE_SIZE` | `1280` | Long edge is downscaled to this before inference; `0` disables it |
| `MAX_UPLOAD_MB` | `25` | Maximum accepted upload size |
| `MAX_IMAGE_PIXELS` | `64000000` | Rejects decompression-bomb style images |
| `PRELOAD_MODEL` | `0` | Load the default model at startup instead of on first request |
| `ALLOW_CUSTOM_MODEL_UPLOAD` | `1` | Permit installing uploaded `.pt` weights |
| `MODELS_DIR` | `models` | Where custom weights live |
| `CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API |
| `API_URL` | `http://127.0.0.1:8000` | Where the UI looks for the API |

> `ALLOW_CUSTOM_MODEL_UPLOAD=1` is convenient on a local machine and unsafe on a public
> one. Set it to `0` before exposing the API to a network.

---

## Project structure

```
Yolo_Building-an-Object-Detection-Web/
├── 一键启动.bat             # one-click launcher (Chinese messages)
├── OneClick-Start.bat       # one-click launcher (ASCII only)
├── start.ps1                # the launcher: interpreter, deps, both servers, browser
├── api.py                   # FastAPI backend          -> uvicorn api:app
├── app.py                   # Streamlit frontend       -> streamlit run app.py
├── detector.py              # shared detection core (load, infer, annotate)
├── detect.py                # single-image CLI
├── batch_detect.py          # folder batch CLI
├── requirements.txt         # dependencies for the web app
├── requirements-atlas.txt   # optional, for the unrelated visualisation only
├── .env.example             # every setting, with its default
├── .gitignore               # excludes weights, logs, caches
├── models/                  # drop-in location for custom .pt weights
│   └── .gitkeep
├── samples/
│   └── bus.jpg              # sample image for a first run
├── atlas_engine.py          # optional extra — see below
├── quantum_dream_atlas.py   # optional extra — see below
├── yolov3.cfg               # legacy config inherited from upstream
├── LICENSE
└── README.md
```

### Optional extra: the Quantum Dream Atlas

`quantum_dream_atlas.py` and `atlas_engine.py` are a separate Streamlit visualisation
playground (21 STEM subjects rendered as dynamic plots). It is **unrelated to object
detection** and is not needed by the web app. Its dependencies are kept apart in
`requirements-atlas.txt`:

```bash
pip install -r requirements-atlas.txt
streamlit run quantum_dream_atlas.py
```

---

## Roadmap

- [ ] Video file detection with per-frame annotation and output video export
- [ ] Multi-stream webcam support for multi-camera monitoring setups
- [ ] A custom training pipeline: fine-tune YOLOv8 on a user-provided labelled dataset
- [ ] ONNX and TensorRT export for optimised edge deployment
- [ ] Tracking mode: DeepSORT integration for object ID persistence across frames

---

## Contributing

Issues and pull requests are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push the branch: `git push origin feature/your-idea`
5. Open a pull request with a clear description

Conventional commit messages and documentation for new features are appreciated.

---

## Credits and relationship to upstream

This repository is a **derivative work**, and the upstream project deserves the credit for
the original concept and structure:

- **[Devanik21/Yolo_Building-an-Object-Detection-Web-App](https://github.com/Devanik21/Yolo_Building-an-Object-Detection-Web-App)**
  by Devanik Debnath (GitHub [@Devanik21](https://github.com/Devanik21)) — the original
  FastAPI + Streamlit object detection web app, MIT licensed.

The upstream commit history is preserved in this repository's log, and the original MIT
copyright notice is retained unchanged in [LICENSE](LICENSE) next to the notice covering
the changes made here.

Relative to upstream, this repository reworks the project around a single shared detection
module, and adds the FastAPI service endpoints, the CLI tools, the one-click launcher, the
`.env` configuration layer, the request guards and this documentation.

YOLOv8 itself is provided by [Ultralytics](https://github.com/ultralytics/ultralytics) and
is licensed separately under AGPL-3.0. Please review that license before commercial use.

---

## License

Released under the [MIT License](LICENSE). The retained upstream notice and the notice for
the modifications made here are both recorded in that file.
