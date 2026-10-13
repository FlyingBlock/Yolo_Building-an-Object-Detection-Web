# YOLO 目标检测 Web 应用

一个用 YOLOv8 做目标检测的小项目：网页里上传图片、用摄像头拍一张、或者贴一个图片链接，页面上直接画出框、类别和置信度。

检测部分只有一份实现（`detector.py`），网页、接口和命令行都调用它，所以三种方式跑出来的结果完全一致。

界面和接口文档都是中文。

## 功能

- **三种输入方式**：上传图片（可一次选多张）、摄像头拍照、图片 URL。
- **实时进度**：检测过程中显示百分比、各阶段耗时（缩放、加载权重、推理、画框、编码）、图片信息以及 GPU 占用和显存。
- **结果展示**：识别框、类别、置信度，外加汇总表格和每类数量图表，标注后的图片和 CSV 都能下载。
- **可选模型**：内置 YOLOv8n/s/m/l/x 等，也可以把自己的 `.pt` 权重放进 `models/` 或在页面上传。
- **参数可调**：置信度、NMS IoU、类别过滤、最大输入尺寸。
- **命令行**：`detect.py` 处理单张图片，`batch_detect.py` 处理整个文件夹，输出标注图、JSON 和 CSV。
- **Windows 一键启动**：双击 `Launch.bat` 就行。

## 环境要求

- Python 3.9 及以上（推荐 3.11 或 3.13）
- 一键启动脚本目前只针对 Windows，接口、界面和命令行在其他系统上同样可用
- 没有显卡也能跑，有 CUDA 会自动用

模型权重不在仓库里。第一次使用时 Ultralytics 会自动下载（`yolov8n` 约 6 MB）。

## 快速开始

### Windows：双击启动

克隆仓库后双击根目录的 `Launch.bat` 即可。它会找一个合适的 Python、装好缺的依赖、把后端和前端一起启动，然后打开浏览器。

找 Python 的顺序是：`-PythonPath` 参数 → 名为 `yoloweb` 的 conda 环境 → 项目里的 `.venv` → 系统里的 Python 3.9+。

常用参数：

```powershell
# 换端口，或者不自动打开浏览器
.\Launch.bat -ApiPort 8010 -UiPort 8511 -NoBrowser

# 只装依赖，不启动服务
.\Launch.bat -SetupOnly

# 查看全部参数
powershell -ExecutionPolicy Bypass -File start.ps1 -?
```

### 手动启动

```bash
git clone https://github.com/FlyingBlock/Yolo_Building-an-Object-Detection-Web.git
cd Yolo_Building-an-Object-Detection-Web

python -m venv .venv
# Windows: .venv\Scripts\activate        macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt

# 终端 1：后端
uvicorn api:app --reload --port 8000

# 终端 2：前端
streamlit run app.py
```

浏览器打开 <http://localhost:8501> 就是界面，<http://127.0.0.1:8000/docs> 是接口文档。

## 使用方法

### 网页

左侧边栏选择用 HTTP 调用后端还是在本进程内直接推理（不想多开一个服务就选后者），下面可以改 API 地址、模型、置信度和 IoU。

主区域有三个标签页：上传图片、摄像头、图片 URL。检测完每张图片会占一张卡片，卡片里左边是原图、右边是标注结果，第三列是这张图的统计和导出按钮。

### 接口

主要接口：

| 接口 | 说明 |
|---|---|
| `POST /detect` | 检测并返回 JSON，可选返回标注图的 data URI |
| `POST /detect/stream` | 同上，但会先把每个阶段的进度逐行返回，最后才是结果 |
| `POST /detect/image` | 检测并直接返回标注后的 PNG |
| `GET /models` | 列出内置、自定义和已缓存的模型 |
| `POST /models/upload` | 安装自定义 `.pt` 权重 |
| `POST /warmup` | 提前加载权重并跑一次预热推理 |
| `GET /health` | 服务和推理设备状态 |

```bash
# 返回 JSON
curl -X POST http://localhost:8000/detect \
  -F 'file=@samples/bus.jpg' \
  -F 'confidence=0.5' \
  -F 'model=yolov8n'

# 直接拿标注图
curl -X POST http://localhost:8000/detect/image \
  -F 'file=@samples/bus.jpg' -o annotated.png

# 预热，让第一个用户不用等
curl -X POST http://localhost:8000/warmup
```

`/detect/stream` 每行一条 JSON 记录，最后一条一定是结果；读的时候看记录内容，不要只看 HTTP 状态码：

```bash
curl -N -X POST http://localhost:8000/detect/stream \
  -F 'file=@samples/bus.jpg' -F 'confidence=0.5'
```

```json
{"type":"progress","stage":"resized","stage_index":2,"elapsed_ms":12.4,"width":480,"height":640}
{"type":"progress","stage":"inferred","stage_index":4,"elapsed_ms":247.9,"inference_ms":231.7}
{"type":"result","success":true,"count":4,"inference_ms":231.7,"total_ms":312.5,"detections":[]}
```

### 命令行

```bash
# 单张图片
python detect.py --image samples/bus.jpg --model yolov8m --save-json detections.json

# 整个文件夹
python batch_detect.py --input ./images/ --output ./results/ --model yolov8s --recursive
```

两个脚本都支持 `--confidence`、`--iou`、`--device`、`--max-size` 等参数，`detect.py` 还可以用 `--classes` 只保留指定类别，加 `--help` 可以看到完整列表。

### 自己的权重

把 `.pt` 文件放进 `models/`，或者从页面侧边栏上传，模型列表里就会多出这一项。

要注意加载 `.pt` 会执行 pickle 数据，所以 `ALLOW_CUSTOM_MODEL_UPLOAD` 建议只在接口仅对本机开放时打开。

## 配置

所有配置都有默认值，`.env` 文件可以完全不要。想改就把 `.env.example` 复制成 `.env`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEFAULT_MODEL` | `yolov8n.pt` | 默认模型 |
| `CONFIDENCE_THRESHOLD` | `0.5` | 置信度下限 |
| `IOU_THRESHOLD` | `0.45` | NMS 的 IoU 阈值 |
| `DEVICE` | `auto` | `auto`、`cpu`、`cuda:0` 或 `mps` |
| `MAX_IMAGE_SIZE` | `1280` | 推理前把长边缩到这个尺寸，`0` 表示不缩放 |
| `MAX_UPLOAD_MB` | `25` | 上传大小上限 |
| `MAX_IMAGE_PIXELS` | `64000000` | 图片像素上限，防止解压炸弹 |
| `PRELOAD_MODEL` | `0` | 是否在服务启动时就加载模型 |
| `ALLOW_CUSTOM_MODEL_UPLOAD` | `1` | 是否允许上传 `.pt` 权重 |
| `MODELS_DIR` | `models` | 自定义权重目录 |
| `CORS_ORIGINS` | `*` | 允许跨域访问的来源，逗号分隔 |
| `API_URL` | `http://127.0.0.1:8000` | 前端去哪里找后端 |

> `ALLOW_CUSTOM_MODEL_UPLOAD=1` 在本机用起来方便，但把接口暴露到网络上之前请改成 `0`。

## 目录结构

```
Yolo_Building-an-Object-Detection-Web/
├── Launch.bat               # Windows 一键启动，双击这个
├── start.ps1                # 启动脚本本体：找解释器、装依赖、起服务、开浏览器
├── api.py                   # FastAPI 后端          -> uvicorn api:app
├── app.py                   # Streamlit 前端        -> streamlit run app.py
├── detector.py              # 共用的检测实现：加载、推理、画框
├── detect.py                # 单张图片命令行
├── batch_detect.py          # 文件夹批量命令行
├── requirements.txt         # Web 应用依赖
├── requirements-atlas.txt   # 可选的额外依赖，见下
├── .env.example             # 全部配置项及默认值
├── models/                  # 放自定义 .pt 权重
├── samples/bus.jpg          # 示例图片
├── atlas_engine.py          # 可选内容，见下
├── quantum_dream_atlas.py   # 可选内容，见下
├── yolov3.cfg               # 沿用自上游的旧配置文件
├── LICENSE
└── README.md
```

### 附带的另一个小东西

`quantum_dream_atlas.py` 和 `atlas_engine.py` 是另一个 Streamlit 可视化页面，和检测功能没有关系，Web 应用也不依赖它。它的依赖单独放在一个文件里：

```bash
pip install -r requirements-atlas.txt
streamlit run quantum_dream_atlas.py
```

## 后续计划

- [ ] 视频检测，逐帧标注并导出视频
- [ ] 多路摄像头
- [ ] 用自己的数据集微调 YOLOv8
- [ ] 导出 ONNX / TensorRT
- [ ] 加上 DeepSORT 做目标跟踪

## 参与

欢迎提 issue 和 PR。改动请开分支，提交信息写清楚改了什么。

## 说明

本项目基于 [Devanik21/Yolo_Building-an-Object-Detection-Web-App](https://github.com/Devanik21/Yolo_Building-an-Object-Detection-Web-App)（作者 Devanik Debnath，MIT 许可）修改而来，上游的提交历史和 MIT 版权声明都保留在仓库里。相对上游，这里把检测逻辑收拢到一个模块，补上了 FastAPI 接口、命令行工具、Windows 一键启动、`.env` 配置和上传限制。

YOLOv8 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，采用 AGPL-3.0 许可，商用前请先确认许可条款。

## 许可证

[MIT](LICENSE)。
