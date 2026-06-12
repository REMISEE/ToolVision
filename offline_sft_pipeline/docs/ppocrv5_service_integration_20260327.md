# PP-OCRv5 服务化接入 CodeVision

日期：2026-03-27

## 1. 结论

`CodeVision/scripts/export_textvqa_paddleocr.py` 不应该直接改造成常驻服务。

原因很简单：

- 这个脚本是离线批处理脚本
- 它自己负责读 Hugging Face dataset、做去重、分片、多进程分卡
- 它在 worker 进程里直接 `PaddleOCR(...)` 初始化并批量 `predict()`

这和线上常驻服务的职责边界不一样。

如果要服务化，推荐做法是：

1. 用 Paddle 官方的 `PaddleX serving` 起常驻 OCR HTTP 服务
2. `CodeImageTool` 只保留 OCR client
3. `_call_ocr_assist()` 继续做统一 helper，不改调用方式

## 2. 官方推荐的服务化方式

官方文档当前推荐直接用 PaddleX serving 起 OCR 服务：

- Serving guide:
  `https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/deployment/serving.html`
- OCR pipeline API:
  `https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html`

最小验证命令：

```bash
paddlex --install serving
paddlex --serve --pipeline OCR --host 0.0.0.0 --port 8080
```

服务启动后，OCR 接口是：

```text
POST /ocr
```

请求体核心字段：

- `file`
- `fileType`
- `useDocOrientationClassify`
- `useDocUnwarping`
- `useTextlineOrientation`
- `textDetLimitSideLen`
- `textDetLimitType`
- `textDetThresh`
- `textDetBoxThresh`
- `textDetUnclipRatio`
- `textRecScoreThresh`
- `visualize`

## 3. 如果你要尽量复用现在脚本里的 PP-OCRv5 参数

你现在批处理脚本里关心的主要是这些参数：

- `PP-OCRv5_server_det`
- `PP-OCRv5_server_rec`
- `text_det_limit_side_len`
- `text_det_limit_type`
- `text_det_thresh`
- `text_det_box_thresh`
- `text_rec_score_thresh`

官方做法不是把这些参数硬塞进 daemon 脚本，而是导出 PaddleX pipeline YAML，再把模型和阈值写进配置文件。

先导出默认 OCR pipeline 配置：

```python
from paddleocr import PaddleOCR

pipeline = PaddleOCR()
pipeline.export_paddlex_config_to_yaml("PaddleOCR.yaml")
```

然后在 `PaddleOCR.yaml` 里重点改这几个位置：

```yaml
SubModules:
  TextDetection:
    model_name: PP-OCRv5_server_det
    model_dir: null
    limit_side_len: 960
    limit_type: max
    thresh: 0.4
    box_thresh: 0.7
  TextRecognition:
    model_name: PP-OCRv5_server_rec
    model_dir: null
    score_thresh: 0.6
```

再用这个 YAML 起服务：

```bash
paddlex --serve --pipeline ./PaddleOCR.yaml --host 0.0.0.0 --port 8080
```

如果你有自己下载好的 det/rec 权重，把 `model_dir` 指到本地路径即可。

## 4. CodeVision 侧怎么接

这次代码已经补了一个新的 PP-OCRv5 HTTP client，接的是官方 `/ocr`。

`_call_ocr_assist()` 不需要改调用方式，只需要把配置切到 service 模式。

推荐配置片段：

```yaml
external_call_mode: "service"
ocr_model_name: "paddleocr"

external_services:
  paddleocr:
    base_url: "http://127.0.0.1:8080"
    request_timeout: 180
    default_file_type: 1
    visualize: false
    line_y_threshold: 0.6
```

调用代码保持不变：

```python
ocr = _call_ocr_assist(
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_det_thresh=0.4,
    text_det_box_thresh=0.7,
    text_rec_score_thresh=0.6,
    visualize=False,
)
print(ocr["text"])
result = ocr["image"]
```

## 5. 这次代码里已经做的事情

- `CodeImageTool` service mode 新增了 `paddleocr` / `paddleocr_v5` / `ocr` client 注册
- `_call_ocr_assist()` 不再写死 `paddleocr_vl`，会按 `ocr_model_name` 选 client
- 新 client 会把 `/ocr` 返回的 `prunedResult` 规整成：
  - `text`
  - `meta.ocr_pages`
  - `meta.num_ocr_items`
- 文本聚合逻辑复用了你批处理脚本那套 `rec_texts + rec_boxes -> line merge`

## 6. 推荐迁移顺序

1. 先单独起 `paddlex --serve --pipeline OCR`
2. 用 `curl` 或 demo 脚本打通 `/ocr`
3. 把 CodeVision 配置切到 `external_call_mode: service`
4. 把 `ocr_model_name` 设成 `paddleocr`
5. 等服务模式稳定后，再删旧的 `OCRModelWorker`
