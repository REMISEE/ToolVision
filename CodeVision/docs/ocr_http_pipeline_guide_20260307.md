# CodeImageTool OCR HTTP Pipeline 说明（2026-03-07）

本文基于当前代码实现，说明 `code_image_tool` 中 OCR 的真实调用链路、每个函数职责、参数来源与默认值、以及实际业务建议传参方式。
  这些参数的用途（按是否常用）

  高频常用（建议优先考虑）：

  - use_layout_detection：是否启用版面检测（一般开）
  - layout_shape_mode：框几何表示（auto 常用）
  - max_new_tokens：输出长度上限
  - temperature / top_p：生成稳定性与多样性
  - visualize：是否返回可视化图（调试时开）

  低频场景化：

  - use_doc_orientation_classify、use_doc_unwarping：文档方向/矫正
  - layout_threshold、layout_nms、layout_unclip_ratio：检测后处理调参
  - prompt_label：特定 prompt 模式（通常只在特定模式下有效）
  - restructure_pages、merge_tables、relevel_titles、concatenate_pages：多页重构（单图一般不需要）

适用代码版本（本地路径）：
- `verl/tools/code_image_tool.py`
- `recipe/codevision/config/code_image_tool_config.yaml`

---

## 1. 总体链路（从 `_call_ocr_assist` 到 OCR 服务）

1. 在工具执行代码里调用 `_call_ocr_assist(...)`  
2. `_call_ocr_assist` 调 `self._call_external_model("paddleocr_vl", image, kwargs)`  
3. `CodeImageTool` 根据 `external_worker_mode` 找到 OCR worker（split 模式是 `OCRModelWorker`）  
4. `OCRModelWorker.infer(...)` 调 `PaddleOCRVLAdapter.infer(...)`  
5. `PaddleOCRVLAdapter` 组装 `/layout-parsing` 请求并 `POST`  
6. 若 `restructurePages=true`，再调用 `/restructure-pages`  
7. 解析结果，返回统一结构：
   - `images`: 结果图（优先服务返回的 `outputImages`，否则原图）
   - `text`: 文本摘要（优先 `markdown.text`）
   - `meta`: 原始结构化结果（用于调试/存档）

---

## 2. 核心函数逐个说明

### 2.1 `_call_ocr_assist(image_index=None, image_obj=None, **kwargs)`
位置：`CodeImageTool._create_safe_globals` 内部定义。  
作用：
- 选择输入图（`image_obj` 优先，其次 `image_index`）
- 把 `kwargs` 原样传给 OCR adapter
- 返回 `{image, images, text, meta}`
- 并把返回的首图设为当前活动图，便于后续链式处理

---

### 2.2 `_call_external_model(model_name, image, kwargs)`
作用：
- 根据模型名选择对应 worker（`paddleocr_vl` -> OCR worker）
- `ray.get(worker.infer.remote(...))` 获取结果
- 统一包装返回格式（一定包含 `image/images/text/meta`）

---

### 2.3 `OCRModelWorker.infer(...)`
作用：
- OCR 的专用 actor
- 仅允许模型名 `paddleocr_vl`
- 调用 `PaddleOCRVLAdapter.infer(...)`

---

### 2.4 `PaddleOCRVLAdapter.__init__/initialize`
作用：
- 读取并缓存 OCR HTTP 服务配置
- 构建两个接口 URL：
  - `{serving_base_url}/layout-parsing`
  - `{serving_base_url}/restructure-pages`

默认值（代码默认）：
- `serving_base_url`: `http://127.0.0.1:8080`
- `request_timeout`: `180`
- `default_file_type`: `1`（图像）
- `visualize`: `None`（不强制，除非 YAML 配置）
- `restructure_pages`: `False`

---

### 2.5 `PaddleOCRVLAdapter._build_layout_payload(image, kwargs)`
作用：
- 把 PIL 图像转 base64，填入 `file`
- 组装 `/layout-parsing` 的请求体
- 处理参数映射（snake_case -> API 的 CamelCase）

内置映射字段（可从 `_call_ocr_assist` 传）：
- `file_type` -> `fileType`
- `use_doc_orientation_classify` -> `useDocOrientationClassify`
- `use_doc_unwarping` -> `useDocUnwarping`
- `use_layout_detection` -> `useLayoutDetection`
- `use_chart_recognition` -> `useChartRecognition`
- `use_seal_recognition` -> `useSealRecognition`
- `use_ocr_for_image_block` -> `useOcrForImageBlock`
- `layout_threshold` -> `layoutThreshold`
- `layout_nms` -> `layoutNms`
- `layout_unclip_ratio` -> `layoutUnclipRatio`
- `layout_merge_bboxes_mode` -> `layoutMergeBboxesMode`
- `layout_shape_mode` -> `layoutShapeMode`
- `prompt_label` -> `promptLabel`
- `format_block_content` -> `formatBlockContent`
- `repetition_penalty` -> `repetitionPenalty`
- `temperature` -> `temperature`
- `top_p` -> `topP`
- `min_pixels` -> `minPixels`
- `max_pixels` -> `maxPixels`
- `max_new_tokens` -> `maxNewTokens`
- `merge_layout_blocks` -> `mergeLayoutBlocks`
- `markdown_ignore_labels` -> `markdownIgnoreLabels`
- `vlm_extra_args` -> `vlmExtraArgs`
- `prettify_markdown` -> `prettifyMarkdown`
- `show_formula_number` -> `showFormulaNumber`
- `restructure_pages` -> `restructurePages`
- `merge_tables` -> `mergeTables`
- `relevel_titles` -> `relevelTitles`
- `visualize` -> `visualize`

补充逻辑：
- 若请求里没给 `visualize`，会使用 YAML 的 `visualize` 默认值（若配置了）
- 若请求里没给 `restructurePages`，会使用 YAML 的 `restructure_pages` 默认值

---

### 2.6 `PaddleOCRVLAdapter.infer(image, kwargs)`
作用：
1. 调 `/layout-parsing`
2. 若 `restructurePages=True` 且有页面结果，再调 `/restructure-pages`
3. 提取文本：
   - 优先 `markdown.text`
   - 其次 `prunedResult` 的 JSON 字符串
4. 选输出图：
   - 优先服务返回 `outputImages`
   - 否则回退原图

返回：
- `images`: `list[PIL.Image]`
- `text`: `str`
- `meta`: `{"model": "paddleocr_vl_http", "layout_result": ..., "used_restructure_pages": ...}`

---

## 3. 参数优先级与默认值

当前优先级：
1. `_call_ocr_assist(..., **kwargs)` 显式传入
2. YAML（`external_models.paddleocr_vl`）默认配置
3. 代码内默认值

当前 YAML（已生效）：
- `serving_base_url: "http://127.0.0.1:8080"`
- `request_timeout: 180`
- `default_file_type: 1`
- `visualize: false`
- `restructure_pages: false`

---

## 4. 什么参数“能用”，什么参数“不该从这里传”

### 4.1 能从 `_call_ocr_assist` 传的
就是上面映射表里的字段（这些会进入 `/layout-parsing` 或 `/restructure-pages` 请求体）。

### 4.2 不建议从 `_call_ocr_assist` 传的
以下属于“服务启动/模型部署参数”，应在服务端配置，不是单次请求参数：
- `device`
- `pipeline_version`
- `vl_rec_model_name / vl_rec_model_dir`
- `layout_detection_model_name / layout_detection_model_dir`
- 以及其他模型初始化相关参数

---

## 5. 是否需要 `/restructure-pages`

结论：
- 单图“图生文”通常不需要，默认关就好
- 多页 PDF、跨页表格、标题层级重建场景再开

你当前配置是 `restructure_pages: false`，所以默认只调 `/layout-parsing`，链路更轻。

---

## 6. 推荐传参策略（实战）

### 6.1 方案 A：最简（推荐默认）
YAML 固定默认；调用只写：

```python
ocr = _call_ocr_assist()
result = ocr["image"]
print(ocr["text"][:500])
```

### 6.2 方案 B：轻量覆盖（常用）
```python
ocr = _call_ocr_assist(
    prompt_label="ocr",
    use_layout_detection=True,
    layout_shape_mode="auto",
    max_new_tokens=1024
)
```

### 6.3 方案 C：调试可视化
```python
ocr = _call_ocr_assist(
    visualize=True,
    prettify_markdown=True
)
```

### 6.4 方案 D：仅多页重构时开启
```python
ocr = _call_ocr_assist(
    restructure_pages=True,
    merge_tables=True,
    relevel_titles=True,
    concatenate_pages=True
)
```

---

## 7. 常见疑问

### Q1: `field_map` 是不是表示“全部参数都要传”？
不是。`field_map` 只是“如果你传了，就映射并透传”；不传就走默认。

### Q2: 我们现在图生文需要很多参数吗？
不需要。多数场景 `_call_ocr_assist()` 或最多 2-4 个覆盖参数。

### Q3: 为什么 `fileType` 默认 1？
当前 helper 输入是 PIL 图像，默认按图像请求服务。  
如果未来接 PDF，需要扩展 helper 的输入形态，而不是只改 `fileType`。

---

## 8. 建议下一步

1. 保持 YAML 默认，不要在每次 tool call 塞大量参数。  
2. 只把“任务相关参数”放在 `_call_ocr_assist` kwargs。  
3. 后续若参数继续增多，建议再加一层“白名单参数集”（基础版/文档版/调试版）以降低 prompt 和调用复杂度。
