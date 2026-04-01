# CodeImageTool External Demo 运行分析（2026-03-05）

## 1. 本次运行结论

本次 `demo_code_image_tool_external.py` 已经基本跑通 GroundSAM2 相关链路。

- 成功：
  - `Case 0: basic_pil`
  - `Case 2: ground_box`
  - `Case 3: sam_mask`
  - `Case 4: dino_crop`
  - `Case 5: blur_bg`
  - `Case 6: focus_alias`
- 失败：
  - `Case 1: ocr_assist`（原因：未安装 `paddleocr`，与 GroundSAM2 无关）

本次日志文件：
- `outputs/code_image_tool_external_grounded/test11.log`

本次输出图片：
- `outputs/code_image_tool_external_grounded/00_basic_pil.png`
- `outputs/code_image_tool_external_grounded/02_ground_box.png`
- `outputs/code_image_tool_external_grounded/03_sam_mask.png`
- `outputs/code_image_tool_external_grounded/04_dino_crop.png`
- `outputs/code_image_tool_external_grounded/05_blur_bg.png`
- `outputs/code_image_tool_external_grounded/06_focus_alias.png`

图片尺寸统计：

| 文件 | 尺寸 (W,H) | 说明 |
|---|---:|---|
| 00_basic_pil.png | 364 x 644 | 旋转后的基线图（PIL） |
| 02_ground_box.png | 644 x 364 | GroundingDINO 检测框可视化 |
| 03_sam_mask.png | 644 x 364 | SAM2 掩码半透明高亮 |
| 04_dino_crop.png | 143 x 63 | 检测区域裁剪图 |
| 05_blur_bg.png | 644 x 364 | 前景保留、背景高斯模糊 |
| 06_focus_alias.png | 644 x 364 | `_call_focus` 别名输出（等价 box） |


## 2. 你问的关键问题：`dino_crop` 为什么这么准？

结论：这次 `dino_crop` 的精准主要来自 **GroundingDINO 检测框**，不是手工给框。

原因如下：

1. 本次 external demo 并没有传入手工 bbox 参数。  
2. `Case 4` 的代码走的是 `_call_dino_crop(...)`，内部先做 `_run_grounding(...)` 得到检测框。  
3. 你这次默认参数是：
   - `based_on="box"`
   - `detection_index=0`
   - `max_crops=1`
   - `padding=0`
4. 因此裁剪逻辑就是：
   - 取第一个检测框（置信度排序后的第 0 个）
   - 不扩边（padding=0）
   - 裁一张图（max_crops=1）

这就会得到一个紧凑裁剪（你这次是 `143x63`）。

补充：
- 如果改成 `based_on="mask"`，会先用 SAM2 生成 mask，再把 mask 转最小外接框裁剪，通常比纯 box 更贴边。
- 所以“是 GroundSAM2 的功劳还是你给框”：这次是模型（GroundingDINO，若 `based_on=mask` 则加上 SAM2）自动给框，不是人工 bbox。


## 3. 本次 demo 的数据通路（真正调用时）

单个 case 的调用链：

1. `demo_code_image_tool_external.py` 构建 code 字符串（比如 `_call_dino_crop(...)`）。  
2. `CodeImageTool.execute(...)` 执行这段 code。  
3. code 中 helper（如 `_call_dino_crop`）调用 `self._call_external_model(...)`。  
4. `ExternalModelWorker.infer(...)` 路由到 `GroundedSAM2Adapter`。  
5. `GroundedSAM2Adapter.infer(...)` 根据 `_operation` 分发到：
   - `infer_box`
   - `infer_mask`
   - `infer_dino_crop`
   - `infer_blur_bg`
6. 每个分支返回统一结构：
   - `images: list[PIL.Image]`
   - `text: str`
   - `meta: dict`
7. helper 把结果中的首图写回当前执行上下文（`image/img`），并返回整包结果。

注意：
- demo 里每个 case 是一次独立 `execute` 调用，默认都从同一张原图 `image_index=0` 开始，不会自动串联上一个 case 的输出图。


## 4. 你这次实际用了哪些 demo 参数

你命令里显式传了：

- `--image /mnt/d/sdu/ToolVision/CodeVision/tmp_demo_input.png`
- `--device cuda`
- `--external-worker-name "2026-03-05_0035_test11"`
- `--out-dir outputs/code_image_tool_external_grounded`

其余走脚本默认值（当前版本）：

| 参数 | 默认值 | 影响 |
|---|---|---|
| `--focus-prompt` | `"car. tire."` | 决定 GroundingDINO 找什么 |
| `--box-threshold` | `0.35` | 检测框阈值 |
| `--text-threshold` | `0.25` | 文本匹配阈值 |
| `--multimask-output` | `False` | SAM2 单 mask 模式 |
| `--mask-alpha` | `0.45` | mask 叠加透明度 |
| `--blur-radius` | `8.0` | 背景模糊半径 |
| `--crop-based-on` | `"box"` | 裁剪依据（box/mask） |
| `--crop-detection-index` | `0` | 选第几个检测目标 |
| `--crop-max-crops` | `1` | 最多返回几张裁剪图 |
| `--crop-padding` | `0` | 裁剪扩边像素 |
| `--external-worker-num-gpus` | `1.0` | External worker 的 GPU 资源 |
| `--external-worker-num-cpus` | `2.0` | External worker 的 CPU 资源 |


## 5. 各 helper 函数作用与输入输出

### `_call_ground_box`

- 作用：只做 GroundingDINO 检测并在原图上画框。
- 核心参数：`text_prompt`, `box_threshold`, `text_threshold`
- 返回：
  - `image`：画框图
  - `images`：同上（列表）
  - `text`：如 `GroundedSAM2(box) detected N objects.`
  - `meta`：`annotations`（类别、bbox、置信度）

### `_call_sam_mask`

- 作用：先 Grounding，再用 SAM2 分割，并叠加半透明 mask。
- 核心参数：`text_prompt`, `multimask_output`, `mask_alpha`, `draw_box_on_mask`
- 返回：
  - `image`：mask 高亮图
  - `meta.mask_scores`：mask 分数

### `_call_dino_crop`

- 作用：基于 box 或 mask 返回裁剪图。
- 核心参数：`based_on`, `detection_index`, `max_crops`, `padding`
- 返回：
  - `images`：裁剪图列表
  - `image`：第一张裁剪图
  - `meta.crop_boxes`：实际用于裁剪的 xyxy

### `_call_blur_bg`

- 作用：前景保留清晰，背景高斯模糊。
- 核心参数：`blur_radius`
- 返回：背景模糊图 + 注释框信息

### `_call_focus`

- 作用：兼容别名，行为等价 `_call_ground_box`。


## 6. 代码层关键实现点（本次修复后）

GroundSAM2 跑通依赖了以下关键适配：

1. GroundingDINO 输入前做官方预处理（`RandomResize + ToTensor + Normalize`），避免把 numpy 直接传给 `predict()`。  
2. SAM2 `build_sam2(...)` 的配置参数改为 Hydra `config_name` 语义（`configs/...`），不是绝对路径。  
3. 本地路径解析和包名兼容（`grounding_dino` / `groundingdino`）已做适配。  
4. 文本编码器已改为本地目录（不依赖联网 HF）。


## 7. 如何进一步验证“裁剪到底用了哪个框”

当前 demo 的 case 只打印了 `text`，没有打印 `meta`。  
如果你要精确核对 crop 框坐标，建议在 Case 4 把这行加上：

```python
print("crop meta:", crop_res["meta"])
```

你会直接看到 `crop_boxes`，可与 `04_dino_crop.png` 尺寸一一对应验证。


## 8. 当前剩余问题

只剩 OCR：

- 报错：`No module named 'paddleocr'`
- 影响范围：仅 `Case 1: ocr_assist`
- 不影响 GroundSAM2 的 box/mask/crop/blur 主流程

