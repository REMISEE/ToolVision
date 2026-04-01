# GroundSAM2 Helper 参数参考（CodeImageTool）

本文对应 `code_image_tool` 里这 4 个函数：

- `_call_ground_box(...)`
- `_call_sam_mask(...)`
- `_call_dino_crop(...)`
- `_call_blur_bg(...)`

目标：解释每个参数的作用、默认值、是否必须显式传，以及为什么很多参数不在 yaml 里。

---

## 1. 先回答核心问题：为什么 yaml 没写这些参数？

因为这两类参数是不同层级：

1. `yaml`（或 `build_tool_config`）主要放**静态初始化参数**  
   例如：模型路径、设备、Ray actor 资源、OCR 服务地址等。

2. 4 个 helper 的参数是**运行时调用参数**  
   例如：`padding`、`max_crops`、`mask_alpha`、`blur_radius`、`multimask_output`。  
   这些通常每个样本/每轮推理都可能不同，所以放在函数调用里，而不是全局 yaml。

结论：这是设计上的“静态配置 vs 动态调用参数”分层，不是漏配。

---

## 2. 参数是否必须显式指定？

不是。除 `text_prompt` 外，其它都有默认值。

- 你可以只传最小参数：`_call_ground_box("car.")`
- 也可以按任务覆盖：`_call_dino_crop("plate.", padding=12, based_on="mask")`

---

## 3. 默认值从哪里来（优先级）

对这 4 个 helper，参数优先级是：

1. 调用时显式传入（最高）
2. helper 函数签名默认值
3.（少数参数）adapter 配置默认值/兜底值

注意：

- `box_threshold` / `text_threshold` 在 helper 里已经有默认值（0.35/0.25），因此通常不会走到 yaml 的同名值。
- `text_prompt` 必填；如果你绕开 helper 直接调 external model，才可能使用 adapter 的 `default_text_prompt`。

---

## 4. 四个函数参数清单（按实际调用）

## 4.1 `_call_ground_box`

签名：

```python
_call_ground_box(
  text_prompt: str,
  image_index: Optional[int] = None,
  image_obj: Optional[Any] = None,
  box_threshold: float = 0.35,
  text_threshold: float = 0.25,
  **kwargs
)
```

参数说明：

- `text_prompt`：必填，检测文本提示词（建议英文、句号分隔）。
- `image_index`：可选，选第几张输入图（多图场景）。
- `image_obj`：可选，直接传 PIL 图像；优先级高于 `image_index`。
- `box_threshold`：可选，检测框阈值。
- `text_threshold`：可选，文本匹配阈值。
- `**kwargs`：透传扩展参数（一般不需要）。

输出：

- `image`：画框后的图。
- `images`：图像列表（通常 1 张）。
- `text`：例如 `GroundedSAM2(box) detected N objects.`
- `meta.annotations`：每个框的类别、bbox、置信度。

---

## 4.2 `_call_sam_mask`

签名：

```python
_call_sam_mask(
  text_prompt: str,
  image_index: Optional[int] = None,
  image_obj: Optional[Any] = None,
  box_threshold: float = 0.35,
  text_threshold: float = 0.25,
  multimask_output: bool = False,
  mask_alpha: float = 0.45,
  draw_box_on_mask: bool = True,
  **kwargs
)
```

新增参数（相对 box）：

- `multimask_output`：是否让 SAM2 产出多 mask 并自动选 best。
- `mask_alpha`：mask 半透明叠加强度（0~1）。
- `draw_box_on_mask`：mask 图上是否继续叠加检测框。

输出：

- `image`：mask 高亮图。
- `meta.mask_scores`：mask 分数。
- `meta.annotations`：检测框信息。

---

## 4.3 `_call_dino_crop`

签名：

```python
_call_dino_crop(
  text_prompt: str,
  image_index: Optional[int] = None,
  image_obj: Optional[Any] = None,
  based_on: str = "box",            # "box" | "mask"
  detection_index: int = 0,
  max_crops: int = 1,
  padding: int = 0,
  box_threshold: float = 0.35,
  text_threshold: float = 0.25,
  multimask_output: bool = False,
  **kwargs
)
```

新增参数（裁剪相关）：

- `based_on`：
  - `"box"`：按 GroundingDINO 框裁剪
  - `"mask"`：按 SAM2 mask 外接框裁剪
- `detection_index`：选第几个检测目标。
- `max_crops`：最多返回多少张裁剪图。
- `padding`：裁剪扩边像素（上下左右）。

输出：

- `image`：第一张 crop。
- `images`：全部 crop 列表。
- `meta.crop_boxes`：实际裁剪使用的框（xyxy）。
- `meta.based_on`：本次是按 box 还是 mask 裁剪。

---

## 4.4 `_call_blur_bg`

签名：

```python
_call_blur_bg(
  text_prompt: str,
  image_index: Optional[int] = None,
  image_obj: Optional[Any] = None,
  blur_radius: float = 8.0,
  box_threshold: float = 0.35,
  text_threshold: float = 0.25,
  multimask_output: bool = False,
  **kwargs
)
```

新增参数：

- `blur_radius`：背景高斯模糊半径（越大越模糊）。

输出：

- `image`：前景清晰、背景模糊图。
- `meta.blur_radius`：本次使用的模糊半径。

---

## 5. 这些参数在 demo 里是怎么喂进去的

`demo_code_image_tool_external.py` 的两条通路：

1. `parse_args -> build_tool_config`：喂给模型初始化（静态）。
2. `parse_args -> demo_cases`：拼进 helper 调用代码（动态）。

因此你看到“同名参数在 yaml 没出现”是正常的，因为 demo 是直接从 CLI 把动态参数拼到调用代码里。

---

## 6. 当前 demo 覆盖了哪些参数，哪些没覆盖

已覆盖（CLI 有）：

- `focus_prompt`
- `box_threshold`
- `text_threshold`
- `multimask_output`
- `mask_alpha`
- `blur_radius`
- `crop_based_on`
- `crop_detection_index`
- `crop_max_crops`
- `crop_padding`

未在 CLI 暴露但 helper 支持：

- `draw_box_on_mask`
- `image_index`
- `image_obj`

说明：这几个不是不能用，只是 demo 脚本没做成命令行参数；你在工具代码里仍可直接传。

---

## 7. 实际跑的时候怎么配

建议：

1. 把“稳定不变”的放配置（路径、设备、资源）。
2. 把“任务相关”的放调用参数（prompt、阈值、padding、blur）。

最小调用示例：

```python
box = _call_ground_box("black rectangle.", box_threshold=0.5, text_threshold=0.3)
crop = _call_dino_crop("black rectangle.", based_on="mask", padding=8, max_crops=1)
result = crop["image"]
```

---

## 8. 一个实践提醒

如果希望 `box_threshold/text_threshold` 全局统一由 yaml 控制，需要把 helper 里的默认参数策略改掉（现在 helper 会主动传默认 0.35/0.25）。  
当前实现下，最稳妥方式是：在调用时显式传你要的阈值。

