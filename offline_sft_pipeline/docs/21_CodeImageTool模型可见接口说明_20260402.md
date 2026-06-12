# 21 CodeImageTool 模型可见接口说明

日期：2026-04-02  
状态：当前实现说明

## 1. 目标

这份说明只描述应该暴露给 planner / executor 的简化接口，不展开底层 HTTP client 细节。

原则：

- 模型只学习 helper 用法
- OCR 调参默认放在后台配置
- 模型只在必要时显式切图

---

## 2. 当前建议暴露给模型的 helper

### `_call_ocr_assist(image_index=None, image_obj=None, **kwargs)`

作用：

- 读取当前图像中的文本

建议：

- 默认直接调用
- 如果前一步已经裁好图，优先写 `image_obj=crop["image"]`
- 除非明确需要，不要让模型调 OCR 细参数

### `_call_manual_box(x1, y1, x2, y2, image_index=None, image_obj=None, outline="lime", width=2, label=None, label_fill=None)`

作用：

- 用显式坐标在图上画框
- 不依赖外部检测模型

建议：

- 当模型已经能自己判断目标坐标时，优先用这个 helper，而不是再走检测模型
- 如果只是简单标注一个局部区域，这比 `_call_ground_box(...)` 更直接

### `_call_manual_crop(x1, y1, x2, y2, image_index=None, image_obj=None, padding=0)`

作用：

- 用显式坐标裁剪局部区域
- 不依赖外部检测模型

建议：

- 当模型已经知道裁剪范围时，优先用这个 helper，而不是再走 `_call_dino_crop(...)`
- 如果后续还要 OCR，常见写法是先 `_call_manual_crop(...)`，再 `_call_ocr_assist(image_obj=...)`

### `_call_ground_box(text_prompt, image_index=None, image_obj=None, box_threshold=0.35, text_threshold=0.25, **kwargs)`

作用：

- 定位与 `text_prompt` 匹配的目标
- 返回画框图

### `_call_sam_mask(text_prompt, image_index=None, image_obj=None, box_threshold=0.35, text_threshold=0.25, multimask_output=False, mask_alpha=0.45, draw_box_on_mask=True, **kwargs)`

作用：

- 定位目标并生成高亮 mask 图

### `_call_dino_crop(text_prompt, image_index=None, image_obj=None, max_crops=1, padding=0, box_threshold=0.35, text_threshold=0.25, **kwargs)`

作用：

- 根据 box 检测结果裁剪目标区域

说明：

- 当前模型可见说明里只推荐 box-based crop
- 不建议把 `based_on="mask"` 暴露给模型

### `_call_blur_bg(text_prompt, image_index=None, image_obj=None, blur_radius=8.0, box_threshold=0.35, text_threshold=0.25, **kwargs)`

作用：

- 保持目标清晰并模糊背景

---

## 3. 图像选择规则

模型必须理解下面 3 条：

1. 如果 `image_obj` 和 `image_index` 都不传，helper 会使用当前 step 的默认输入图。
2. 如果要在同一步里继续处理前一个 helper 的输出，写 `image_obj=prev["image"]`。
3. 如果要回到另一张候选输入图，例如 root image，写 `image_index=...`。

补充：

- helper 成功后，运行环境会把返回的 `result["image"]` 写回当前活跃的 `image` / `img`
- 这对后续直接做 PIL 操作有用
- 但后续 helper 不会自动吃这张活跃图，仍然建议显式传 `image_obj`

---

## 4. OCR 默认配置策略

当前建议是：

- 把 OCR 细参数写到后台默认配置
- 模型侧默认只写 `_call_ocr_assist()`

推荐后台默认值：

- `use_doc_orientation_classify = False`
- `use_doc_unwarping = False`
- `use_textline_orientation = True`
- `text_det_limit_side_len = 960`
- `text_det_limit_type = "max"`
- `text_det_thresh = 0.4`
- `text_det_box_thresh = 0.7`
- `text_rec_score_thresh = 0.6`
- `return_word_box = False`

`visualize` 是否默认打开，应由运行环境决定，不建议让模型负责。

---

## 5. 推荐给模型的最小用法模板

```python
box = _call_manual_box(120, 80, 260, 180, label="serial")
crop = _call_manual_crop(120, 80, 260, 180, image_obj=box["image"], padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
print(ocr["text"])
result = crop["image"]
```

如果需要先让检测模型帮助定位，再基于其结果继续处理，也可以写：

```python
box = _call_ground_box("serial number")
crop = _call_dino_crop("serial number", image_obj=box["image"], padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
print(ocr["text"])
result = crop["image"]
```

如果已经有合适的当前默认图，也可以更短：

```python
crop = _call_dino_crop("serial number", padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
result = crop["image"]
```

---

## 6. 不建议暴露给模型的内容

- HTTP service client 名称
- PaddleOCR 请求字段全表
- `based_on="mask"` 这类低频分支参数
- 运行时 store / artifact 内部结构

这些内容应保留在后端实现、配置文件或工程文档中。
