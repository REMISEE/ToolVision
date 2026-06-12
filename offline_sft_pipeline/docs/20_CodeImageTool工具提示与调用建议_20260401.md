# 20 Step 4：CodeImageTool 工具提示与调用建议

日期：2026-04-01  
状态：建议稿  
目的：基于当前 `CodeVision/verl/tools/code_image_tool.py` 和 `verl/external_services` 的真实实现，整理：

- 现在 executor 代码环境里真正可调用的工具是什么
- 这些工具的返回值到底是什么
- `result` / `image_index` / `image_obj` / active image 的真实语义
- planner 和 executor 应该分别看到什么级别的工具描述

---

## 1. 先给结论

当前真正应该暴露给模型的，不是：

- HTTP service client 名字
- `GroundedSAM2HTTPClient`
- `PaddleOCRHTTPClient`

而是 `CodeImageTool` 执行沙箱里注入的 helper 函数。

当前 executor 代码里可直接调用的主 helper 是：

- `_call_ocr_assist(...)`
- `_call_manual_box(...)`
- `_call_manual_crop(...)`
- `_call_ground_box(...)`
- `_call_sam_mask(...)`
- `_call_dino_crop(...)`
- `_call_blur_bg(...)`

另外还有一个兼容别名：

- `_call_focus(...)`

但它本质等同 `_call_ground_box(...)`，不建议继续作为主推荐接口。

---

## 2. 当前运行时真正的两层接口

### 2.1 外层 runtime tool

从 orchestrator / runtime wrapper 看，真正调用的是：

- `code_image_tool(code, description, image_index)`

其中：

- `code` 是 executor 生成的 Python 代码
- `image_index` 是这一步起始输入图索引

### 2.2 内层 helper API

executor 写进 `code` 的，并不是 HTTP 请求，而是 helper 调用。

也就是说：

- 模型不应该学习 service client
- 模型应该学习 helper API

---

## 3. helper 的统一返回结构

所有 helper 最终都遵循统一返回结构：

- `image`
- `images`
- `text`
- `meta`

当前外部模型 helper 的统一封装位置是：

- `CodeImageTool._call_external_model(...)`

本地 helper 也会返回同样的结构，只是不经过外部模型调用。

返回结构是：

```python
{
    "image": images[0] if images else image,
    "images": images,
    "text": result.get("text", ""),
    "meta": result.get("meta", {}),
}
```

所以对 executor 来说，最重要的是：

- `helper_result["image"]`
- `helper_result["text"]`
- `helper_result["meta"]`

### 3.1 `image`

主输出图，单张 `PIL.Image`。

### 3.2 `images`

全部输出图列表。

例如：

- `dino_crop` 可能返回多个 crop
- `manual_crop` 当前返回 1 张 crop

### 3.3 `text`

该 helper 的文字结果。

例如：

- OCR 文本
- “detected 2 objects”
- “returned 1 crop images”

### 3.4 `meta`

结构化信息。

例如：

- annotations
- crop_boxes
- OCR pages
- mask scores

---

## 4. 最终 step 输出到底能不能是多张图

结论：

> helper 可以返回多张图，但 executor 整个 step 的最终输出目前仍然必须是单张图。

### 4.1 helper 层可以是多图

例如：

- `_call_dino_crop(...)` 底层会返回 `images`

### 4.2 但整个 `CodeImageTool.execute()` 最终只接受一张主图

`execute()` 在代码执行完成后，会去找：

- `result`
- `output`
- `processed_image`
- `img`
- `image`

然后要求这个最终值必须是：

- 单个 `PIL.Image`

否则直接报错：

- `Code must return a PIL Image object.`

最后对外返回的 `ToolResponse` 也是：

```python
ToolResponse(
    image=[processed_image],
    ...
)
```

也就是说：

- 外层 `ToolResponse.image` 是列表
- 但当前列表长度实际上固定是 1
- 这 1 张图来自 executor 代码里最终选中的那张 `processed_image`

### 4.3 实务建议

因此推荐 executor 代码这样写：

```python
crop = _call_dino_crop("price tag", max_crops=2, padding=4)
ocr = _call_ocr_assist(image_obj=crop["images"][0])
print(ocr["text"])
result = crop["images"][0]
```

而不是把：

- `result = crop["images"]`

因为那会失败。

---

## 5. “helper 成功后，result['image'] 会自动成为当前活跃图” 到底是什么意思

这句话要拆开理解。

### 5.1 当前代码里真的会发生的事

每次 helper 成功后，运行环境会执行：

1. 记录 `__last_helper_result__`
2. 把 `result["image"]` 设为当前全局 `image`
3. 把 `img` 同步成这张图
4. 把 `draw` 也绑定到这张图

也就是说：

- 后续你如果直接读 `image`
- 或直接做 PIL 操作

看到的是最新 helper 输出图。

### 5.2 但这不等于 helper 会自动链到上一张活跃图

当前 helper 选输入图的逻辑是：

1. 如果传了 `image_obj`
   - 用 `image_obj`
2. 否则如果传了 `image_index`
   - 用该索引图
3. 否则
   - 用当前 step 默认 `image_index`

注意这里默认回退的是：

- step 的默认输入图索引

不是：

- 最新活跃图

所以这个差别非常重要：

- 活跃图会影响全局变量 `image/img/draw`
- 但不会自动改变 helper 的默认输入图选择逻辑

### 5.3 实务建议

如果要做 helper 串联，推荐永远显式传：

- `image_obj=prev["image"]`

不要假设：

- 第二个 helper 会自动吃到第一个 helper 的输出图

推荐写法：

```python
box = _call_ground_box("serial number")
crop = _call_dino_crop("serial number", image_obj=box["image"], based_on="box", padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
print(ocr["text"])
result = crop["image"]
```

如果模型已经能直接给出坐标，则更适合用本地 helper：

```python
box = _call_manual_box(120, 80, 260, 180, label="serial")
crop = _call_manual_crop(120, 80, 260, 180, image_obj=box["image"], padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
print(ocr["text"])
result = crop["image"]
```

---

## 6. `image_index` 和 `image_obj` 的区别

### 6.1 `image_index`

它表示：

- 从当前 runtime instance 的输入图片列表里，选第几张作为输入

这张图片来自：

- `visible_images`

再由 runtime wrapper 编译成：

- `CodeImageTool.create(image=[...])`
- `CodeImageTool.execute(... image_index=...)`

所以：

- `image_index` 是运行时索引
- 是外层 pipeline 和 runtime wrapper 的桥接字段

### 6.2 `image_obj`

它表示：

- 显式把一张内存中的 `PIL.Image` 对象作为 helper 输入

典型来源：

- 上一个 helper 的 `result["image"]`
- `result["images"][k]`
- 自己通过 PIL 处理过的临时图

所以：

- `image_obj` 是代码内部链式加工字段
- 更适合局部流水线串联

### 6.3 两者的关系

可以这么记：

- `image_index`：从 runtime 输入池里取图
- `image_obj`：从 Python 变量里取图

### 6.4 推荐规则

推荐模型按这个优先级使用：

1. 如果要继续处理上一步 helper 的输出
   - 用 `image_obj`
2. 如果要重新从 root 图或候选图开始
   - 用 `image_index`
3. 如果当前只有一张默认输入图且逻辑很简单
   - 可省略，两者都不传

---

## 7. 这是不是原来的 CodeVision pipeline

结论：

> 一半是原来的，一半是为了 helper 链式能力补出来的。

### 7.1 属于原始 `CodeImageTool` 主协议的部分

这些是原始主协议风格：

- `create(image=[...])`
- `execute(parameters={"code": ..., "image_index": ...})`
- 最终产出一张主结果图

也就是说：

- 多输入候选图
- 当前起手图索引
- 单步代码执行

这是原始 CodeImageTool 路线。

### 7.2 属于 helper 扩展层的部分

这些是为了视觉工具链串联能力补出来的：

- `_call_ground_box`
- `_call_manual_box`
- `_call_manual_crop`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_ocr_assist`
- `_call_blur_bg`
- `image_obj`
- active image 机制
- helper trace

所以现在更准确地说：

- 外层 runtime 协议继承了原始 CodeImageTool
- 内层 helper 编程体验是围绕离线 pipeline 逐步增强出来的

---

## 8. 当前主 helper 的推荐描述

下面这份是建议给 executor 的真实工具说明基线。

### 8.1 `_call_ground_box(...)`

用途：

- 根据文本提示定位目标区域并绘制检测框

典型场景：

- 先找“price tag”
- 先找“serial number region”
- 先找“the highlighted object”

关键参数：

- `text_prompt: str`
- `image_index: Optional[int] = None`
- `image_obj: Optional[Any] = None`
- `box_threshold: float = 0.35`
- `text_threshold: float = 0.25`

常用输出：

- `result["image"]`：带框图
- `result["meta"]["annotations"]`：框信息

### 8.2 `_call_sam_mask(...)`

用途：

- 分割并高亮目标区域

适合：

- 目标区域复杂
- 需要 mask 而不只是框

常用输出：

- `result["image"]`：mask 高亮图
- `result["meta"]["annotations"]`

### 8.3 `_call_dino_crop(...)`

用途：

- 根据文本提示先 grounding，再裁剪目标区域

这是最适合接 OCR 的 helper 之一。

关键参数：

- `text_prompt`
- `based_on="box" | "mask"`
- `detection_index`
- `max_crops`
- `padding`

常用输出：

- `result["image"]`：第一张主 crop
- `result["images"]`：全部 crop
- `result["meta"]["crop_boxes"]`

### 8.4 `_call_ocr_assist(...)`

用途：

- 对当前图或 crop 进行 OCR

适合：

- 读取标签数字
- 读取 serial / receipt / sign

常用输出：

- `result["text"]`：OCR 文本
- `result["image"]`：OCR 可视化图或输入图副本
- `result["meta"]`：OCR 明细

### 8.5 `_call_blur_bg(...)`

用途：

- 保留前景、模糊背景

适合：

- 背景太杂导致阅读困难

但它不是主路径工具，优先级通常低于：

- `ground_box`
- `dino_crop`
- `ocr_assist`

---

## 9. 给 planner 的工具描述建议

planner 不需要知道 Python 签名细节。

planner 更适合看到的是 capability 级别描述。

建议给 planner 的工具目录写成：

- `ground_box`
  - localize a target region matching a text prompt and return a boxed image plus box annotations
- `sam_mask`
  - segment and highlight the matched target region
- `dino_crop`
  - localize and crop the matched region for closer inspection
- `ocr_assist`
  - read text from the current image or crop
- `blur_bg`
  - blur the background while preserving the matched foreground region

planner 只需要决定：

- 用哪个能力
- 顺序是什么
- 为什么这样做

它不需要知道：

- helper 的每个默认参数

---

## 10. 给 executor 的工具描述建议

executor 必须看到比 planner 更具体的说明。

建议 executor prompt 至少包含：

### 10.1 helper 函数签名级说明

至少说明：

- helper 名
- 作用
- 关键参数
- 返回结构

### 10.2 输入规则

要明确告诉模型：

- `image_obj` 优先于 `image_index`
- 想串联前一步 helper 输出时，优先显式传 `image_obj=prev["image"]`

### 10.3 输出规则

要明确告诉模型：

- helper 返回 dict，不是直接返回图
- 最终必须把单张 `PIL.Image` 放进 `result`
- 不要把列表直接赋给 `result`

### 10.4 推荐代码模式

建议在 prompt 里直接放 2-3 个模式示例。

例如：

#### 模式 A：定位后裁剪

```python
box = _call_ground_box("price tag")
crop = _call_dino_crop("price tag", image_obj=box["image"], based_on="box", padding=4)
result = crop["image"]
```

#### 模式 B：裁剪后 OCR

```python
crop = _call_dino_crop("serial number", padding=4)
ocr = _call_ocr_assist(image_obj=crop["image"])
print(ocr["text"])
result = crop["image"]
```

#### 模式 C：多 crop 里选第一张

```python
crops = _call_dino_crop("label", max_crops=2, padding=4)
ocr = _call_ocr_assist(image_obj=crops["images"][0])
print(ocr["text"])
result = crops["images"][0]
```

---

## 11. 最终建议

如果目标是“让模型更稳定地产生可执行调用代码”，最关键的不是继续堆工具名，而是把以下 5 条写清楚：

1. 模型真正能调用的是 helper，不是 service client
2. helper 返回 dict，最常用字段是 `image / images / text / meta`
3. helper 可以返回多图，但最终 step 输出必须是单张 `PIL.Image`
4. helper 串联时优先显式传 `image_obj=prev["image"]`
5. 最终把最有用的主图放进 `result`

一句话总结：

> 当前最稳的提示方式应该是：planner 看到抽象 capability 描述；executor 看到 helper 签名、返回值规则、`image_obj` 串联规则，以及 2-3 段典型代码模板。
