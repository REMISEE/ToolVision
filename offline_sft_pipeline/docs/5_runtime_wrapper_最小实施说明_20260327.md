# 5 Runtime Wrapper 最小实施说明

日期：2026-03-27  
状态：handoff 实施稿  
目的：给下一窗口直接续做，明确当前 `CodeImageTool` 真实行为、还差什么、`runtime wrapper` 应该怎么实现并接入 offline pipeline。

> 状态提示（更新于 2026-03-27）：
> 本文中关于“`CodeImageTool` 还没补 `observed_helper_calls` / helper trace / stdout-stderr 捕获”的描述，已经不再代表当前代码状态。
> 这些内容现已在 tool 层落地。
> 后续新增或改动 helper 时，请同时参考：
> [offline_sft_pipeline/docs/6_CodeImageTool_helper_新增改动维护说明_20260327.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/6_CodeImageTool_helper_%E6%96%B0%E5%A2%9E%E6%94%B9%E5%8A%A8%E7%BB%B4%E6%8A%A4%E8%AF%B4%E6%98%8E_20260327.md)

---

## 1. 当前已经成立的事实

### 1.1 `CodeImageTool` 已经是 service-only

当前 `CodeImageTool` 已经不再接受旧 worker 模式：

- `external_call_mode != "service"` 会直接报错
- OCR / GroundedSAM2 都通过 service client 走 HTTP

对应位置：

- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L383)
- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L401)

### 1.2 helper 统一返回协议已经成立

当前 helper 已经统一返回：

```python
{
    "image": PIL.Image,
    "images": list[PIL.Image],
    "text": str,
    "meta": dict,
}
```

对应位置：

- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L447)
- [offline_sft_pipeline/docs/2_helper_服务化后统一返回协议_20260326.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/2_helper_%E6%9C%8D%E5%8A%A1%E5%8C%96%E5%90%8E%E7%BB%9F%E4%B8%80%E8%BF%94%E5%9B%9E%E5%8D%8F%E8%AE%AE_20260326.md#L39)

### 1.3 `execute()` 已经会把真实 `text/meta` 往外带

当前 `execute()` 不再固定回 `"Here is the processed image..."` 这类样板文案，而是：

1. 优先取最后一次 helper 返回的 `text`
2. 优先取最后一次 helper 返回的 `meta`
3. `ToolResponse.image` 仍然返回最终处理后的图

对应位置：

- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L925)
- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L936)
- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L964)

---

## 2. 一个 code block 能不能调用多个 helper

可以。

当前一段 executor code 里可以连续调用多个 helper，例如：

```python
crop = _call_dino_crop("serial number.", based_on="box", max_crops=2)
ocr = _call_ocr_assist(image_obj=crop["images"][0])
result = ocr["image"]
```

也可以直接依赖“active image”连续状态：

```python
crop = _call_dino_crop("serial number.", based_on="box", max_crops=2)
ocr = _call_ocr_assist()
result = ocr["image"]
```

原因是每个 helper 调用后都会把返回的主图写回当前活动图像：

- `safe_globals["image"]`
- `safe_globals["img"]`
- `safe_globals["draw"]`

对应位置：

- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L574)
- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L595)
- [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py#L662)

---

## 3. 多 helper 串联时，图像状态到底怎么流动

### 3.1 当前 step 的可见图片列表

`execute()` 开始时会拿到当前 `images` 列表和一个默认 `image_index`。

这个列表是“当前 step 的初始可见图片”。

### 3.2 helper 调用时的输入图

helper 支持两种输入方式：

1. `image_obj=...`
2. `image_index=...`

如果都不传，就默认用当前 step 的默认输入图，或者上一轮 helper 更新后的 active image。

### 3.3 helper 调用后的 active image

helper 返回后会：

1. 记录最后一次 helper 结果
2. 把 `result["image"]` 设为 active image

这意味着图像状态在一个 code block 里是连续的。

所以“先 crop，再对 crop 出来的图 OCR”是支持的。

### 3.4 多图返回时怎么处理

像 `_call_dino_crop(...)` 这种 helper 可能返回多张图：

- `result["images"]` 是完整列表
- `result["image"]` 默认取 `images[0]`

因此：

- 如果你想默认链式继续处理，当前行为会沿着第一张 crop 往下走
- 如果你想指定第 2 张 crop 去 OCR，应该显式传：

```python
crop = _call_dino_crop("serial number.", max_crops=3)
ocr = _call_ocr_assist(image_obj=crop["images"][1])
result = ocr["image"]
```

---

## 4. 多 helper 调用后，`CodeImageTool` 当前到底返回什么

这个问题要分两层看。

### 4.1 helper 在代码里返回什么

每个 helper 自己返回统一字典：

```python
{
    "image": ...,
    "images": ...,
    "text": ...,
    "meta": ...,
}
```

### 4.2 整个 `execute()` 对外返回什么

当前 `execute()` 最终返回：

1. `ToolResponse.image`
   - 最终 `result` 对应的一张 PIL 图
   - 也就是 executor code 最后决定返回的那张图
2. `ToolResponse.text`
   - 当前实现优先取“最后一次 helper 的 text”
3. `ToolResponse.meta`
   - 当前实现优先取“最后一次 helper 的 meta”
4. `metrics`
   - 目前已有 `helper_text` / `helper_meta`

注意：

- 现在不是把“所有 helper 的 text/meta”都完整往外返
- 现在只稳定保留“最后一次 helper 结果”

这就是当前还需要补 `observed_helper_calls` 和 helper trace 的原因。

---

## 5. `observed_helper_calls` 是什么，为什么要有

这是 schema 已经定义好的字段，不是临时发明。

对应位置：

- [offline_sft_pipeline/schemas/executor_runtime_result_schema.json](/data/home/suchenghao/ToolVision/offline_sft_pipeline/schemas/executor_runtime_result_schema.json#L19)
- [offline_sft_pipeline/docs/1_单步_runtime_wrapper_接口定义_20260326.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/1_%E5%8D%95%E6%AD%A5_runtime_wrapper_%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89_20260326.md#L166)

当前 schema 要求至少有：

```json
{
  "observed_helper_call_count": 2,
  "observed_helper_calls": [
    {"order": 1, "name": "_call_ground_box", "status": "ok"},
    {"order": 2, "name": "_call_dino_crop", "status": "ok"}
  ]
}
```

为什么要有：

1. runtime wrapper 不需要再自己猜 executor 调用了什么
2. judge / replay / exporter 可以直接看到真实调用链
3. 一步里即使只调用 0 到 1 个 helper，这个字段也能稳定表达

现阶段不用做得很重。

最小实现就够：

- 不记录复杂参数
- 不记录中间大 payload
- 只记录 `order / name / status`

---

## 6. `runtime wrapper` 到底是什么

`runtime wrapper` 不是 planner，也不是 executor model。

它只是“单步真实执行器外壳”：

1. 读取当前 step 的输入状态
2. 读取 `executor_code.py`
3. 调一次 `CodeImageTool`
4. 保存输出图和运行记录
5. 写出符合 schema 的 `runtime_result.json`

定义文档：

- [offline_sft_pipeline/docs/1_单步_runtime_wrapper_接口定义_20260326.md](/data/home/suchenghao/ToolVision/offline_sft_pipeline/docs/1_%E5%8D%95%E6%AD%A5_runtime_wrapper_%E6%8E%A5%E5%8F%A3%E5%AE%9A%E4%B9%89_20260326.md#L11)

可以把它理解成：

- `CodeImageTool` 是执行引擎
- `runtime wrapper` 是把执行引擎接入 offline pipeline 的标准外壳

---

## 7. 哪些字段该在 wrapper 里做，不该在 `CodeImageTool` 里做

### 7.1 应该由 `CodeImageTool` 负责的

1. 真正执行代码
2. helper 注入
3. helper 返回的 `image / text / meta`
4. 最小 helper trace

### 7.2 应该由 wrapper 负责的

1. `sample_id`
2. `trajectory_id`
3. `round_idx`
4. `step_idx`
5. `created_at`
6. `code_execution.code_path`
7. `stdout_path`
8. `stderr_path`
9. `error`
10. 输出图 artifact 路径

这些字段是 offline pipeline 的执行上下文，不是 tool 本身知道的上下文。

所以：

- 不是“以后后端再随便补”
- 而是“本来就应该在 wrapper 这一层组装”

---

## 8. 当前完成度

### 8.1 已完成

1. `CodeImageTool` service-only
2. helper 返回统一协议
3. 最后一次 helper 的 `text/meta` 已经可透出
4. OCR demo 参数口径已经纠正到 `paddleocr` 真实可用参数

### 8.2 未完成

1. `observed_helper_calls`
2. `observed_helper_call_count`
3. `offline_sft_pipeline/runtime/` 里的真实 wrapper 代码
4. `runtime_result.json` 落盘逻辑
5. `stdout/stderr` 记录
6. `code_execution` 时间与耗时

结论：

- 现在“执行引擎”差不多能用了
- 但“单步执行记录器”还没真正落地

---

## 9. 下一窗口的最小实施顺序

### 第一步：给 `CodeImageTool` 补 helper trace

目标：

- 在一次 `execute()` 内维护：

```python
[
  {"order": 1, "name": "_call_dino_crop", "status": "ok"},
  {"order": 2, "name": "_call_ocr_assist", "status": "ok"}
]
```

最小做法：

1. 在 `_create_safe_globals()` 里新增 `__helper_trace__`
2. 每个 helper 调用前后记录：
   - `order`
   - `name`
   - `status`
3. `_execute_code()` 返回：
   - `helper_result`
   - `helper_trace`
4. `execute()` 把它放进 metrics

建议 metrics 结构至少新增：

```python
{
    "observed_helper_call_count": 2,
    "observed_helper_calls": [
        {"order": 1, "name": "_call_dino_crop", "status": "ok"},
        {"order": 2, "name": "_call_ocr_assist", "status": "ok"},
    ],
}
```

### 第二步：实现最小 `runtime wrapper`

建议新增文件：

- `offline_sft_pipeline/runtime/code_image_runtime_wrapper.py`

最小职责：

1. 接收 request 对象
2. 读取 `executor_code.py`
3. 加载 `visible_images`
4. 调 `CodeImageTool.create -> execute -> release`
5. 保存 `output_0.png`, `output_1.png`, ...
6. 保存 `runtime_result.json`

### 第三步：实现 wrapper 输入 request dataclass

建议新增：

- `RuntimeStepRequest`
- `RuntimeStepOutput`

最小字段直接按文档：

- `sample_id`
- `trajectory_id`
- `round_idx`
- `step_idx`
- `executor_code_path`
- `executor_cot_path`
- `visible_images`
- `image_index`
- `step_output_dir`

### 第四步：实现 schema-compatible result builder

wrapper 内部组装：

```python
runtime_result = {
    "schema_version": "0.1.0",
    "sample_id": ...,
    "trajectory_id": ...,
    "round_idx": ...,
    "step_idx": ...,
    "created_at": ...,
    "success": ...,
    "images": ...,
    "text": ...,
    "meta": ...,
    "observed_helper_call_count": ...,
    "observed_helper_calls": ...,
    "code_execution": ...,
    "error": ...,
}
```

### 第五步：做一个最小 smoke test

只测单样本单步：

1. 输入一张 root image
2. 写一段手工 executor code
3. 运行 wrapper
4. 检查：
   - `runtime_result.json`
   - `output_0.png`
   - schema 校验能过

---

## 10. 实现完后怎么嵌入整个 pipeline

实现完 wrapper 后，它可以直接成为 pipeline 的正式执行组件。

上层调用关系应为：

1. planner / executor 产出当前 step 的 code
2. orchestrator 构造 `RuntimeStepRequest`
3. wrapper 执行
4. wrapper 返回 `runtime_result`
5. store / orchestrator 再更新：
   - `trajectory.json`
   - `messages.json`
   - frontier
   - judge inputs

也就是说：

- wrapper 不是临时 mock
- wrapper 做完就可以直接挂进真实 pipeline

---

## 11. 下一窗口建议直接开做的文件

1. [CodeVision/verl/tools/code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py)
   - 补 helper trace
2. `offline_sft_pipeline/runtime/code_image_runtime_wrapper.py`
   - 新建最小 wrapper
3. `offline_sft_pipeline/runtime/types.py`
   - 新建 request / output dataclass
4. `offline_sft_pipeline/scripts/run_single_runtime_smoke.py`
   - 新建单步 smoke test

不要先做：

1. trajectory store
2. branching
3. exporter
4. depth / count

先把“单步真实执行 + 标准落盘”这条闭环跑通。
