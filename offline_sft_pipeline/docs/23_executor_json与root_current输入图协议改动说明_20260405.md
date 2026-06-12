# 23 Executor JSON 与 `root/current` 输入图协议改动说明

日期：2026-04-05  
状态：已改代码并完成回归验证  
目的：完整记录本次围绕 executor 原始协议、planner 到 runtime 的输入图选择语义、以及后续 CodeVision 风格导出兼容所做的改动，方便后续协作、接真实 backend、接 exporter 时直接对照。

---

## 1. 一句话结论

本次改动把 executor 侧协议和输入图语义收敛成下面这套：

- planner 每个 step 必须显式指定：
  - `input_image = "root" | "current"`
- executor 模型原始输出不再使用 XML `<think> + <code>`
- executor 现在统一输出 JSON：
  - `think + tool_call(name="code_image_tool", arguments={code, description})`
- executor 不再负责选择起始图索引
- runtime 的实际 `image_index` 由 orchestrator 根据 planner 的 `input_image` 和当前 `visible_images` 自动编译
- pipeline 内部继续用 `artifact_id` 做稳定图像标识
- `StepRecord` 会额外记录：
  - `input_image`
  - `input_artifact_id`
  - `executor_description`

这样做之后：

1. 避免了 executor XML 协议与 Qwen 模板冲突  
2. 把“选原图还是当前图”的决策明确放回 planner  
3. 给后续 exporter 重写 CodeVision 全局 `image_index` 留出了稳定映射点

---

## 2. 这次改动要解决的核心问题

本次改动前，executor 侧有两个主要问题。

### 2.1 executor 原始协议仍是 XML

此前 executor client 采用：

```text
<think>...</think>
<code>...</code>
```

这和 planner 早期协议一样，会遇到两个实际问题：

- 与 Qwen 模板/推理风格更容易冲突
- 和后续目标训练格式不一致

planner 之前已经从 XML 改成 JSON，因此 executor 继续保留 XML 会导致两侧协议风格不统一。

### 2.2 `image_index` 的语义混杂

我们讨论时已经明确：

- pipeline 内部长期标识不能靠 `image_index`
- 长期标识应该继续用 `artifact_id`
- runtime 真正执行时才临时编译成当前 step 的局部 `image_index`

但此前 executor 侧还没有把“谁决定用原图，谁决定用当前图”讲清楚。  
如果让 executor 再选一次 `image_index`，会导致：

- planner 和 executor 在“选图”上职责重复
- `visible_images` 的局部索引和 CodeVision 导出时的全局索引混在一起

所以本次改成：

- planner 决定：
  - `root` 还是 `current`
- executor 只负责写代码和描述
- orchestrator 编译 runtime 用的局部 `image_index`

---

## 3. 本次协议设计最终冻结版本

### 3.1 planner step 新字段

`PlannerStepSpec` 新增：

```json
"input_image": "root" | "current"
```

语义：

- `root`
  - 下一执行步默认从原图开始
- `current`
  - 下一执行步默认从上一执行步的最新工作图开始

这里没有保留 `auto`。

原因：

- executor 不需要再做起始图选择推理
- 让 planner 直接在策略层决定更清晰
- 避免后续行为不稳定

### 3.2 executor 原始输出协议

executor 模型现在要求输出一个 JSON object：

```json
{
  "think": "......",
  "tool_call": {
    "name": "code_image_tool",
    "arguments": {
      "code": "......",
      "description": "......"
    }
  }
}
```

约束：

- 必须是严格 JSON
- 不允许 XML
- 不允许 markdown fence
- `tool_call.name` 必须是 `code_image_tool`
- `arguments.code` 非空
- `arguments.description` 非空

注意：

- executor 原始输出里不包含 `image_index`
- 起始输入图由 planner 决定，orchestrator 编译

### 3.3 runtime `image_index` 语义

runtime 里的 `image_index` 继续保持当前定义：

- 只对当前 step 的 `visible_images` 生效
- 是局部索引
- 不是长期标识

因此：

- pipeline 内部稳定标识：`artifact_id`
- runtime 临时执行索引：局部 `image_index`
- 后续训练导出索引：CodeVision 全局 `image_index`

这三层语义现在是显式分开的。

---

## 4. 代码层具体改动

下面按模块说明。

### 4.1 `core/models.py`

改动点：

#### A. `PlannerStepSpec`

新增字段：

- `input_image: Literal["root", "current"]`

当前 step schema 从：

```python
step_id
step_goal
capability_plan
executor_instruction
```

变成：

```python
step_id
step_goal
input_image
capability_plan
executor_instruction
```

#### B. `ExecutorStepOutput`

新增字段：

- `description: str`

因此内部标准输出从：

```python
cot
code
raw_response_text
metadata
```

变成：

```python
cot
code
description
raw_response_text
metadata
```

并新增校验：

- `code` 非空
- `description` 非空

#### C. `StepRecord`

新增字段：

- `input_image`
- `input_artifact_id`
- `executor_description`

最终 `StepRecord` 现在会记录：

- planner 这一步要求从 `root/current` 哪类图开始
- 真实被编译并执行的那张图的 `artifact_id`
- executor 生成的自然语言 step 描述

这样后续 exporter 就不需要从 prompt 或局部索引去猜“这一步到底是拿哪张图跑的”。

---

### 4.2 `schemas/planner_output_schema.json`

对 step spec 做了 schema 扩展。

新增 required 字段：

- `input_image`

新增约束：

```json
"input_image": {
  "type": "string",
  "enum": ["root", "current"]
}
```

这意味着：

- 任何 planner suggestions 路径，现在每个 step 都必须把输入图语义写清楚

---

### 4.3 `schemas/executor_step_output_schema.json`

新增 required 字段：

- `description`

新增 schema：

```json
"description": {
  "type": "string",
  "minLength": 1
}
```

这样 executor 内部标准输出已经能容纳：

- reasoning
- code
- 自然语言 step 描述

---

### 4.4 `schemas/trajectory_schema.json`

对 `stepRecord` 的 schema 做了扩展。

新增 required 字段：

- `input_image`
- `input_artifact_id`
- `executor_description`

新增约束：

- `input_image` 只能是 `root/current`
- `input_artifact_id` 非空字符串
- `executor_description` 非空字符串

---

### 4.5 `pipelines/parsing.py`

新增 executor JSON 识别工具：

- `looks_like_executor_json_payload(...)`
- `try_parse_executor_json_payload(...)`

作用：

- 让 executor client 不再走旧 XML 解析路径
- 统一先尝试把模型原始输出整体当 JSON object 解析

当前 executor 解析逻辑已经不再依赖：

- `ensure_tag_order(<think>, <code>)`
- `extract_required_tag("think")`
- `extract_required_tag("code")`

---

### 4.6 `pipelines/executor_client.py`

这是本次 executor 协议切换的核心改动。

#### A. user prompt 注入了新的信息

现在 executor prompt 会多注入两个字段：

- `input_image`
- `selectable_input_images_json`

也就是说，executor 现在会明确看到：

1. planner 选的是 `root` 还是 `current`
2. 当前 step 的可选输入图槽位表

#### B. executor parser 改成 JSON contract

原来：

- 解析 `<think> + <code>`

现在：

1. 整体解析 JSON
2. 校验 `think`
3. 校验 `tool_call`
4. 校验 `tool_call.name == "code_image_tool"`
5. 校验 `tool_call.arguments.code`
6. 校验 `tool_call.arguments.description`
7. 转成内部 `ExecutorStepOutput`

#### C. executor 不再解析 `image_index`

这是本次职责收缩的关键。

executor 现在只产：

- `cot`
- `code`
- `description`

不产：

- `image_index`

因为起始图索引由 orchestrator 统一编译。

#### D. 新增 `_build_selectable_input_images(...)`

这个方法用来生成给 prompt 的局部图槽位说明：

```json
[
  {"index": 0, "role": "root", "label": "original image"},
  {"index": 1, "role": "current", "label": "latest previous-step image"}
]
```

这里的目的不是让 executor 选索引，而是：

- 让模型明确知道当前图池结构
- 理解默认 `image/img` 来自 planner 选定的那一类图

---

### 4.7 `pipelines/backends.py`

改了两件事。

#### A. 默认 fake planner 文本

step 示例里新增：

- `"input_image": "root"`

#### B. 默认 fake executor 文本

从旧 XML：

```text
<think>...</think>
<code>...</code>
```

改成 JSON：

```json
{
  "think": "...",
  "tool_call": {
    "name": "code_image_tool",
    "arguments": {
      "code": "...",
      "description": "..."
    }
  }
}
```

这样 fake backend 路径和未来真实 executor backend 路径就统一了。

---

### 4.8 `pipelines/scripted_components.py`

这个文件主要同步 scripted/demo/test 场景。

改动包括：

#### A. `make_step(...)`

新增参数：

- `input_image`

默认值暂设为 `root`，并显式写进 `PlannerStepSpec`

#### B. `make_executor_output(...)`

新增：

- `description`

并把 scripted executor raw text 改成 JSON tool_call。

#### C. `render_executor_output_as_model_text(...)`

从 XML 渲染改成 JSON 渲染。

#### D. 三轮 demo 场景补齐 `input_image`

目前 demo 里已经显式区分了：

- 重新从原图开始的 step -> `root`
- 继续沿用上一工作图的 step -> `current`

例如：

- `s21` 的 OCR continue 路径 -> `current`
- `s22` 的 reground 路径 -> `root`

这正好覆盖了本次新协议的典型两种分支。

---

### 4.9 `pipelines/orchestrator_v01.py`

这是本次输入图选择逻辑真正落地的核心。

#### A. 删除旧的“自动选 runtime image index”语义

此前有一个启发式：

- 如果有 latest primary image，则优先用它
- 否则回到 root 图

这由 `_select_runtime_image_index(...)` 实现。

现在不再使用这个函数决定语义起始图。

#### B. 新增 `_resolve_runtime_input(...)`

新规则：

- `input_image == "root"`
  - 返回 `visible_images[0]`
  - runtime `image_index = 0`
- `input_image == "current"`
  - 查当前 trajectory 的 latest primary image
  - 在 `visible_images` 中定位其局部槽位
  - 返回对应 artifact 和局部 `image_index`

如果 planner 选了 `current`，但当前轨迹还没有上一步图，则直接报错。

也就是说：

- 现在 runtime 起始图选择完全由 planner 明示
- 不再由 orchestrator 悄悄启发式覆盖 planner 语义

#### C. assistant message 写回格式改动

此前 assistant step message content 是：

```text
<think>...</think>
<tool_call name="code_image_tool">
...python code...
</tool_call>
```

现在变成：

```text
<think>...</think>
<tool_call>
{
  "name": "code_image_tool",
  "arguments": {
    "code": "...",
    "description": "...",
    "image_index": 1
  }
}
</tool_call>
```

注意这里的 `image_index`：

- 不是模型生成的
- 是 orchestrator 根据 planner 的 `input_image` 编译进去的运行时局部索引

#### D. assistant message metadata 新增

现在每一步 assistant message metadata 会记录：

- `input_image`
- `input_artifact_id`
- `executor_description`
- `runtime_image_index`

#### E. `StepRecord` 落盘新增

写 step record 时现在会带上：

- `input_image`
- `input_artifact_id`
- `executor_description`

这三者是后续 exporter 最关键的输入。

---

### 4.10 prompts 改动

#### A. `prompts/planner_system_v03.txt`

新增 step schema 示例：

- `input_image`

新增规则说明：

- `root` = 原图
- `current` = 上一步最新图
- 没有 previous-step image 时不能选 `current`

#### B. `prompts/planner_user_v01.txt`

新增用户态说明：

- `root/current` 的语义解释

#### C. `prompts/executor_system_v01.txt`

从 XML contract 彻底切成 JSON contract。

现在 system prompt 明确要求：

- 返回一个 JSON object
- `tool_call.name` 必须是 `code_image_tool`
- `arguments.code` 和 `arguments.description` 都必须有

#### D. `prompts/executor_user_v01.txt`

新增说明：

- planner-selected default input image
- selectable input images
- 默认 `image/img` 绑定的是 planner 选定的起始图
- 同一步内部串联 helper 时优先 `image_obj=prev["image"]`
- 最终要给 `result`

这一步是为了让 executor 明确：

- 它不是在选第一张图
- 它是在已经选定的默认图基础上写执行代码

---

## 5. 现在的职责边界

本次改动后，planner / executor / orchestrator / runtime 的分工是：

### planner

负责：

- 选策略分支
- 决定每个 step 从 `root` 还是 `current` 开始

不负责：

- 输出运行时 `image_index`
- 写工具调用 JSON

### executor

负责：

- 写 step-level reasoning
- 写 Python code
- 写自然语言 `description`

不负责：

- 决定起始输入图来自 root 还是 current
- 决定最终运行时 `image_index`

### orchestrator

负责：

- 根据 planner 的 `input_image` 编译 runtime 局部索引
- 记录真实起始输入图的 `artifact_id`
- 把 executor 输出和 runtime 参数拼成可回放消息

### runtime

继续只负责：

- 拿到 `code + image_index + visible_images`
- 执行代码
- 返回 runtime result

它不关心：

- planner 选的是 `root` 还是 `current`
- `artifact_id`
- CodeVision 最终导出索引

---

## 6. `root/current` 到 runtime `image_index` 的统一规则

这是本次最关键的逻辑。

### 6.1 当前 `visible_images` 规则没变

当前 orchestrator 仍然按下面的顺序组织 `visible_images`：

1. 所有 root artifacts
2. 最新一步主输出图（如果存在）

因此最常见情况下：

- 只有 root 图时：
  - `visible_images = [root]`
- 已有上一步图时：
  - `visible_images = [root, current]`

### 6.2 planner 选 `root`

则：

- `input_artifact_id = root artifact`
- `runtime image_index = 0`

### 6.3 planner 选 `current`

则：

- 找到 latest primary image 的 `artifact_id`
- 在当前 `visible_images` 中定位它
- 用那个局部槽位做 runtime `image_index`

常见情况下：

- `visible_images = [root, current]`
- 则 `current` 对应局部索引 `1`

### 6.4 为什么这不等于 CodeVision 的最终索引

因为：

- runtime `image_index` 是当前 step 的局部索引
- CodeVision 的 `image_index` 是最终导出样本 `images[]` 的全局索引

这两个本来就不是一个层级的东西。

因此后续 exporter 正确做法不是“直接拿运行期局部索引”，而是：

1. 先根据 `input_artifact_id` 找到真实输入图
2. 再在导出样本的全局 images 列表里查它的位置
3. 最后把 tool_call 里的 `image_index` 重写成全局索引

比如：

- 运行时这一步 `current -> local index = 1`
- 但这个 `current` 图在最终导出样本 `images[]` 中是第 4 张

那么导出时应重写为：

- `image_index = 4`

也正因为如此，本次一定要把：

- `input_artifact_id`

落进 `StepRecord` 和 assistant metadata。

---

## 7. `description` 的处理原则

这次最终定的是：

- executor 直接生成 `description`
- runtime 不依赖它
- orchestrator 保存它
- exporter 后续复用它

为什么不传给 runtime：

- runtime 当前执行只真正需要：
  - `code`
  - `image_index`
  - `visible_images`
- `CodeImageTool` 虽然接受 `description`
  - 但当前 pipeline 不依赖它做决策

所以当前做法是：

- `description` 留在 executor 输出、assistant message、step record 中
- runtime 继续使用它自己的内部占位描述

这能让我们先把协议层理顺，而不额外扩大 runtime 接口面。

---

## 8. 测试与验证

本次实际执行了两层验证。

### 8.1 单元测试

执行命令：

```bash
python -m unittest offline_sft_pipeline.tests.test_orchestrator_v01 offline_sft_pipeline.tests.test_pipelines
```

结果：

```text
Ran 22 tests in 12.052s
OK (skipped=2)
```

本次新增/更新覆盖了：

- planner JSON suggestions 中 `input_image` 的解析
- executor JSON tool_call 协议解析
- orchestrator 在 `root/current` 下编译不同 runtime `image_index`
- `StepRecord` 是否正确记录：
  - `input_image`
  - `input_artifact_id`
  - `executor_description`

### 8.2 脚本级 smoke

执行命令：

```bash
python offline_sft_pipeline/scripts/run_single_sample_pipeline.py \
  --mode client_fake_backend \
  --runtime-mode scripted \
  --run-id verify_json_executor_root_current
```

结果：

- 成功跑完整个 scripted branching demo
- 说明：
  - planner 新字段
  - executor 新 JSON parser
  - orchestrator 输入图编译
  - message / step record 落盘

这些链路都能一起工作。

同时手动检查了产出：

- `messages.json`
- `trajectory.json`

确认 assistant message 已经写成 JSON tool_call 风格，并且：

- `runtime_image_index`
- `input_image`
- `input_artifact_id`

都已正确落盘。

---

## 9. 当前还没做的部分

本次没有继续做以下内容。

### 9.1 exporter 还没重写成 CodeVision 全局 `image_index`

当前已经具备所需输入：

- `input_artifact_id`
- assistant message 里的 JSON tool_call
- trajectory 全部历史输出图

下一步 exporter 只需要：

1. 建立 `artifact_id -> export_global_index`
2. 重写 tool_call.arguments.image_index

即可对齐 CodeVision 的全局图索引格式。

### 9.2 真实 executor backend 还没接

当前变更只完成了：

- executor 协议
- parser
- prompt
- orchestrator 编译规则

但 `ApiTextBackend` 仍未实现 executor stage。

也就是说：

- fake/scripted executor 已经走新协议
- 真实 HTTP executor backend 还需要下一步接线

### 9.3 runtime 还没显式消费 executor `description`

当前这是刻意不做。

因为：

- 运行逻辑不依赖它
- 先不扩大 runtime 接口面更稳

如果以后需要对齐某些训练回放或调试面板，再考虑让 runtime result 也持久化这份 description。

---

## 10. 对后续工作的直接建议

现在比较合理的下一步顺序是：

1. 接 executor 的真实 HTTP backend
   - 复用 planner backend 的 API 调用模式
   - 但输入整理按 executor 新 JSON contract 走
2. 实现 exporter 的 `artifact_id -> global image_index` 重写
3. 再跑一轮更贴近 CodeVision 数据格式的导出 smoke

也就是说：

本次改动已经把最容易混乱的协议边界切清楚了：

- planner 决定 `root/current`
- executor 负责 `think + code + description`
- orchestrator 编译 runtime 局部索引
- exporter 再重写成 CodeVision 全局索引

这是后续最稳的基线。

---

## 11. 本次实际改动文件清单

本次实际改动的文件有：

- `offline_sft_pipeline/core/models.py`
- `offline_sft_pipeline/schemas/planner_output_schema.json`
- `offline_sft_pipeline/schemas/executor_step_output_schema.json`
- `offline_sft_pipeline/schemas/trajectory_schema.json`
- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/executor_client.py`
- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/scripted_components.py`
- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/prompts/planner_system_v03.txt`
- `offline_sft_pipeline/prompts/planner_user_v01.txt`
- `offline_sft_pipeline/prompts/executor_system_v01.txt`
- `offline_sft_pipeline/prompts/executor_user_v01.txt`
- `offline_sft_pipeline/tests/test_orchestrator_v01.py`
- `offline_sft_pipeline/tests/test_pipelines.py`

---

## 12. 最终总结

本次不是单纯“把 executor 从 XML 改成 JSON”。

真正落地的，是一整套更清晰的职责划分：

- 用 planner 明示 `root/current`
- 用 executor 只生成代码和描述
- 用 orchestrator 统一编译运行时索引
- 用 `artifact_id` 保持内部长期稳定标识

这样后面无论接：

- 真实 executor backend
- exporter 的 CodeVision 全局索引
- 甚至别的训练格式转换

都不会再被“这个 `image_index` 到底是哪一层的索引”这个问题反复绊住。
