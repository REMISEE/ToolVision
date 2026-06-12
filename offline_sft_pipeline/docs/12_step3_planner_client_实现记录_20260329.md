# 12 Step 3：Planner Client 实现记录

日期：2026-03-29  
状态：已实现  
目的：记录当前 `PlannerClient` 的代码落地情况，明确它读什么、产什么、如何解析模型返回，以及它和后续 `executor / orchestrator` 的边界。

---

## 1. 一句话结论

当前 planner 这一条链已经落成到可直接使用 fake backend 跑通的程度：

1. 有 request 对象
2. 有 backend 抽象
3. 有 prompt 模板
4. 有解析器
5. 有 `PlannerClient`
6. 能返回 schema 合法的 `PlannerOutput`

也就是说：

> planner 现在已经不是设计稿，而是一个可直接接入 orchestrator 的真 client 骨架。

---

## 2. 当前新增文件

本轮新增的 planner 相关文件有：

- `offline_sft_pipeline/pipelines/__init__.py`
- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/request_models.py`
- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/prompts/planner_system_v01.txt`
- `offline_sft_pipeline/prompts/planner_user_v01.txt`

这些文件的角色分别是：

- `backends.py`
  - 文本生成 backend 协议
  - fake backend
  - API backend TODO 占位
- `parsing.py`
  - tag 提取
  - JSON 解析
  - 统一 parse error
- `request_models.py`
  - planner request 对象
  - tool capability 对象
- `planner_client.py`
  - prompt 组装
  - backend 调用
  - planner 文本解析
  - `PlannerOutput` 结构化校验
- `prompts/*.txt`
  - planner 的 system / user 模板

---

## 3. 当前 planner 的输入是什么

当前 planner 统一读取 `PlannerClientRequest`。

字段如下：

- `sample_id`
- `trajectory_id`
- `round_idx`
- `question`
- `messages`
- `visible_images`
- `budget`
- `tool_capabilities`
- `latest_runtime_result`
- `metadata`

其中最关键的是：

1. `question`
   - 原始问题
2. `messages`
   - 当前 trajectory 的滚动消息历史
3. `visible_images`
   - 当前真正给 planner 看的可见图片集合
4. `tool_capabilities`
   - planner 可用能力目录
5. `budget`
   - 控制 planner 是否继续扩展

当前校验规则是：

- `question` 不能为空
- `messages` 不能为空
- `visible_images` 不能为空

也就是说，planner 当前明确要求：

> 不是只给问题和图，也不是只给历史对话，而是问题、历史消息、当前可见图一起给。

---

## 4. planner 现在怎么构造 prompt

planner prompt 分成两部分：

1. `planner_system_v01.txt`
2. `planner_user_v01.txt`

### 4.1 system prompt 负责什么

system prompt 当前主要负责冻结输出协议：

- 必须先输出 `<think>`
- 然后只能输出：
  - `<answer>`
  - 或 `<suggestions>`
- 两者不能同时出现
- `<suggestions>` 内部必须是 JSON array

### 4.2 user prompt 负责什么

user prompt 当前会注入：

- sample / trajectory / round 信息
- 问题
- 当前可见图
- 当前消息历史
- 剩余 budget
- 最新 runtime 结果
- 可用能力列表

也就是说，当前 planner prompt 的意图很明确：

> 让模型结合问题、可见图、已执行历史和能力目录，判断能否直接回答；不能就继续规划。

---

## 5. planner 的模型返回协议

当前 planner 已经固定为轻量 tag 协议：

### 5.1 直接回答时

```text
<think>
...
</think>
<answer>
...
</answer>
```

### 5.2 需要继续规划时

```text
<think>
...
</think>
<suggestions>
[
  ...
]
</suggestions>
```

其中：

- `think` 必须存在
- `answer` 和 `suggestions` 必须二选一
- `suggestions` 必须能被 `json.loads`

当前没有要求模型直接返回完整 `PlannerOutput` JSON。

这是故意的。

原因是：

- `sample_id`
- `trajectory_id`
- `round_idx`
- `created_at`

这些字段本来就属于系统上下文，不应该让模型来生成。

---

## 6. planner 现在怎么解析模型返回

当前解析逻辑在：

- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/planner_client.py`

### 6.1 通用 helper

当前已经有：

- `extract_tag_block(...)`
- `extract_required_tag(...)`
- `ensure_tag_order(...)`
- `parse_json_text(...)`
- `ModelResponseParseError`

### 6.2 planner 的具体规则

当前 `PlannerClient._parse_model_text(...)` 的规则是：

1. `<think>` 必须存在
2. `<think>` 必须出现在 `<answer>` 或 `<suggestions>` 之前
3. `<answer>` 和 `<suggestions>` 不能同时出现
4. `<answer>` / `<suggestions>` 至少要有一个
5. 如果是 `<answer>`
   - 生成：
     - `can_answer_now = true`
     - `direct_answer = answer`
     - `suggestions = []`
6. 如果是 `<suggestions>`
   - 必须是 JSON array
   - 每个 suggestion 用 `PlannerSuggestion.model_validate(...)` 校验
   - 生成：
     - `can_answer_now = false`
     - `direct_answer = null`
     - `suggestions = [...]`

最终会把 `<think>` 写入：

- `PlannerOutput.global_chain_cot`

这意味着：

> 当前 planner 的 `global_chain_cot` 已经直接由 `<think>` 承接。

---

## 7. planner 最终产什么

`PlannerClient.run(...)` 当前最终返回：

- `offline_sft_pipeline.core.models.PlannerOutput`

返回前会补齐：

- `sample_id`
- `trajectory_id`
- `round_idx`
- `created_at`

然后执行：

- `PlannerOutput.validate_against_schema()`

因此当前 planner 的返回不是裸字典，而是：

> 已经能直接给 `store.register_planner_round(...)` 用的正式对象。

---

## 8. fake backend 当前是什么形态

当前 fake backend 在：

- `offline_sft_pipeline/pipelines/backends.py`

已经实现：

- `BackendResponse`
- `TextGenerationBackend`
- `FakeTextBackend`
- `ApiTextBackend(TODO)`

其中当前 planner 默认 fake 返回是：

- 一段带 `<think> + <suggestions>` 的原始文本

不是：

- 直接返回 `PlannerOutput`

这样做的目的很明确：

- 不绕过 parser
- 不绕过校验
- 让 fake 路径和未来真实模型路径尽量一致

也就是说，当前能被验证的不是“字典拼装”，而是整条链：

1. prompt
2. backend
3. parse
4. validate
5. object build

---

## 9. 当前 planner 和 executor 的边界

planner 当前只负责：

- 判断能否直接回答
- 如果不能回答，产出 suggestions

planner 当前不负责：

- 选择哪个 suggestion 进入执行
- fork child trajectory
- 生成 step messages
- 生成 executor code
- 写 `executor_cot.md`
- 写 `executor_code.py`

这些都属于：

- orchestrator
- executor client

也就是说，当前 planner 的边界是：

> 产 proposal，不产执行文件。

---

## 10. 当前 planner 和 messages / visible_images 的关系

### 10.1 messages

当前 planner request 直接接收 `messages`。

这里的 `messages` 指的是：

- 当前 trajectory 的滚动消息账本
- 包含原始 user 问题
- 包含已执行 step 的 assistant / tool 历史
- 不包含 planner proposal 自身

### 10.2 visible_images

当前 planner request 直接接收 `visible_images`。

它的意义是：

- 当前真正给 planner 看的图像上下文

注意：

- 当前 `visible_images` 的传播策略还没在代码里实现
- 但 runtime 层已经有这个输入概念
- 后续 orchestrator 应决定每轮给 planner 带哪些图

当前最合理的默认策略仍然是：

1. root images
2. 最新一步主图

---

## 11. 当前已完成验证

本轮已经完成的验证包括：

1. `offline_sft_pipeline/pipelines/*.py` 可 `py_compile`
2. `FakeTextBackend -> PlannerClient -> PlannerOutput` 最小链已跑通
3. 返回对象能通过 `planner_output_schema.json` 校验

也就是说：

> planner 当前已经不是“文件存在”，而是真的跑出了合法结构化输出。

---

## 12. 当前还没做的部分

planner 这一侧当前还没有：

1. 真实 `ApiTextBackend`
2. prompt 版本管理
3. planner 输出的 richer stop reason
4. suggestion-level ranking 字段
5. frontier selection
6. child trajectory fork
7. planner output 写盘与 orchestrator 联调

所以目前的状态是：

- client 骨架已可用
- 但还没进入完整主循环

---

## 13. 下一步最合理的衔接顺序

planner 做完后，下一步最顺的是：

1. `executor_client.py`
2. step message builder
3. visible image selector
4. `judge_client.py`
5. `orchestrator_v01.py`

其中最关键的不是 prompt，而是把以下对象接顺：

- `PlannerOutput`
- `PendingExecution`
- `ExecutorStepOutput`
- `RuntimeStepRequest`
- `StepRecord`
- `JudgeRecord`

---

## 14. 一句话版本

当前 planner 已经落成：

> 它能读取问题、消息历史、当前可见图和能力目录，按 `<think> + <answer|suggestions>` 协议解析模型文本，并返回一个 schema 合法的 `PlannerOutput`；接下来真正要补的是 executor、judge 和 orchestrator 的接线。
