# 13 Step 3：Executor Client 实现记录

日期：2026-03-29  
状态：已实现  
目的：记录当前 `ExecutorClient` 的代码落地情况，明确它读什么、产什么、如何解析模型返回，以及它与 runtime / step messages / store 的边界。

---

## 1. 一句话结论

当前 executor 这一条链已经落成到可直接使用 fake backend 跑通的程度：

1. 有 request 对象
2. 有 executor prompt 模板
3. 有 `<think> + <code>` 解析
4. 有 `ExecutorClient`
5. 能返回 schema 合法的 `ExecutorStepOutput`

也就是说：

> executor 现在已经具备把“当前 step 的上下文”翻译成 `cot + code` 的真实 client 骨架。

---

## 2. 当前 executor 相关文件

当前 executor 依赖的主要文件有：

- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/request_models.py`
- `offline_sft_pipeline/pipelines/executor_client.py`
- `offline_sft_pipeline/prompts/executor_system_v01.txt`
- `offline_sft_pipeline/prompts/executor_user_v01.txt`

其中：

- `backends.py`
  - 新增了 executor 默认 fake 文本返回
- `request_models.py`
  - 新增 `ExecutorClientRequest`
- `executor_client.py`
  - 负责 prompt 组装、backend 调用、标签解析、`ExecutorStepOutput` 校验
- `prompts/*.txt`
  - 定义 executor 的 system / user prompt

---

## 3. 当前 executor 的输入是什么

当前 executor 统一读取 `ExecutorClientRequest`。

字段如下：

- `sample_id`
- `trajectory_id`
- `round_idx`
- `step_idx`
- `question`
- `messages`
- `visible_images`
- `suggestion_id`
- `suggestion_step_index`
- `step_spec`
- `planner_global_chain_cot`
- `suggestion_cot`
- `tool_capabilities`
- `metadata`

其中最关键的是：

1. `messages`
   - 当前 trajectory 的滚动消息历史
2. `visible_images`
   - 当前真正给 executor 看的图片集合
3. `step_spec`
   - 当前要执行的 `PlannerStepSpec`
4. `planner_global_chain_cot`
   - planner 这一轮的全局思路
5. `suggestion_cot`
   - 当前 suggestion 的局部 rationale
6. `tool_capabilities`
   - 当前可用能力目录

也就是说：

> executor 当前读的不是整个 planner proposal，而是“已经被 orchestrator 选中”的一个 step。

---

## 4. executor 现在怎么构造 prompt

executor prompt 分成两部分：

1. `executor_system_v01.txt`
2. `executor_user_v01.txt`

### 4.1 system prompt 负责什么

system prompt 当前主要负责冻结输出协议：

- 必须先输出 `<think>`
- 再输出 `<code>`
- `<code>` 中不要放 markdown code fence
- 不要输出额外 tag

### 4.2 user prompt 负责什么

user prompt 当前会注入：

- sample / trajectory / round / step
- 当前 suggestion 信息
- 原始问题
- planner 的全局思路
- 当前 suggestion 的 rationale
- 当前 step spec
- 当前可见图
- 当前消息历史
- 可用能力列表

这意味着：

> executor prompt 当前是围绕“当前一个 step 的代码生成”来构造的，不是让模型重新做 planning。

---

## 5. executor 的模型返回协议

当前 executor 已固定为：

```text
<think>
...
</think>
<code>
...
</code>
```

其中：

- `<think>` 必须存在
- `<code>` 必须存在
- `<think>` 必须在 `<code>` 之前

当前没有要求模型返回 JSON。

这是刻意这样设计的。

原因是：

- executor 的自然输出本来就是“局部思路 + 代码”
- 这里最脆弱的不是 JSON schema，而是代码块本身是否稳定

---

## 6. executor 现在怎么解析模型返回

当前解析逻辑在：

- `offline_sft_pipeline/pipelines/parsing.py`
- `offline_sft_pipeline/pipelines/executor_client.py`

具体规则是：

1. 校验 `<think>` 在 `<code>` 之前
2. 提取 `<think>`
3. 提取 `<code>`
4. 生成 `ExecutorStepOutput`
5. 补齐：
   - `raw_response_text`
   - `metadata`
6. 执行 `validate_against_schema()`

最终映射关系是：

- `<think>` -> `ExecutorStepOutput.cot`
- `<code>` -> `ExecutorStepOutput.code`
- 原始模型文本 -> `ExecutorStepOutput.raw_response_text`
- backend metadata -> `ExecutorStepOutput.metadata`

---

## 7. executor 最终产什么

`ExecutorClient.run(...)` 当前最终返回：

- `offline_sft_pipeline.core.models.ExecutorStepOutput`

这意味着 executor 当前的边界是：

- 负责生成 `cot + code`
- 不负责直接写 `executor_cot.md`
- 不负责直接写 `executor_code.py`
- 不负责直接跑 runtime

后续正确衔接方式应当是：

1. orchestrator 调 `ExecutorClient.run(...)`
2. 得到 `ExecutorStepOutput`
3. 调 `store.write_executor_step_files(...)`
4. 再构造 `RuntimeStepRequest`
5. 再交给 runtime wrapper

---

## 8. fake backend 当前是什么形态

当前 fake backend 在：

- `offline_sft_pipeline/pipelines/backends.py`

已经补了 executor 的默认 fake 返回：

- 一段带 `<think> + <code>` 的文本

不是：

- 直接返回 `ExecutorStepOutput`

这样做的目的和 planner 一样：

- 不绕过 parser
- 不绕过 schema 校验
- 让 fake 路径和未来真实模型路径一致

---

## 9. 当前 executor 和 messages 的关系

当前 executor request 会直接读取：

- `messages`

这里的 `messages` 指的是：

- 当前 trajectory 的滚动消息历史
- 包含 user 问题
- 包含已执行 step 的 assistant / tool 历史
- 不包含未执行 planner proposal

这意味着：

> executor 当前能看到“之前已经执行过什么”，因此具备在上一步基础上继续构思下一步的上下文。

---

## 10. 当前 executor 和 visible_images 的关系

当前 executor request 直接读取：

- `visible_images`

它的意义是：

- 当前真正给 executor 看的图像集合

但当前还没有在代码里实现：

- visible image 的默认传播策略
- 历史图显式回看策略

后续最合理的默认仍然应是：

1. root images
2. 最新一步主图

---

## 11. 当前 executor 和 runtime 的边界

当前 executor 只生成：

- `cot`
- `code`

runtime 负责：

- 读取 `executor_code.py`
- 读取 `visible_images`
- 真执行一次
- 产出 `runtime_result.json`

也就是说：

> executor 不是执行器，runtime 才是真执行器。

---

## 12. 当前已完成验证

本轮已完成的验证包括：

1. executor 相关 Python 文件可 `py_compile`
2. fake backend -> executor client -> `ExecutorStepOutput` 的最小链将被验证
3. `ExecutorStepOutput` 会执行 schema 校验

---

## 13. 当前还没做的部分

executor 这一侧当前还没有：

1. 真实 `ApiTextBackend`
2. executor tool reference 的更细说明
3. step messages 的自动构造
4. 和 `store.write_executor_step_files(...)` 的主循环接线
5. visible image selector

所以当前状态是：

- executor client 骨架已可用
- 但还没进入 runtime 闭环

---

## 14. 一句话版本

当前 executor 已经落成：

> 它能读取当前 step 上下文、当前可见图、历史消息和能力目录，按 `<think> + <code>` 协议解析模型返回，并返回一个 schema 合法的 `ExecutorStepOutput`；接下来要补的是 runtime 接线、step messages 和 orchestrator。
