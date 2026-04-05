# 22 Planner JSON 协议兼容改动说明

日期：2026-04-04  
状态：已改代码，待和真实 backend 接线方对齐  
目的：给协作同学快速同步这次 planner 输出协议调整到底改了什么、没改什么，以及合并时最需要看的文件。

---

## 1. 一句话结论

这次只改了：

- planner 的“模型原始返回协议”
- `PlannerClient` 的解析逻辑
- 对应 prompt / fake backend / scripted backend

这次没有改：

- `PlannerOutput` 内部字段
- `planner_output_schema.json`
- `store`
- `orchestrator`
- `executor` 协议

也就是说：

> 模型现在可以直接返回 JSON：`mode + think + suggestions/answer`，但 pipeline 内部仍然继续使用原来的 `PlannerOutput(can_answer_now/global_chain_cot/direct_answer/suggestions)`。

---

## 2. 背景

原先 planner 的模型返回协议是：

```text
<think>...</think>
<answer>...</answer>
```

或者：

```text
<think>...</think>
<suggestions>[...]</suggestions>
```

现在遇到的问题是：

- 强模型 / reasoning model 的某些推理模板会吞掉 `<think>`
- 导致 planner 虽然表面上能返回 `<answer>` 或 `<suggestions>`，但拿不到稳定的 `think`

因此本次改成新的模型原始返回协议：

### suggestions 模式

```json
{
  "mode": "suggestions",
  "think": "......",
  "suggestions": [
    {
      "suggestion_id": "s1",
      "suggestion_cot": "......",
      "steps": [
        {
          "step_id": "step_1",
          "step_goal": "......",
          "capability_plan": [
            {
              "order": 1,
              "capability": "ground_box",
              "instruction": "......"
            }
          ],
          "executor_instruction": "......"
        }
      ]
    }
  ]
}
```

### answer 模式

```json
{
  "mode": "answer",
  "think": "......",
  "answer": "......"
}
```

---

## 3. 关键设计决定

### 3.1 不改内部 `PlannerOutput`

当前内部仍然保持：

- `can_answer_now`
- `global_chain_cot`
- `direct_answer`
- `suggestions`

映射规则如下：

- `mode="answer"` -> `can_answer_now=True`
- `think` -> `global_chain_cot`
- `answer` -> `direct_answer`
- `mode="suggestions"` -> `can_answer_now=False`
- `suggestions` -> `suggestions`

这样做的原因是：

- 不影响 `store`
- 不影响 `orchestrator`
- 不影响 `executor_client` 里对 `planner_global_chain_cot` 的读取
- 不需要改现有落盘 schema

### 3.2 解析器做“双协议兼容”

当前 `PlannerClient` 的策略是：

1. 先尝试把模型返回整体当 JSON object 解析
2. 如果存在 `mode` 字段，则走新 JSON 协议
3. 如果不是 JSON object，或没有 `mode`，则回退到旧 `<think>/<answer>/<suggestions>` 标签协议

这意味着：

- 新 backend 可以直接切到 JSON 返回
- 老 fake 数据、老 prompt、历史样例暂时不会立刻失效

---

## 4. 这次实际改了哪些文件

### 必看文件

- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/prompts/planner_system_v01.txt`
- `offline_sft_pipeline/prompts/planner_user_v01.txt`

这三个文件是本次协议改动的核心。

### 同步改的配套文件

- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/scripted_components.py`
- `offline_sft_pipeline/tests/test_orchestrator_v01.py`

它们的作用分别是：

- `backends.py`
  - 默认 fake planner 返回样例改成 JSON 协议
- `scripted_components.py`
  - scripted text backend 渲染 planner 输出时，改成返回 JSON 协议
- `test_orchestrator_v01.py`
  - 新增 parser 级测试，锁住：
    - 新 JSON 协议可解析
    - 旧标签协议仍兼容

---

## 5. 各文件改动点

### `planner_client.py`

新增了两层逻辑：

- `_try_parse_json_contract(...)`
- `_parse_json_contract(...)`

当前解析规则：

- 必须有 `mode`
- 必须有非空 `think`
- `mode="answer"` 时：
  - 必须有非空 `answer`
  - 不能带有效 `suggestions`
- `mode="suggestions"` 时：
  - 必须有 `suggestions` 数组
  - 不能带有效 `answer`

解析成功后，仍然组装成原来的内部字段。

### `planner_system_v01.txt`

从“要求输出 `<think> + <answer>/<suggestions>` 标签协议”改成：

- 输出一个 JSON object
- 包含 `mode`
- 包含 `think`
- 根据 `mode` 输出 `answer` 或 `suggestions`
- 不要加 markdown fence

### `planner_user_v01.txt`

把最后的输出要求改成：

- 能回答时，返回 `mode="answer"` 的 JSON
- 不能回答时，返回 `mode="suggestions"` 的 JSON

### `backends.py`

`DEFAULT_FAKE_PLANNER_TEXT` 已从标签文本改成 JSON 文本。

### `scripted_components.py`

`render_planner_output_as_model_text(...)` 已从：

- 渲染 `<think> + <answer>/<suggestions>`

改成：

- 渲染 JSON object

这样 `ScriptedTextBackend -> PlannerClient` 这条链也会走新的 planner 原始协议。

### `test_orchestrator_v01.py`

新增了两条测试：

1. `test_planner_client_parses_new_json_contract`
2. `test_planner_client_keeps_legacy_tag_contract_compatibility`

当前本地验证命令：

```bash
python -m unittest offline_sft_pipeline.tests.test_orchestrator_v01
```

结果：

- 6 个测试通过

---

## 6. 哪些地方这次明确没动

下面这些这次都没动，如果你那边正在改这些文件，基本不用因为这次协议调整去重构：

- `offline_sft_pipeline/core/models.py`
- `offline_sft_pipeline/schemas/planner_output_schema.json`
- `offline_sft_pipeline/core/store.py`
- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/pipelines/executor_client.py`

原因是：

- 这次只改“模型原始输出格式”
- 不改 pipeline 内部标准结构

---

## 7. 协作同学最需要确认的点

如果你那边负责真实 planner backend 或推理模板，请重点确认下面几件事。

### 7.1 真实 backend 是否能稳定返回纯 JSON object

需要确认：

- 不会自动包 markdown code fence
- 不会额外吐解释性前后缀
- `mode`、`think`、`answer/suggestions` 这几个键能稳定产出

### 7.2 如果 provider 支持 structured output / json schema

建议优先直接约束成：

- `mode`
- `think`
- `answer`
- `suggestions`

而不是再让模型输出 XML tag。

### 7.3 真实 backend 是否仍需要兼容老 tag 协议

当前代码已经兼容。

所以如果你那边短期内：

- 一部分模型还走旧 prompt
- 一部分模型已经切新 JSON prompt

目前是可以共存的。

---

## 8. 合并时最容易冲突的文件

如果你那边也在改真实接线，最可能冲突的是：

- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/prompts/planner_system_v01.txt`
- `offline_sft_pipeline/prompts/planner_user_v01.txt`
- `offline_sft_pipeline/pipelines/backends.py`

其中优先级最高的是：

1. `planner_client.py`
2. `planner_system_v01.txt`
3. `planner_user_v01.txt`

因为这三个文件决定了：

- 模型被要求输出什么
- 客户端到底按什么协议解析

---

## 9. 推荐合并策略

建议按下面顺序看和合：

1. 先确认你那边真实 planner backend 最终想返回的格式是不是 `mode + think + suggestions/answer`
2. 如果是，就直接保留本次 `planner_client.py + prompts` 的改动
3. 如果你那边还没切 backend，只是先接 API，那么也建议保留这次“双协议兼容”逻辑
4. 等真实 backend 稳定后，再决定要不要彻底删除旧 tag 协议 fallback

当前不建议现在就删除旧协议兼容，因为：

- scripted/fake 链路刚同步完
- 真 backend 还在接线中
- 保留 fallback 成本很低，但能减少联调阻塞

---

## 10. 当前结论

这次改动的定位很明确：

> 只把 planner 的模型原始输出协议，从 `<think> + <answer>/<suggestions>`，升级为更适合 reasoning model 的 JSON object 协议；pipeline 内部结构和主循环不变。

如果你那边负责真实 planner backend，请重点看：

- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/prompts/planner_system_v01.txt`
- `offline_sft_pipeline/prompts/planner_user_v01.txt`

如果这三个文件与你那边方案一致，基本就可以直接合。
