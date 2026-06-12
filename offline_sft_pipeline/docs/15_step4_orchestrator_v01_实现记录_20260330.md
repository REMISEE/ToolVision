# 15 Step 4：Orchestrator v01 实现记录

日期：2026-03-30  
状态：已实现  
目的：记录 `offline_sft_pipeline/pipelines/orchestrator_v01.py` 的当前落地状态，明确它如何把 `planner -> executor -> runtime -> judge -> trajectory/store` 串起来，尤其说明：

- 多 `suggestion` 怎么处理
- 多步 `steps` 现在怎么理解
- 自定义 `k` 个 suggestions 怎么传
- 全局上限 `M` 怎么收束 child trajectories
- 各类特例和边界情况当前怎么走

---

## 1. 一句话结论

当前 `orchestrator_v01.py` 已经不是空骨架，而是一个可运行的最小 branching orchestrator。

它现在已经能做下面这些事：

1. 初始化 root trajectory
2. 调 planner 产出一轮 `PlannerOutput`
3. 支持 planner 一次给出多条 `suggestions`
4. 支持 suggestion 内部带多步 `steps`
5. 先做 frontier selection，再决定哪些 suggestion 真正变成 child trajectory
6. 每个 child trajectory 只执行当前被选 suggestion 的第 1 步
7. 调 executor 产出 `ExecutorStepOutput`
8. 调 runtime 执行
9. 生成 assistant/tool message
10. 注册 `StepRecord`
11. 调 judge 产出 `JudgeRecord`
12. 根据结果更新 trajectory 状态

也就是说：

> Step 4 这次真正补上的，是 offline pipeline 的第一版“主循环调度器”。

---

## 2. 当前新增 / 改动文件

### 2.1 核心新增

- `offline_sft_pipeline/pipelines/orchestrator_v01.py`

### 2.2 为 orchestrator 配套补的小改动

- `offline_sft_pipeline/pipelines/request_models.py`
  - `PlannerClientRequest` 新增 `requested_suggestion_count`
- `offline_sft_pipeline/pipelines/planner_client.py`
  - planner user prompt 注入 `requested_suggestion_count`
- `offline_sft_pipeline/prompts/planner_system_v01.txt`
  - 增加“尽量遵守 requested suggestion count”的约束
- `offline_sft_pipeline/prompts/planner_user_v01.txt`
  - 增加 `requested_suggestion_count`
- `offline_sft_pipeline/pipelines/__init__.py`
  - 导出 `OrchestratorV01` / `OrchestratorConfig` / `OrchestratorRunResult`

---

## 3. 当前 orchestrator 的职责边界

当前 orchestrator 负责的是：

1. 调 planner / executor / judge client
2. 调 runtime
3. 做 frontier selection
4. 决定哪些 suggestion 被实例化为 child trajectory
5. 把 step 执行结果写回 `messages.json` / `trajectory.json`
6. 更新 running / expanded / terminal 状态

当前 orchestrator 不负责的是：

1. 真实大模型调用
   - 这仍是 `ApiTextBackend` 以后再接
2. 真实 committee judge 聚合
   - 这仍是 `CommitteeJudgeBackend` 以后再接
3. canonical export
4. replay / inspect 工具
5. 更复杂的 delta-score stop policy

所以当前边界可以总结成：

> `orchestrator_v01` 已经把热路径主循环串起来了，但还不是完整生产版调度系统。

---

## 4. 关键配置项

当前配置对象是：

- `OrchestratorConfig`

主要字段如下。

### 4.1 `planner_suggestion_count`

含义：

- 希望 planner 这一轮最多给出多少条 suggestion

当前特点：

- 会通过 `PlannerClientRequest.requested_suggestion_count` 传给 planner prompt
- 当前实现会再被 schema 上限收束到最多 `3`

也就是说：

- 你可以自定义 `k`
- 但当前有效范围其实是 `1..3`

### 4.2 `max_child_trajectories`

含义：

- 当前这轮所有父 trajectory 加起来，最多实例化多少条 child trajectory

这就是全局上限 `M`。

它的作用是：

- planner 可以提多个 proposal
- 但 executor / runtime 不会无限展开

### 4.3 `default_budget`

含义：

- root trajectory 的默认预算

当前 budget 仍沿用：

- `remaining_rounds`
- `remaining_children`
- `remaining_steps`

### 4.4 `judge_stage`

当前默认：

- `"cheap_filter"`

也就是说，当前主循环默认接的是最小同步 judge。

### 4.5 `stop_unselected_trajectories`

当前默认：

- `True`

含义：

- 如果某个父 trajectory 本轮没有 suggestion 被选进 frontier，它会被收束成 terminal，而不是继续留作 running leaf

这和最近 v0.1 文档里的“先 frontier selection，再实例化 child；未选中的 suggestion 只留在 planner output，不继续展开”是一致的。

---

## 5. 当前主循环实际怎么跑

当前 `run(...)` 的热路径可以概括为：

1. `store.init_root_trajectory(...)`
2. 取当前 frontier 中的 `running` trajectories
3. 对每条 running trajectory 调 planner
4. 汇总所有 planner outputs
5. 从所有 suggestions 中做全局 frontier selection
6. 对被选中的 suggestion：
   - fork child trajectory
   - executor 只执行该 suggestion 的第 1 步
   - runtime 执行
   - 追加 messages
   - 注册 step
   - judge
   - 决定 child trajectory 后续状态
7. 父 trajectory 设为 `expanded`
8. 如果某条 child 还可继续，则进入下一轮 frontier
9. frontier 为空时结束

这个流程和最近文档已经基本一致。

---

## 6. 多 suggestion 现在怎么支持

### 6.1 planner 可以返回多条 suggestion

当前 `PlannerOutput.suggestions` 仍然允许最多 3 条 suggestion。

orchestrator 不再只取：

- `suggestions[0]`

而是会读取一整轮 planner 返回的多个 suggestion。

### 6.2 不是全部 suggestion 都会直接执行

这是这次实现最关键的点。

当前做法不是：

- planner 给 3 条
- 就创建 3 个 child 全跑

而是：

1. planner 先提案
2. orchestrator 汇总所有父 trajectory 的 suggestions
3. 先做一轮 frontier selection
4. 只有被选中的 suggestion 才实例化为 child trajectory

也就是说：

> planner 负责 proposal，orchestrator 负责 selection。

### 6.3 当前 frontier selection 规则

当前实现遵循最近文档里建议的简单版规则：

1. 先从每个父 trajectory 的 suggestions 里取 top-1
2. 如果 top-1 的总数已经超过 `M`
   - 按全局优先级截断到 `M`
3. 如果 top-1 的总数少于 `M`
   - 再从剩余 suggestion 里按优先级补满

当前全局优先级是：

1. 父 trajectory 最近一次对应 `judge_stage` 的 `overall_score`
2. suggestion 在 planner 输出中的顺序
3. 父 trajectory id

如果某条 parent 还没有 judge score，则默认分数是：

- `0.0`

### 6.4 一个具体例子

假设当前 frontier 里有两条 running trajectories：

- `traj_A`
- `traj_B`

它们的 planner 输出分别是：

- `traj_A` -> `A1, A2, A3`
- `traj_B` -> `B1, B2`

配置：

- `k = 3`
- `M = 3`

那么当前 selection 过程是：

1. 先取 top-1：
   - `A1`
   - `B1`
2. 现在总数是 2，小于 `M=3`
3. 再从剩余里按优先级补 1 条：
   - 可能补 `A2`

最终本轮真正实例化 child 的 suggestion 可能是：

- `A1`
- `B1`
- `A2`

而不是：

- `A1`
- `A2`
- `A3`
- `B1`
- `B2`

---

## 7. 多步 `steps` 现在怎么支持

### 7.1 当前已经支持 planner 输出多步 proposal

当前 schema 和 parser 都允许：

- 一条 suggestion 内有多个 `steps`

例如：

```json
{
  "suggestion_id": "s1",
  "suggestion_cot": "先 ground，再 crop，再 OCR。",
  "steps": [
    {"step_id": "step_ground", "...": "..."},
    {"step_id": "step_crop", "...": "..."},
    {"step_id": "step_ocr", "...": "..."}
  ]
}
```

这部分现在是被完整保留在：

- `planner/round_xxx.json`

里的。

### 7.2 但当前不会自动顺着旧链把第 2 步、第 3 步跑完

当前 orchestrator 的真实执行语义仍然是：

- 被选中的 suggestion 只执行第 1 步

也就是：

- `steps[0]`

对应的 `step_spec`

这不是漏实现，而是当前版本有意保持和总 spec 一致：

> 每次执行一步后重新进入 planner，而不是沿旧 suggestion 机械续跑。

### 7.3 这意味着什么

这意味着当前“支持多步 steps”指的是：

1. planner 可以把未来路线表达成多步 proposal
2. 这些未来步骤会被落盘保存
3. orchestrator 会把它们视为“候选未来后缀”
4. 但执行器只消费当前一步

不是：

1. 生成一条 3-step suggestion
2. 然后 executor 连续跑完 3 步

### 7.4 一个具体例子

planner 当前轮给出：

- suggestion `s1`
  - step 1: ground object
  - step 2: crop object
  - step 3: OCR text

当前 orchestrator 会做的是：

1. 如果 `s1` 被选中
2. 创建 child trajectory
3. 只执行：
   - `step 1`
4. 执行完后回到 planner
5. planner 再基于“新图 + 新 messages + 新 runtime result”重新规划

所以：

- `step 2`
- `step 3`

当前只是 proposal history，不是 pending fixed script。

---

## 8. `k` 和 `M` 当前怎么协同

### 8.1 `k`

`k` 是：

- 单个 planner round 希望产出的 suggestion 数

当前入口是：

- `OrchestratorConfig.planner_suggestion_count`

然后通过：

- `PlannerClientRequest.requested_suggestion_count`

传进 prompt。

### 8.2 `M`

`M` 是：

- 当前这轮所有 parent trajectories 总共最多落多少条 child trajectories

当前入口是：

- `OrchestratorConfig.max_child_trajectories`

### 8.3 一个直接例子

假设：

- frontier 当前有 3 个 parent trajectories
- 每个 parent planner 都给 `k=3`

那么理论上一共会有：

- `3 * 3 = 9` 条 candidate suggestions

如果：

- `M = 4`

则最终真正实例化成 child trajectories 的最多只有：

- 4 条

其余 suggestion：

- 保留在 planner round 文件里
- 不进入 executor / runtime

### 8.4 当前的一个硬约束

虽然 orchestrator 现在已经支持传 `k`，但当前仍有一个 schema 限制：

- `planner_output_schema.json` 里 `suggestions.maxItems = 3`

所以当前真实可用范围是：

- `k <= 3`

如果以后要支持：

- `k = 5`
- `k = 8`

则需要同步调整：

1. planner schema
2. planner prompt
3. 相关验证逻辑

---

## 9. 当前 messages 是怎么写的

当前 orchestrator 已经把 step 执行结果线性化写回 `messages.json`。

### 9.1 每执行一步后追加两条消息

1. `assistant`
2. `tool`

### 9.2 assistant message 当前长什么样

当前格式大致是：

```text
<think>
...executor cot...
</think>
<tool_call name="code_image_tool">
...executor code...
</tool_call>
```

metadata 里会带：

- `message_kind = "executor_step"`
- `step_idx`
- `planner_round_idx`
- `suggestion_id`
- `step_id`
- `executor_code_path`

### 9.3 tool message 当前长什么样

当前格式是：

- `content = runtime_result.text`
- `image_artifact_ids = runtime_result.images[*].artifact_id`

metadata 里会带：

- `message_kind = "tool_result"`
- `step_idx`
- `tool_name = "code_image_tool"`
- `runtime_result_path`
- `primary_image_artifact_id`

### 9.4 final answer 当前怎么写

如果 planner 直接回答，则当前会追加一条：

- `role = "assistant"`
- `content = <answer>...</answer>`

这条消息当前是在 orchestrator 里直接生成的。

---

## 10. 当前 visible images 策略

当前 orchestrator 已经把“默认带哪些图进下一轮”的规则固化到了代码里。

默认可见图是：

1. root images
2. 最新一步主输出图

其中：

- 主输出图 = `runtime_result.images[0]`

辅助图虽然会被保存，也会挂在 tool message 上，但当前不默认继续带入下一轮 planner / executor。

这和 Step 0 冻结稿一致。

---

## 11. 当前 parent / child 状态怎么更新

### 11.1 parent trajectory

当 parent 某一轮 planner 产出 suggestions，且其中至少一条 suggestion 被选中并成功实例化 child 时：

- parent status -> `expanded`

含义是：

- 它已经不是可继续直接执行的 leaf
- 后续推进交给 child trajectories

### 11.2 child trajectory

child 执行一步并经过 judge 后，当前可能进入：

- `running`
- `pruned`
- `failed`
- `max_step_reached`
- `error`

其中当前规则是：

1. runtime 失败
   - `failed`
2. runtime 成功，但既没有图也没有有效文本
   - `pruned`
3. judge `keep_for_frontier = false`
   - `pruned`
4. step budget 或 round budget 用尽
   - `max_step_reached`
5. 否则
   - `running`

### 11.3 未被选中的 parent 当前怎么处理

如果某个 parent 本轮没有任何 suggestion 被选进 frontier，且：

- `stop_unselected_trajectories = True`

则它会被收束。

当前收束规则是：

1. 如果 child budget 或 step budget 已用尽
   - `max_step_reached`
2. 否则
   - `stopped_early`

---

## 12. 特例 / 边界情况说明

这一节是这份文档最重要的部分。

### 12.1 planner 直接回答

如果 planner 返回：

```text
<think>...</think>
<answer>final answer</answer>
```

当前 orchestrator 会：

1. `register_planner_round(...)`
2. 追加一条 final assistant message
3. `trajectory.status = "answered"`
4. 写入 `final_answer`

它不会：

1. fork child trajectory
2. 调 executor
3. 调 runtime

### 12.2 planner 返回多个 suggestion，但 `M=1`

例如 planner 返回：

- `s1`
- `s2`

配置：

- `k = 2`
- `M = 1`

则当前真实结果是：

1. 只会实例化 1 条 child trajectory
2. 另一条 suggestion 只保留在 planner round 文件里

这个行为已经做过最小 smoke 验证。

### 12.3 planner 返回多步 suggestion

例如：

- `s1.steps = [step_a, step_b, step_c]`

当前 orchestrator 只会执行：

- `step_a`

执行完之后：

- 重新进入 planner

不会：

- 直接继续执行 `step_b`

### 12.4 parent 已经没有 child budget

如果某条 trajectory：

- `remaining_children <= 0`

当前行为是：

1. 仍允许 planner 看一眼当前状态
   - 这样 planner 还有机会直接 `<answer>`
2. 但 orchestrator 不会再把 suggestion 选进 child execution
3. 如果 planner 没有直接回答，这条 trajectory 会被收束成：
   - `max_step_reached`

### 12.5 parent 已经没有 step budget

如果某条 trajectory：

- `remaining_steps <= 0`

当前行为和 child budget 用尽类似：

1. 仍允许 planner 判断是否可直接回答
2. 不再进入新的 step execution
3. 若不能直接回答，则收束成：
   - `max_step_reached`

### 12.6 round budget 用尽

如果某条 running trajectory 在进入 planner 前就已经：

- `remaining_rounds <= 0`

当前实现会直接：

- `status = "max_step_reached"`

不会再调 planner。

### 12.7 executor 出错

如果 executor client 抛异常，当前 child trajectory 会进入：

- `status = "error"`

并写：

- `last_error.code = "executor_error"`

### 12.8 runtime 出错

如果 runtime wrapper 抛异常，当前 child trajectory 会进入：

- `status = "error"`

并写：

- `last_error.code = "runtime_error"`

注意：

这和 runtime 正常返回但 `success = false` 不是一回事。

当前实现里：

- “wrapper 调用阶段直接异常” -> `error`
- “wrapper 正常返回但 result 不成功” -> `failed`

### 12.9 judge 出错

如果 judge client 抛异常，当前 child trajectory 会进入：

- `status = "error"`

并写：

- `last_error.code = "judge_error"`

### 12.10 parent 没有历史 judge score

如果某条 parent trajectory 还没有当前 `judge_stage` 对应的历史 judge record，则：

- frontier priority 默认用 `0.0`

因此当前 top-1 / 补位逻辑仍然能跑，只是优先级会退化到：

- suggestion 顺序
- trajectory id

### 12.11 root trajectory 的 planner round

当前 root 也是完全走同一套主循环，不是单独 hardcode。

也就是说：

1. root init
2. root planner round
3. root 产生 suggestions
4. root 自己转 `expanded`
5. 后续真正执行的是 child trajectories

这和总 spec 中“root proposal -> child fork -> child execution”的语义一致。

---

## 13. 一个端到端例子

下面给一个更完整的例子。

### 13.1 输入

root sample：

- `sample_id = textvqa__train__000001`
- 问题：`价格标签上写的数字是多少？`
- root image：`img_root_0`

配置：

- `k = 2`
- `M = 2`
- `remaining_rounds = 2`
- `remaining_children = 2`
- `remaining_steps = 2`

### 13.2 Round 0 planner

planner 返回：

- `s1`
  - step 1: ground price tag
  - step 2: OCR crop
- `s2`
  - step 1: enhance likely text region
  - step 2: OCR enhanced region

### 13.3 frontier selection

因为：

- 只有 1 个 parent
- `M = 2`

所以：

- `s1`
- `s2`

都会被选中。

### 13.4 child fork

系统创建：

- `traj__textvqa__train__000001__root__r000_s1`
- `traj__textvqa__train__000001__root__r000_s2`

root 自己变成：

- `expanded`

### 13.5 child execution

当前版本只执行每条 suggestion 的第 1 步：

- `s1` child 执行：
  - `ground price tag`
- `s2` child 执行：
  - `enhance likely text region`

执行完之后：

- 各自追加 assistant/tool message
- 各自注册 step record
- 各自得到一步 judge

### 13.6 下一轮

如果某个 child judge 后仍是：

- `running`

它才会进入下一轮 frontier。

下一轮 planner 看到的是：

- 原问题
- 原图
- 已执行的一步 messages
- 最新主输出图

它会重新给出新的 suggestions，而不是沿旧 suggestion 直接继续跑原来的 step 2。

---

## 14. 当前已完成验证

这次实现完成后，已经验证了下面几件事：

1. `orchestrator_v01.py` 可 `py_compile`
2. 最小 stub runtime smoke 已跑通
3. planner 可返回：
   - 多条 suggestion
   - 每条 suggestion 带多步 steps
4. orchestrator 能正确：
   - fork child trajectories
   - 每个 child 只执行当前一步
   - 写 messages
   - 注册 step
   - 调 judge
   - 更新状态
5. `M=1` 时，planner 即使返回 2 条 suggestion，也只会实例化 1 个 child

---

## 15. 当前仍然没有做的部分

虽然 Step 4 已经闭环，但下面这些仍未完成：

1. 更复杂的 judge policy
   - delta score
   - drop threshold
   - 连续停滞判定
2. trajectory-level final judge
3. committee judge backend
4. exporter 主循环接线
5. `run_single_sample_pipeline.py`
6. 更强的 visible image selection 策略
7. suggestion-level explicit score
8. 重试 / resume / partial recovery 策略
9. `core/messages.py`
   - 当前 message builder 还内嵌在 orchestrator
10. `core/judge_policy.py`
   - 当前 stop 规则还内嵌在 orchestrator

---

## 16. 当前最重要的理解结论

这次 Step 4 的实现，最容易被误解的点只有一个：

> “支持多步 steps” 不等于 “自动把一条 suggestion 的所有 step 顺着执行完”。

当前正确理解应该是：

1. planner 可以表达多步未来路线
2. orchestrator 可以处理多条 suggestion
3. orchestrator 可以用 `M` 控制 child 数量
4. child 只执行当前一步
5. 执行完一步后重新规划

这正是当前 offline pipeline 最核心的 `rolling replanning` 语义。

---

## 17. 一句话版本

当前 `orchestrator_v01.py` 已经把 offline branching pipeline 的热路径主循环真正串起来了：

> planner 可以按 `k` 生成多条 suggestion，orchestrator 会先按全局 `M` 做 frontier selection，再只把被选中的 suggestion fork 成 child trajectory；每个 child 只执行当前 suggestion 的第 1 步，执行完后写回 messages / step / judge，并根据预算和 judge 结果进入下一轮或终止。
