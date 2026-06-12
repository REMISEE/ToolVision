# 14 Step 3：Judge Client 实现记录

日期：2026-03-29  
状态：已实现  
目的：记录当前 `JudgeClient` 的代码落地情况，明确它读什么、产什么、judge backend 返回什么，以及它和本地 policy / orchestrator 的边界。

---

## 1. 一句话结论

当前 judge 这一条链已经落成到可直接使用 fake backend 跑通的程度：

1. 有 request 对象
2. 有 judge backend 协议
3. 有 fake judge backend
4. 有 `JudgeClient`
5. 能返回 schema 合法的 `JudgeRecord`

也就是说：

> judge 现在已经不是单纯设计概念，而是一个真实可接 orchestrator 的评分 client 骨架。

---

## 2. 当前 judge 相关文件

当前 judge 依赖的主要文件有：

- `offline_sft_pipeline/pipelines/backends.py`
- `offline_sft_pipeline/pipelines/request_models.py`
- `offline_sft_pipeline/pipelines/judge_client.py`
- `offline_sft_pipeline/prompts/judge_system_v01.txt`
- `offline_sft_pipeline/prompts/judge_user_v01.txt`

其中：

- `backends.py`
  - 新增：
    - `JudgeBackendResult`
    - `JudgeBackend`
    - `FakeJudgeBackend`
    - `CommitteeJudgeBackend(TODO)`
- `request_models.py`
  - 新增 `JudgeClientRequest`
- `judge_client.py`
  - 负责：
    - backend 调用
    - score -> `JudgeRecord` 拼装
    - schema 校验
- `prompts/*.txt`
  - 当前先作为 committee judge 后续接入的占位文件

---

## 3. 当前 judge backend 返回什么

当前 judge backend 返回的不是 `JudgeRecord`，而是一个更轻的内部对象：

- `JudgeBackendResult`

字段如下：

- `overall_score`
- `per_model_scores`
- `metadata`
- `note`

这是当前最重要的设计点。

原因是：

- judge 的复杂性应当压在 backend 里
- backend 可以内部调用多个模型
- `JudgeClient` 不应该直接处理多模型聚合逻辑

所以当前语义是：

> backend 负责“打分并聚合”，client 负责“写成标准记录”。

---

## 4. 当前 judge request 读什么

当前 judge 统一读取 `JudgeClientRequest`。

字段如下：

- `sample_id`
- `trajectory_id`
- `scope_type`
- `scope_step_idx`
- `judge_stage`
- `question`
- `messages`
- `visible_images`
- `planner_output`
- `step_record`
- `runtime_result`
- `final_answer`
- `metadata`

这里最关键的是：

1. `question`
   - 原始问题
2. `messages`
   - 当前 trajectory 的历史消息
3. `visible_images`
   - 当前 judge 真正参考的图片集合
4. `step_record`
   - 当前 step 的系统登记信息
5. `runtime_result`
   - 当前 step 的真实执行结果

也就是说：

> judge 当前不是只看当前图片，而是能结合问题、历史记录、当前 step 语义和 runtime 结果一起判断。

---

## 5. 当前 JudgeClient 怎么工作

`JudgeClient.run(...)` 当前流程是：

1. 接收 `JudgeClientRequest`
2. 调 `JudgeBackend.score(...)`
3. 得到 `JudgeBackendResult`
4. 补齐系统字段：
   - `judge_record_id`
   - `sample_id`
   - `trajectory_id`
   - `scope_type`
   - `scope_step_idx`
   - `judge_stage`
   - `created_at`
5. 本地根据 policy 派生：
   - `keep_for_frontier`
   - `exportable`
6. 生成 `JudgeRecord`
7. 执行 schema 校验

也就是说：

> 当前 judge 的 hot path 仍然兼容 Step 0 文档里“总分由模型返回，继续/停止规则由本地 policy 决定”的思路。

---

## 6. 当前本地 policy 是什么

当前 `JudgeClient` 内部有一个轻量 policy：

- `keep_threshold = 0.25`
- `export_threshold = 0.85`

对应行为：

- `overall_score >= keep_threshold`
  - `keep_for_frontier = true`
- `scope_type == "trajectory"` 且 `overall_score >= export_threshold`
  - `exportable = true`

这个 policy 只是当前最小骨架。

后续完全可以：

- 替换成更复杂的 frontier / stop 规则
- 引入 delta score
- 引入 budget 约束

但当前至少已经满足：

- `JudgeRecord` 必填字段能被稳定填上

---

## 7. fake judge backend 当前是什么形态

当前 fake judge backend 在：

- `offline_sft_pipeline/pipelines/backends.py`

已经实现：

- 默认 `overall_score`
- 默认 `per_model_scores`
- 默认 `metadata`
- 默认 `note`

它不会直接返回 `JudgeRecord`。

这样做的目的和 planner / executor 一样：

- 不把 client 逻辑绕掉
- 保留将来接入真实 committee judge 的接口形状

---

## 8. 当前 committee judge 的位置

当前已经补了：

- `CommitteeJudgeBackend`

但它还是：

- `NotImplementedError`

它后续的职责已经很明确：

1. 调 10 个 judge model
2. 收集单模型打分
3. 聚合为总分
4. 返回 `JudgeBackendResult`

也就是说，后续真正接 judge model 部署时，优先填的应该就是这里。

---

## 9. 当前 judge 和 prompts 的关系

当前 `JudgeClient` 还不读取 prompt 文件。

这是有意为之。

原因是：

- judge 这一侧当前设计成 backend 聚合接口
- 不是文本 tag parser

所以：

- `judge_system_v01.txt`
- `judge_user_v01.txt`

当前只是占位，给以后 `CommitteeJudgeBackend` 内部 prompt wiring 用。

---

## 10. 当前 judge 的边界

judge 当前负责：

- 消费聚合后的评分结果
- 写标准 `JudgeRecord`

judge 当前不负责：

- 直接更新 `trajectory.status`
- 直接更新 frontier
- 决定 child trajectory fork
- 改写 messages

这些仍然属于：

- orchestrator

也就是说，当前边界是：

> judge 只给出标准记录，不直接驱动主循环状态变更。

---

## 11. 当前已完成验证

本轮已完成的验证包括：

1. judge 相关 Python 文件可 `py_compile`
2. fake judge backend -> judge client -> `JudgeRecord` 的最小链将被验证
3. `JudgeRecord` 会执行 schema 校验

---

## 12. 当前还没做的部分

judge 这一侧当前还没有：

1. 真实 `CommitteeJudgeBackend`
2. 多模型调用与聚合逻辑
3. richer note 生成
4. delta score / stop rule 与 orchestrator 的联动
5. judge prompt 的真实接线

所以当前状态是：

- judge client 骨架已可用
- 但真实 judge stack 还没接上

---

## 13. 一句话版本

当前 judge 已经落成：

> 它能读取问题、历史消息、当前 step 记录和 runtime 结果，通过 `JudgeBackend` 获取聚合评分，再写成一个 schema 合法的 `JudgeRecord`；后续真正要填实的是 committee judge backend 和 orchestrator 的控制策略。
