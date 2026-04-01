# Offline SFT Pipeline 说明与 Schema 解读

日期：2026-03-26  
适用目录：`offline_sft_pipeline/`  
目的：把当前项目的运行逻辑、5 份 schema 的职责、与现有 `CodeVision` 数据格式的关系讲清楚，便于继续推进实现。

---

## 1. 当前已经对齐的主循环

当前主循环不是：

`planner -> executor -> runtime -> planner`

而是：

`planner -> executor -> runtime -> judge/frontier -> planner`

停止 trajectory 的两种主要情况：

1. `planner` 看完新图和新历史后，判断已经可以直接回答。
2. `judge/frontier` 判断这条 trajectory 不值得继续扩展，需要终止、剪枝或停早。

所以：

- `planner` 负责“下一步该怎么走，或者现在能不能直接答”
- `judge/frontier` 负责“这条轨迹还值不值得继续投入预算”

这两者都可以导致 trajectory 停止，但职责不同。

---

## 2. 先建立一个简单心智模型

先不要把 5 份 schema 当成“5 个都要直接喂模型的复杂 JSON”。

更容易理解的方式是把它们分成 3 层：

### 2.1 运行与状态层

这是后台 orchestration 真正在维护的东西。

- `trajectory_schema.json`
- `planner_output_schema.json`
- `executor_runtime_result_schema.json`
- `judge_record_schema.json`

它们的用途是：

- 跑流程
- 存状态
- 断点恢复
- 回放调试
- 做 judge 和 frontier 控制

这几份 schema 主要给“系统自己”用，不是最终训练样本本体。

### 2.2 导出层

- `canonical_sft_sample_schema.json`

它的用途是：

- 把一条终止 trajectory 线性化
- 变成更接近训练数据的对象
- 但它仍然是“导出中间层”，不是最终唯一训练格式

### 2.3 最终训练层

这层目前仓库里真正成熟、简单、可参考的格式其实是：

```json
{
  "messages": [...],
  "tools": [...],
  "enable_thinking": true
}
```

也就是 `CodeVision` / `verl` / `LLaMA-Factory` 都已经能理解的多轮 message 数据。

所以要记住一句话：

> trajectory 是后台状态对象，messages 才是最终训练的核心对象。

---

## 3. 5 份 schema 分别在干什么

## 3.1 `planner_output_schema.json`

这个对象表示：

“当前这一轮 planner 的产物是什么”

核心字段：

- `can_answer_now`
- `global_chain_cot`
- `direct_answer`
- `suggestions`

直观理解：

- 如果 `can_answer_now=true`，说明 planner 认为已经可以结束这条轨迹，`direct_answer` 应该有值，`suggestions` 应为空。
- 如果 `can_answer_now=false`，说明还要继续走，`suggestions` 里会给 1 到 3 条候选路线。

每条 `suggestion` 下面又是多个 step，每个 step 至少有：

- `step_goal`
- `capability_plan`
- `executor_instruction`

当前新的对齐建议：

- `capability_plan` 先按“真实 helper 名”写，不强行抽象成 `detect/segment/...`
- 例如：
  - `_call_ground_box`
  - `_call_dino_crop`
  - `_call_ocr_assist`

这样更贴近当前真实底座，也更容易让 planner 输出可执行的计划。

---

## 3.2 `executor_runtime_result_schema.json`

这个对象表示：

“executor 给出一段代码后，runtime 真正执行这一 step 的结果是什么”

核心字段：

- `success`
- `images`
- `text`
- `meta`
- `observed_helper_calls`
- `code_execution`
- `error`

直观理解：

- 这是“单步执行结果”
- 一次 step 执行一次，产出一个 runtime result
- 它应该尽量接近 `CodeImageTool` 底层 helper 的真实返回

这里最重要的不是“模型要看什么”，而是：

- 这一步跑没跑成功
- 生成了哪些图
- 工具到底返回了什么文本
- 调了哪些 helper
- 失败时怎么定位

### 当前 schema 是否能直接用

大体可以直接用。

它的优点是已经把单步执行里最关键的东西都放进去了：

- 图像产物
- 文本产物
- 结构化 `meta`
- helper 观测
- stdout/stderr/error

### 当前可能需要轻微收敛的地方

1. `code_execution.exit_code`
   - 如果底层仍然主要经 `CodeImageTool.execute()` 跑，而不是 shell 脚本子进程，`exit_code` 语义会偏弱。
   - 但保留它问题不大，可以约定：
     - 成功为 `0`
     - 代码校验失败 / 执行失败给非 0

2. `observed_helper_calls`
   - 这个字段非常有用，建议保留。
   - 但实现时要么通过 wrapper 显式埋点，要么先做最小版，只记录 helper 名和顺序。

3. `meta`
   - 这个字段必须保留。
   - 因为 offline pipeline 需要的很多中间语义，当前 `ToolResponse` 本体并不直接承载。

结论：

> 这份 schema 不是太重，反而是当前 5 份里最值得尽快落地的一份。

---

## 3.3 `judge_record_schema.json`

这个对象表示：

“judge 对一步或一条轨迹的判断结果”

核心字段：

- `scope_type`
- `judge_stage`
- `keep_for_frontier`
- `exportable`
- `overall_score`
- `note`

直观理解：

- `scope_type=step` 时，可以给某一步打分
- `scope_type=trajectory` 时，可以给整条轨迹打分
- `keep_for_frontier` 决定它还要不要继续扩展
- `exportable` 决定它后续还要不要进入导出集合

### 当前 schema 是否能直接用

基本可以直接用。

### 当前略显超前的部分

- `weak_model_summary`
- `committee_summary`

这两个字段原本是给未来 judge committee 预留的。

状态更新（2026-03-29）：

- Step 0 冻结后，v0.1 最小 schema 已移除这两个字段
- 当前热路径只保留单分数 judge record

在更早的设计阶段，V0.1 如果还没有弱模型 committee，可以先：

- 置为 `null`
- 只跑 `cheap_filter`

这段历史说明保留在这里，仅用于解释最初设计思路。

---

## 3.4 `trajectory_schema.json`

这个对象表示：

“一条 trajectory 当前的总状态”

这是整个 offline pipeline 最核心的状态对象。

它不是训练样本，而是 orchestrator 的后台索引。

核心字段：

- `status`
- `round_idx`
- `step_idx`
- `messages_path`
- `planner_history`
- `pending_execution`
- `steps`
- `judge_records`
- `final_answer`
- `budget`

直观理解：

- `trajectory.json` 只放索引和状态
- 大块内容放在独立文件里，再由路径引用回来

比如：

- 完整 messages 在 `messages.json`
- 某轮 planner 输出在 `planner_round_000.json`
- 某步 runtime 结果在 `steps/step_001/runtime_result.json`

### 当前 schema 是否能直接用

基本能直接用。

### 当前略重或略绕的地方

1. `messages_path`
   - 很合理，建议保留。

2. `planner_history + latest_planner_round_idx + latest_planner_output_path`
   - 这里有一点重复，但属于可接受重复。
   - 原因是：
     - `planner_history` 用于完整回放
     - `latest_*` 用于快速恢复和索引

3. `pending_execution`
   - 很值得保留。
   - 它是“断点恢复”的关键。
   - 否则你重启后很难知道下一步该执行哪条 suggestion 的哪一个 step。

4. `steps`
   - 值得保留。
   - 因为它记录了已经真正执行过的 step。

结论：

> 这份 schema 略长，但不是无意义地长。它长的原因是它承担了 orchestration 状态总表的角色。

---

## 3.5 `canonical_sft_sample_schema.json`

这个对象最容易让人困惑。

它不是 trajectory。
它也不是 judge。
它表示：

“从一条终止 trajectory 里整理出来的一条线性训练样本”

核心字段：

- `tools`
- `artifacts`
- `messages`
- `final_answer`
- `judge_summary`

### 为什么会有这一层

因为后台 trajectory 很脏、很重：

- 有分叉信息
- 有 judge 记录
- 有中间状态
- 有很多调度信息

这些东西不适合直接拿去训练。

所以需要一个“中间导出层”，把一条终止 trajectory 压成更干净的线性样本。

### 但它是不是最终必须的训练格式

不一定。

如果你后面确认最终训练就是对齐 `CodeVision` 风格的数据，那么最终真正要写出去的最小对象大概率是：

```json
{
  "messages": [...],
  "tools": [...],
  "enable_thinking": true
}
```

所以：

- `canonical_sft_sample_schema.json` 更像 exporter 内部的“标准中间对象”
- 最终还可以再转一层，变成你真正想要的 `CodeVision-SFT-like` JSON/JSONL

### 当前它有没有冗余

有一点。

`final_answer` 和 `messages` 里的最后一个 assistant 回答会部分重复。

但这个重复不是致命问题，因为：

- `final_answer` 方便快速检索和分析
- `messages` 才是训练真正要用的上下文

如果后面你觉得这个重复没有价值，也可以删掉 `final_answer`。

---

## 4. 一句话总结：这 5 份 schema 现在能不能直接用

结论：

### 4.1 能直接用的部分

- `planner_output_schema.json`
- `executor_runtime_result_schema.json`
- `judge_record_schema.json`
- `trajectory_schema.json`

这 4 份已经足够作为 v0.1 的工程接口。

### 4.2 需要带着“中间层”心态使用的部分

- `canonical_sft_sample_schema.json`

它不是错，而是现在容易让人误以为“这就是最终训练格式”。

更准确的理解应该是：

> 它是 exporter 的中间标准层，不一定等于最后喂训练的数据文件长相。

---

## 5. 从输入样本到最终样本，顺一遍完整流程

下面用一个简化例子说明。

任务：

- 输入图片是一张商品图
- 问题是“价格标签上写的数字是多少？”

---

## 5.1 初始输入

最小 root sample 可以非常简单：

```json
{
  "sample_id": "sample_0001",
  "question": "价格标签上写的数字是多少？",
  "image_path": "inputs/sample_0001.png"
}
```

这个对象目前还没有专门 schema。

它只是“生成 pipeline 的入口样本”。

后续可以再补一份 `input_sample_schema.json`，但不是本周第一优先级。

---

## 5.2 创建 root trajectory

orchestrator 收到这个 root sample 后，创建：

- `trajectory.json`
- `messages.json`
- 原始图 artifact

这时 `messages.json` 很可能长这样：

```json
[
  {"message_id": "m_sys", "role": "system", "content": "You are a helpful vision tool-use assistant.", "image_artifact_ids": [], "metadata": {}},
  {"message_id": "m_user", "role": "user", "content": "价格标签上写的数字是多少？", "image_artifact_ids": ["img_root"], "metadata": {}}
]
```

此时 trajectory 的状态大致是：

- `status = "running"`
- `round_idx = 0`
- `step_idx = 0`
- `pending_execution = null`

---

## 5.3 planner 第 0 轮

planner 读取当前历史后输出：

```json
{
  "can_answer_now": false,
  "global_chain_cot": "先定位价格标签，再放大或裁剪，再 OCR。",
  "suggestions": [
    {
      "suggestion_id": "s1",
      "suggestion_cot": "先检测标签区域，再 OCR。",
      "steps": [
        {
          "step_id": "step_a",
          "step_goal": "定位价格标签并裁出局部区域",
          "capability_plan": [
            {"order": 1, "capability": "_call_ground_box", "instruction": "找出 price tag 或 label"},
            {"order": 2, "capability": "_call_dino_crop", "instruction": "裁出最可能的标签区域"}
          ],
          "executor_instruction": "写代码先框出标签，再裁剪标签区域，返回裁剪图。"
        }
      ]
    }
  ]
}
```

然后 trajectory 更新为：

- `latest_planner_round_idx = 0`
- `pending_execution = { planner_round_idx: 0, suggestion_id: "s1", suggestion_step_index: 0, step_id: "step_a" }`

---

## 5.4 executor 执行当前 step

executor 读取：

- 全部历史消息
- 第 0 轮 planner 输出
- 被选中的 `s1.step_a`

然后只生成这一 step 的内容，例如：

- `executor_cot.md`
- `executor_code.py`

`executor_code.py` 可能类似：

```python
box = _call_ground_box("price tag. label.")
crop = _call_dino_crop("price tag. label.", image_obj=box["image"], based_on="box", max_crops=1, padding=8)
print(crop["text"])
result = crop["image"]
```

注意：

- executor 只生成“当前一步”
- 不是把整条 suggestion 一次性跑完

---

## 5.5 runtime 执行这一段代码

这就是“单步 runtime wrapper”的含义。

它的职责非常具体：

输入：

- 一段 executor 代码
- 当前可见图片列表
- 当前 step 上下文信息

处理：

- 调 `CodeImageTool`
- 执行一次
- 采集图像/text/meta/stdout/stderr/helper 调用信息
- 保存 artifact

输出：

- 一个 `runtime_result.json`

比如：

```json
{
  "success": true,
  "images": [{"artifact_id": "img_step_1", "path": "steps/step_001/output_0.png"}],
  "text": "detected 1 region and returned 1 crop",
  "meta": {"model": "grounded_sam2", "operation": "dino_crop"},
  "observed_helper_call_count": 2,
  "observed_helper_calls": [
    {"order": 1, "name": "_call_ground_box", "status": "ok"},
    {"order": 2, "name": "_call_dino_crop", "status": "ok"}
  ],
  "error": null
}
```

然后把这一步转成新的多轮消息追加回 `messages.json`：

1. assistant：记录这一步的思考和代码
2. tool：记录工具返回文本和新图像

---

## 5.6 judge / frontier

judge 看到这一步的新结果后，做判断。

例如 cheap filter 给出：

```json
{
  "scope_type": "trajectory",
  "judge_stage": "cheap_filter",
  "keep_for_frontier": true,
  "exportable": true,
  "overall_score": 0.72,
  "note": "runtime success, new crop is valid, trajectory should continue"
}
```

如果结果很差，也可能：

- `keep_for_frontier = false`
- trajectory 进入 `pruned` 或 `stopped_early`

---

## 5.7 planner 下一轮

如果 judge 没剪掉它，这条 trajectory 会进入下一轮 planner。

这时 planner 看到的是：

- 原始问题
- 原始图
- 新 crop 图
- tool 返回文本
- 已执行过的一步历史

planner 可能判断：

1. 现在直接 OCR 就够了  
或
2. 已经可以直接回答了

如果它觉得能直接答：

- `can_answer_now = true`
- `direct_answer = "价格标签写着 39"`
- trajectory 终止为 `answered`

---

## 5.8 导出成线性训练样本

最后 exporter 会把这条终止 trajectory 线性化。

如果你后面要的是 `CodeVision-SFT-like` 格式，那么最终更关心的是：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...当前 step thinking + code 或 tool call..."},
    {"role": "tool", "content": "...tool 返回文本/图像占位..."},
    {"role": "assistant", "content": "最终答案"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "code_image_tool",
        "description": "...",
        "parameters": {...}
      }
    }
  ],
  "enable_thinking": true
}
```

这才是离现有 `CodeVision` 训练数据最接近的对象。

---

## 6. 为什么我前面说“先做单步 runtime wrapper”

这里的“单步 runtime wrapper”不是新搞一套复杂系统。

它只是一个非常小但非常关键的模块：

> 给我一段 executor 代码和当前可见图片，我帮你执行一次，并把结果整理成 `executor_runtime_result_schema.json`

它是后续所有模块的公共底座。

没有它：

- planner 产物没法验证
- store 没法保存真实 step 结果
- judge 没法看真实执行结果
- exporter 没法拿到中间图和工具文本

所以它是最优先的“跑通链路点”。

---

## 7. 下一步到底该从哪开始构造

如果按最稳妥的工程顺序，建议如下：

### Step 1：补一份最小输入样本约定

哪怕先不单独建 schema，也先口头冻结最小字段：

- `sample_id`
- `question`
- `image_path` 或 `image_paths`
- 可选 `answer`
- 可选 `metadata`

原因：

- 否则 root trajectory 从哪来会一直模糊。

### Step 2：实现单步 runtime wrapper

目标：

- 输入一段 executor 代码
- 输入可见图片
- 执行一次
- 产出 `runtime_result.json`

这是第一条必须跑通的真实链路。

### Step 3：实现 `core/models.py`

把现有 5 份 schema 映射成 Python 模型。

这一步适合在 schema 已经看顺眼后做。

### Step 4：实现 `core/store.py`

负责：

- 创建 trajectory 目录
- 读写 `trajectory.json`
- 读写 `messages.json`
- 保存 planner/runtime/judge 文件
- resume

### Step 5：接 planner / executor client

先让：

- planner 产出一轮 JSON
- executor 产出一步代码

不必一上来就追求全自动高质量。

### Step 6：最后再做 exporter

因为 exporter 依赖前面对象都稳定。

---

## 8. 当前 schema 里我建议保留、不建议动的点

建议保留：

- `trajectory.pending_execution`
- `trajectory.steps`
- `runtime_result.meta`
- `runtime_result.observed_helper_calls`
- `judge_record.keep_for_frontier`
- `judge_record.exportable`

这些字段都直接服务于 offline orchestration。

---

## 9. 当前 schema 里可以后续再考虑精简的点

### 9.1 `canonical_sft_sample.final_answer`

和最终 `messages` 最后一条 assistant 内容部分重复。

可后续评估是否删除。

### 9.2 `trajectory.latest_planner_*`

和 `planner_history` 有轻度重复。

但为了恢复效率，当前保留更合适。

### 9.3 `judge_record` 的 committee 相关字段

状态更新（2026-03-29）：

- Step 0 冻结后，v0.1 schema 已删掉这些字段
- 后续如果重新引入 committee judge，再单独扩 schema

---

## 10. 当前版本最推荐的对齐结论

1. 主循环冻结为：
   `planner -> executor -> runtime -> judge/frontier -> planner`

2. planner 的 `capability_plan` 先按真实 helper 名来写。

3. `executor_runtime_result_schema.json` 是近期最关键、最应该最早落地的 schema。

4. `trajectory_schema.json` 虽然长，但它承担的是“后台状态总表”职责，不建议现在大砍。

5. `canonical_sft_sample_schema.json` 先理解为 exporter 中间层，不要把它误认为最终训练格式。

6. 最终训练输出优先对齐 `CodeVision` 风格的多轮 `messages/tools/enable_thinking`。

---

## 11. 一句话版本

如果只记一句话：

> 现在最该先做的不是 exporter，也不是大改 schema，而是先定义清楚 root sample，再跑通“executor 一步代码 -> runtime 一次执行 -> 保存 image/text/meta -> 回写 messages/trajectory”这条最短闭环。
