# 27 planner与CodeVision图像绑定机制说明 2026-04-15

这份文档回答两个问题：

1. 现在 `offline_sft_pipeline` 里的 planner 逻辑到底是什么，是否已经真正接上，真实 prompt 是怎么拼出来的。
2. 现在 `CodeVision` 在线推理里的 `image_index` 到底是谁维护、谁来选、是否能表达从 `current` 切回 `root`，以及为什么我认为需要补一个新的结构化字段。

---

## 1. 当前 planner 逻辑是否已经接上

结论：

- `planner_system_v05.txt` 已经成为默认 planner sysprompt。
- planner 的轮次策略已经在 orchestrator 里接上，不再只是 prompt 文案。
- planner 的真实调用路径现在是：
  - `OrchestratorV01._plan_trajectory(...)`
  - `PlannerClient.run(...)`
  - `ApiTextBackend.generate(stage="planner", ...)`
  - `planner_to_openai_messages(...)`

对应代码位置：

- `offline_sft_pipeline/pipelines/orchestrator_v01.py`
- `offline_sft_pipeline/pipelines/planner_client.py`
- `offline_sft_pipeline/pipelines/api_text_multimodal.py`
- `offline_sft_pipeline/prompts/planner_system_v05.txt`

但“完全成功”这句话现在还不能下。

更准确的判断是：

- planner 的策略链路已经接通。
- planner 的 prompt 拼接链路已经统一到 system prompt + conversation history + dynamic control block。
- 但是它还不是最终稳定态，因为还有两个残留风险：
  - parser 仍然保留了旧 XML fallback，说明“严格 JSON only”在解析层还没有完全收口。
  - 当前输出协议里仍然没有把 `root/current` 这种输入源语义显式带到最终 CodeVision 在线协议里。

---

## 2. planner 当前真实策略

### 2.1 orchestrator 决定 planner round policy

当前策略在 `offline_sft_pipeline/pipelines/orchestrator_v01.py`：

- `force_first_round_must_suggest = True`
- `must_suggest_score_threshold = 0.6`
- `must_answer_score_threshold = 0.9`

真实决策逻辑在 `_determine_planning_policy(...)`：

1. 如果 `remaining_exec_steps <= 0`
   - `must_answer`
2. 如果命中 `forced_final_answer`
   - `must_answer`
3. 如果是第一轮且 `force_first_round_must_suggest=True`
   - `must_suggest`
4. 如果没有最新 judge score
   - `may_answer_or_suggest`
5. 如果 `latest_overall_score < 0.6`
   - `must_suggest`
6. 如果 `latest_overall_score >= 0.9`
   - `must_answer`
7. 否则
   - `may_answer_or_suggest`

这意味着 planner 现在不是自由决定模式，而是先被 orchestrator 指定 round policy，再根据该 policy 输出。

---

## 3. planner 当前真实 prompt 拼接方式

### 3.1 不是旧的 `planner_user_v01.txt`

当前 planner user prompt 不是从 `planner_user_v01.txt` 读出来直接发的。

真实链路是：

- `PlannerClient.run(...)` 只加载 `planner_system_v05.txt`
- `user_prompt=""`
- 真正的 planner user 控制文本由 `build_planner_control_user_text(...)` 动态生成
- 最终通过 `planner_to_openai_messages(...)` 拼成 OpenAI-style messages

所以当前 planner prompt 由三部分组成：

1. system message
   - 来自 `planner_system_v05.txt`
2. conversation history
   - 来自 trajectory 已执行消息，包括：
     - 原始 user question
     - 历史 assistant action
     - 历史 tool output
     - 对应可见图片
3. 最后一条 dynamic user control block
   - 来自 `build_planner_control_user_text(...)`

### 3.2 最终 message 顺序

当前顺序是：

1. `role=system`
2. 历史 `user/assistant/tool`
3. 最后一条 `role=user` 的 control block

也就是说，planner 是先看到真实执行历史，再看到一条控制说明。

### 3.3 dynamic control block 结构

当前 control block 在 `offline_sft_pipeline/pipelines/api_text_multimodal.py` 里由下面几块拼接：

- `Round policy`
- `Answer format constraint`
- `Budget constraints`
- `Planning guidance`
- `Capability reference`

其中：

- `Round policy` 由 `_build_planner_round_policy_block(...)` 生成
- `Answer format constraint` 由 `_build_planner_answer_format_block(...)` 生成
- `Budget constraints` 由 `_build_planner_budget_block(...)` 生成
- `Planning guidance` 由 `_build_planner_guidance_block(...)` 生成
- `Capability reference` 由 `_format_capabilities(...)` 生成

### 3.4 当前真实 system prompt 的职责

`planner_system_v05.txt` 负责：

- 定义严格 JSON 输出格式
- 定义 answer / suggestions 两种模式
- 说明 top-level `think`、`suggestion_cot`、`step_goal`、`executor_instruction`、`input_image`、`capability_plan` 的语义
- 给出完整 answer-mode 示例
- 给出完整 suggestions-mode 示例

这次相比旧版本最大的变化是：

- system prompt 里终于有完整 suggestions 示例
- planner 被明确要求服从 `MUST_SUGGEST` / `MAY_ANSWER_OR_SUGGEST` / `MUST_ANSWER`

---

## 4. planner 当前真实 prompt 样例

下面是当前真实 planner prompt 结构的一个缩写样例。它是用测试里的 `PlannerClientRequest` 直接调用 `planner_to_openai_messages(...)` 渲染出来的。

### 4.1 message 0: system

核心内容：

```text
Choose the next best strategy-level action...
OUTPUT FORMAT (STRICT JSON ONLY)
...
Possible policies:
- MUST_SUGGEST
- MAY_ANSWER_OR_SUGGEST
- MUST_ANSWER
...
COMPLETE EXAMPLE: ANSWER MODE
...
COMPLETE EXAMPLE: SUGGESTIONS MODE
...
```

### 4.2 message 1: conversation history

这一条会把历史 user/tool/assistant 消息转换成多模态 OpenAI message。

如果历史消息带图，会被展开成：

- 图片本体
- 一段 geometry text
- 原消息文本

### 4.3 message 2: dynamic control block

在 `MUST_SUGGEST` 的情况下，真实形态类似：

```text
The conversation above already contains the original user question, visible images, prior assistant actions, and tool outputs.

Round policy:
- This round is `MUST_SUGGEST`.
- Return `mode="suggestions"`.
- Do not return any `answer` field.
- The top-level `suggestions` array must contain exactly 3 branch objects.
- Each suggestion must be an executable alternative strategy branch grounded in the evidence above.

Budget constraints:
- `remaining_exec_steps = 3` is the total number of executor steps still available on this trajectory before the final answer.
- Every suggested branch must fit within this remaining budget.
...

Planning guidance:
- The top-level `think` is the round-level Global CoT...
- Use `current` only when a previous-step image already exists...

Capability reference (use only these names in `capability_plan`):
- capability `ocr_assist`
...
- capability `rotate_image`
...
```

这个样例说明两件事：

1. planner 现在确实先吃 system，再吃历史，再吃控制块。
2. 真正控制 planner 是否能答、是否必须给 suggestions 的，不是 system prompt，而是最后这条 dynamic control block。

---

## 5. planner 当前还没完全收口的地方

### 5.1 parser 还保留了旧 XML fallback

`offline_sft_pipeline/pipelines/planner_client.py` 里，当前解析顺序是：

1. 先尝试 JSON
2. 如果失败，再走旧 `<think>/<answer>/<suggestions>` 解析

这说明系统虽然在 prompt 层要求“严格 JSON only”，但解析层并没有完全删掉旧协议。

这带来的问题不是功能错误，而是：

- 调试时错误信息可能仍然带旧 XML 语义
- 系统边界还不够干净

### 5.2 planner 输出里没有对 `requested_suggestion_count` 做 schema 级硬校验

现在 `PlannerOutput` 只校验：

- answer 模式时必须有 `direct_answer`
- suggestions 模式时必须有非空 `suggestions`

但它没有在 schema 层检查：

- `must_suggest` 时 suggestions 数量是否恰好等于请求值

这个数量要求目前主要靠 prompt 控制，不是模型结构校验。

所以：

- 现在策略已经接上
- 但还不是“完全不可跑偏”

---

## 6. offline pipeline 里的 `root/current` 语义

当前 offline pipeline 的 planner 不是直接产出 `image_index`，而是先产出：

- `input_image = "root" | "current"`

之后 orchestrator 在执行 child step 时调用：

- `_select_visible_images(...)`
- `_resolve_runtime_input(...)`

当前可见图像规则是：

- `visible_images[0]` = root image
- 如果已经有上一步主输出图，则把它追加成最后一个 visible image

也就是说，在 offline pipeline 里：

- `root` 是“原始图”
- `current` 是“上一步主输出图”

然后 `_resolve_runtime_input(...)` 再把它编译成当前 runtime step 里的局部 `image_index`：

- `root -> 0`
- `current -> visible_images` 里那张 latest primary image 的索引

接着 executor prompt 会告诉模型：

- 默认 `image/img` 已经绑定到 planner-selected 的输入图
- 如果要切到别的 visible image，再用 helper 的 `image_index=...`

这套逻辑在 offline pipeline 内部是自洽的。

问题不在 offline pipeline 内部。
问题在于它和 CodeVision 在线协议不一致。

---

## 7. CodeVision 在线推理里，`image_index` 到底是谁维护

这个问题必须分三层来看。

### 7.1 第一层：训练样本里的 `_images`

在 `CodeVision/recipe/codevision/uvtr.py`：

- 数据集样本会带 `multi_modal_data["image"]`
- 如果开启自动工具参数生成，还会把原图列表塞进 `tools_kwargs[code_image_tool]["create_kwargs"]["image"]`

也就是说，样本初始图片列表是在 dataset 侧准备好的。

### 7.2 第二层：AsyncRolloutRequest 里的 `multi_modal_data`

在 `CodeVision/verl/workers/rollout/schemas.py`：

- `AsyncRolloutRequest` 持有 `multi_modal_data`
- 初始时，里面放的是该样本当前可见的多模态输入
- 当 tool 返回新图时，`add_tool_response_messages(...)` 会：
  - 把 tool 输出图追加进 `self.multi_modal_data["image"]`
  - 同时把这些图变成新的 tool message，多模态地插进对话

这意味着：

- 模型在后续轮次里确实能“看到”新图
- 在线对话上下文里的图片集合是会增长的

### 7.3 第三层：CodeImageTool 实例自己的 `images`

这是最关键的一层。

在 `CodeVision/verl/tools/code_image_tool.py`：

- `create(...)` 会从 `create_kwargs["image"]` 初始化工具实例的 `images`
- `execute(...)` 会读取 `instance_data["images"]`
- `image_index` 就是在这份 `images` 上做索引

关键问题是：

- 当前 `execute(...)` 返回新图后，并没有把 `processed_image` 追加回 `instance_data["images"]`

所以现在在线系统里其实有两份图片状态：

1. `AsyncRolloutRequest.multi_modal_data["image"]`
   - 会随着 tool outputs 增长
   - 模型能看到
2. `CodeImageTool._instance_dict[instance_id]["images"]`
   - 由 create 时初始化
   - 当前不会因为后续 tool 输出而自动增长
   - `image_index` 真正索引的是它

这就是当前最大的协议错位。

---

## 8. 当前 `image_index` 在 CodeVision 在线里到底表示什么

严格说，当前在线 `image_index` 表示的是：

- “这个 tool instance 在 create 时拿到的那份初始 `images` 列表中的第几个元素”

它不是：

- “整条对话历史里第几张图”
- 也不是 “当前最新可见图”
- 更不是 offline pipeline 里的 `current`

所以如果你问：

“在线推理时有没有一个持续增长的图片数据库，让 `image_index` 指向全局图片时间线？”

当前答案是：

- 对模型可见的多模态历史有增长列表
- 但对 `code_image_tool.execute()` 来说，没有同一份会增长的工具内图片库

因此现在的 `image_index` 语义是不完整的。

---

## 9. 这为什么会直接卡死 `root/current` 导出

offline pipeline 的 executor 其实依赖一个很关键的前提：

- backend 会先根据 planner 选的 `root/current`，把正确的图绑定到默认 `image/img`

这样 executor code 才能直接写：

```python
result = image.rotate(270, expand=True)
```

而不用每次显式写：

```python
result = images[0].rotate(...)
```

但是在 CodeVision 在线协议里，当前只暴露了：

- `image_index`

没有暴露：

- “这一步默认绑定的是 root 还是 current”
- “这一步绑定的是哪个 artifact”

所以如果你只把 offline 轨迹导成：

```json
{
  "code": "...",
  "description": "...",
  "image_index": 0
}
```

那么训练样本里可能还能勉强成立，
但在线推理时会有两个根本问题：

1. 模型怎么知道 `0` 是 root，还是别的历史图？
2. 如果当前 step 需要“切回 root 作为默认活动图”，后台并没有一个显式协议告诉 runtime 去做这个绑定动作。

这正是你问的核心。

---

## 10. 只靠静态 SFT 的全局图片索引，为什么不够

理论上，离线导出时你可以把一条固定轨迹编译成：

- root 图先放进全局 images 列表
- 每一步 tool 输出图按时间顺序 append
- assistant tool_call.arguments.image_index 指向某个全局 index

对静态 SFT 数据来说，这当然可以做。

但在线推理不是静态回放，它有两个本质不同点：

1. 未来生成的图在一开始并不存在
2. 当前 runtime 的 `CodeImageTool` 不是按“全局 append-only 图库”在工作

所以：

- 静态 SFT 可以把 `root/current` 编译成 index
- 在线 runtime 目前不能天然复现这个语义

换句话说：

**SFT 数据能拼对，不代表真实 runtime 就实现了同一个协议。**

---

## 11. 现在 executor 的 CoT 和 code 为什么会错位

offline executor 当前的设计是：

- `think` 可以说“我要回到 root 图重新看”
- 代码部分仍然直接从默认绑定的 `image` 开始写

这在 offline runtime 是对的，因为 orchestrator 先做了图绑定。

但在 CodeVision 在线 runtime：

- tool_call 里没有显式 `input_image=root/current`
- runtime 不知道这次该把哪张图绑定成默认 `image`
- 它只会根据 `image_index` 从固定 `instance.images` 里取图

所以如果只导出原有 `think + code + image_index`：

- `think` 说的是 symbolic source
- `code` 依赖的是 pre-bound default image
- 在线 runtime 执行的是 fixed image_index semantics

这三件事当前不是同一个协议。

---

## 12. 我的结论：需要补一个新的结构化字段

结论先说：

**是的，我认为需要。**

而且最好不是只补一个 `root/current` 的软文案，而是补一个真正进 tool-call arguments 的结构化字段。

### 12.1 为什么只靠 `image_index` 不够

因为 `image_index` 只表达了：

- “选第几张”

它没有表达：

- “为什么选这张”
- “这是 root 还是 current”
- “这是 symbolic source 还是 concrete global index”
- “这一步默认绑定后的 `image/img` 应该是什么”

### 12.2 推荐的最小新字段

我建议给 `code_image_tool` 新增一个可选结构位，例如：

```json
{
  "code": "...",
  "description": "...",
  "image_index": 0,
  "input_image": "root"
}
```

或者更稳一点：

```json
{
  "code": "...",
  "description": "...",
  "image_index": 0,
  "input_binding": {
    "source": "root"
  }
}
```

如果想把 offline artifact 语义也保住，我更推荐：

```json
{
  "code": "...",
  "description": "...",
  "image_index": 0,
  "input_binding": {
    "source": "artifact",
    "artifact_id": "img_root_0",
    "fallback": "root"
  }
}
```

---

## 13. 我推荐的协议优先级

在线 runtime 可以按这个顺序解析：

1. 如果有 `input_binding.artifact_id`
   - 优先按 artifact 绑定
2. 否则如果有 `input_image`
   - `root` -> 绑定原始 root 图
   - `current` -> 绑定最新 step output 图
3. 否则才回退到 legacy `image_index`

然后 runtime 在真正执行 code 前做两件事：

1. 把解析出来的图绑定到默认 `image/img/draw`
2. 把最终解析得到的 concrete index 记录进 metrics 或 observation

这样：

- `think` 可以明确说“回到 root 图”
- `code` 仍然可以自然地从 `image` 开始写
- runtime 也知道默认活动图该怎么切

这才是协议闭环。

---

## 14. 仅靠后台“自动切 root 到 img”可以吗

可以，但前提是这个行为要有协议依据。

也就是说：

- 如果模型只是随便在 CoT 里说“我要看 root”
- 但 tool_call arguments 里没有任何结构化字段
- 那 runtime 不应该偷猜

否则会导致：

- 训练协议不可验证
- runtime 行为不可重现
- 日后 debug 很难知道模型到底选了 root，还是 backend 自作主张改了图

所以如果你想要“后台自动切 root，再让 code 从 `img` 开始写”，我的判断是：

**可以做，但必须有显式结构字段来驱动，而不是靠读 CoT。**

---

## 15. 还需要补一个 runtime 状态修复

即使加了新字段，我仍然建议同时修一件事：

### 15.1 让 CodeImageTool 实例图片池变成 append-only

当前 `CodeImageTool.execute(...)` 成功后没有把 `processed_image` 追加回 `instance_data["images"]`。

建议改成：

- root image(s) 先进入 instance image pool
- 每次 tool 成功执行后，把新图 append 到该 pool
- 同时维护：
  - `root_image_indices`
  - `current_image_index`
  - 可选 `artifact_id -> index`

这样做的价值是：

- helper 里 `image_index=` 能真的 revisit 历史图
- 在线 runtime 更接近静态 SFT 的全局图库语义
- 即使未来仍然保留 `image_index`，它也终于有稳定时间线意义

### 15.2 但这一步不能替代新字段

即便 image pool 变成 append-only，也仍然建议保留 `input_image` / `input_binding`。

因为：

- append-only pool 解决的是“历史图能不能被引用”
- 新结构字段解决的是“默认活动图该绑定谁”

这是两层不同的问题。

---

## 16. 推荐的最终设计

如果目标是把 offline pipeline 和 CodeVision 在线协议打通，我建议最终落成下面这个形态。

### 16.1 offline planner / step record 保留 symbolic 语义

继续保留：

- `input_image = root/current`
- `input_artifact_id`

### 16.2 exporter 导出 assistant tool_call 时同时写两层信息

建议导出成：

```json
{
  "name": "code_image_tool",
  "arguments": {
    "code": "...",
    "description": "...",
    "image_index": 0,
    "input_image": "root",
    "input_artifact_id": "img_root_0"
  }
}
```

其中：

- `image_index` 是为兼容旧协议的 concrete fallback
- `input_image` / `input_artifact_id` 才是主语义

### 16.3 CodeVision runtime 做 source resolution

在 `CodeImageTool.execute(...)` 前：

- 根据 `input_artifact_id` / `input_image` 选图
- 绑定默认 `image/img`
- 再执行 code

### 16.4 executor 的 CoT 要显式提 source switch

如果这一步是从 `current` 回到 `root`，建议 `think` 明确说：

- 当前 crop 已经丢失全局上下文
- 需要回到原始图重新定位目标

这不是为了让 backend 读 CoT 做判断，而是为了让 SFT 轨迹本身自洽。

---

## 17. 最终回答

### 问题 1：现在 planner 的逻辑完全成功了吗？

回答：

- 逻辑链已经接上了，真实 round policy 也已经接上了。
- 当前 planner 的真实行为已经不是旧模板拼接，而是：
  - `planner_system_v05.txt`
  - 历史 conversation
  - dynamic control block
- 但还不能说“完全成功”，因为：
  - parser 还保留旧 XML fallback
  - `requested_suggestion_count` 还没做严格结构校验
  - 更关键的是 planner 产出的 `root/current` 语义还没有完整映射到 CodeVision 在线 tool 协议

### 问题 2：CodeVision 在线推理时 image_index 怎么安排？

回答：

- 当前 `image_index` 不是全局对话图片库索引。
- 它实际索引的是 `CodeImageTool.create(...)` 时初始化进去的那份 `instance.images`。
- 在线对话里新增的 tool output 图片会进入 `AsyncRolloutRequest.multi_modal_data["image"]`，模型能看到，但 `code_image_tool` 当前不会自动把这些图加进自己的 `instance.images`。
- 所以现在在线系统里，模型可见图历史和 tool 可索引图历史不是一回事。

### 问题 3：如果要从 current 切回 root，是不是要新字段？

回答：

- 我认为是。
- 最小可行方案是给 tool_call arguments 增加 `input_image: "root" | "current"`。
- 更稳方案是增加：
  - `input_image`
  - `input_artifact_id`
- runtime 先解析这个字段，再把默认 `image/img` 绑定好，代码就仍然可以自然地从 `image` 开始写。

这一步如果不做，offline pipeline 的 symbolic source 语义和 CodeVision 在线 runtime 的 concrete image_index 语义会长期错位。

