# Offline 多轮 SFT 数据生成 Pipeline 设计文档

文档版本: v0.2  
日期: 2026-03-26  
适用目录: `D:\sdu\ToolVision`  
状态: 当前对齐版，可直接指导后续分工和实现

---

## 1. 目标

本项目要做的是一个 offline、多轮、分支式的多模态 SFT 数据生成 pipeline。

核心目标：

1. 在现有 `CodeVision` 的 `CodeImageTool` 上继续扩展能力函数。
2. 用强模型自动生成多轮 tool-use 轨迹。
3. 用 judge 对轨迹筛选、排序和打标。
4. 最终导出可供 `LLaMA-Factory` 训练的 SFT 数据。

一句话总结：

> 生成阶段保留完整轨迹，训练阶段导出线性对话样本。

---

## 2. 当前已经确认的决定

### 2.1 训练和生成的职责分离

- SFT 训练端使用 `LLaMA-Factory`。
- `verl` / `CodeVision` 当前的 tool framework 不作为训练入口。
- 生成端需要复用 `CodeImageTool` 的 sandbox 和 helper 执行能力。

也就是说：

- 训练靠 `LLaMA-Factory`
- 生成靠我们新建的 offline pipeline
- tool 执行底座继续复用 `CodeVision`

### 2.2 需要保留 planner

- 不能完全去掉 planner。
- 原因不是为了让 planner 决定每一步代码细节，而是为了保留一个全局规划层。
- 最终训练样本需要一个总的全步骤 CoT，放在 step 0。

这样做的好处：

1. 避免模型只盯着显眼强工具，比如 OCR。
2. 保留“先全局拆解，再分步执行”的行为。
3. 提升模型主动使用辅助工具的可能性，比如增强、旋转、局部 crop。

### 2.3 planner 要尽量把路线先定掉

- planner 不只是给“允许能力列表”。
- planner 应尽量提前把路线分歧解决掉，减少 executor 的歧义空间。

例如：

- 不推荐：`["detect", "crop", "box"]`
- 推荐：
  - `["detect", "crop"]`
  - 或 `["detect", "box"]`

也就是说：

- 路线选择前移到 planner
- executor 主要负责把已定路线写成 CoT 和代码

### 2.4 一个 step 可以多次 helper 调用

- 一个 step 的代码块内允许多次 helper 调用。
- 这通常是必要的，因为很多有效视觉步骤本身就是组合操作。

例如：

- 先增强对比度，再 OCR
- 先 detect，再 crop
- 先 detect，再 depth compare

### 2.5 不再增加过多派生字段

本轮明确去掉以下设计：

- `expected_output_type`
- `primary_image_required`
- `step_success_signal`

原因：

1. 这些字段和真实执行容易重复描述。
2. 会让 planner schema 变得过重。
3. judge 是否成功，更适合交给 judge model 直接判断。
4. image-to-image 约束可以写进 planner prompt 的全局规则，而不必变成额外字段。

### 2.6 ToolResponse 尽量贴近现有 CodeVision

- 不希望大幅改动现有 `ToolResponse` 的概念。
- offline pipeline 里也尽量沿用：
  - `image`
  - `text`
  - `meta`

这里的 `text` 不要求是工具自述性的总结。

它更现实的来源通常是：

- OCR 文本
- OCR tokens
- 检测框数量
- 简短状态信息
- 结构化格式化内容

### 2.7 多图返回允许，但不强制合图

- 每一步必须是 image-to-image。
- 但不强制每一步只能返回一张图。
- 也不强制为了“主图”概念去拼接或合并图片。

如果 planner 觉得：

- 原图上框出两个目标最好，就返回一张原图加框图
- 两个局部 crop 更好，就允许返回多张图

这部分自由交给 planner。

### 2.8 V1 历史先按全量

- planner / executor / judge 在 V1 里默认看全部历史消息和全部中间图。
- 当前预期一条链通常 4 到 5 步，先不做激进压缩。

### 2.9 新 helper 命名按能力

helper 命名原则：

- 按能力命名
- 不按后端模型命名

当前优先能力：

- `depth`
- `count`

后续也尽量走这种抽象。

### 2.10 judge 的高层目标

judge 的关键目标不是只判断“看起来合理”。

更重要的是：

- 原本只有强模型能做对
- 引入这条工具轨迹后
- 更弱的模型是否也更容易做对

这说明轨迹确实提升了“可理解性”。

### 2.11 全部轨迹都保留

- 所有轨迹都落盘保存。
- 但不是所有轨迹都继续扩展。
- 需要区分：
  - `保存`
  - `进入下一轮 frontier`

### 2.12 V1 导出策略先支持全部终止轨迹

当前阶段 exporter 先支持导出全部终止轨迹，而不是只导出成功轨迹。

原因：

1. 某些样本可能一条成功轨迹都没有。
2. 如果只导成功轨迹，可能直接导空。

因此 V1 建议：

- 先完整导出全部终止轨迹
- 打清楚状态、judge score 和标签
- 再在后处理阶段筛选高质量子集

长期目标仍然是：

- 优先使用高质量成功轨迹做 SFT

### 2.13 planner 和 executor 是交替循环，不是一次性总规划

这是本项目的关键机制。

- planner 不是只在 step 0 跑一次。
- 更合理的流程是：
  - planner 先基于当前轨迹提出 2 到 3 条候选后续路线
  - executor 只执行当前被选中的第一步
  - 新产生的图和文本再次进入 trajectory
  - planner 再基于更新后的 trajectory 重新规划剩余路线

因此：

1. step 0 的全局 CoT 是“初始总计划”
2. 但后续路线不是锁死的
3. 每一轮 planner 都允许根据新图和新信息改写后续步骤
4. 直到：
   - planner 判断可以直接回答
   - 或命中错误 / 预算 / 步数上限等停止条件

这是一种 rolling replanning，而不是 rigid script execution。

---

## 3. 当前仓库中的真实基线

### 3.1 当前多轮格式已经存在

当前 `CodeVision` 里已经有多轮 tool-use 格式：

1. `assistant` 输出 `<tool_call>...</tool_call>`
2. `tool` 回一条消息
3. 新图像被追加到当前对话可见图像列表
4. 模型继续下一轮推理

对应实现主要在：

- `CodeVision/recipe/codevision/config/grpo_trainer.yaml`
- `CodeVision/verl/experimental/agent_loop/tool_agent_loop.py`
- `CodeVision/verl/utils/dataset/multiturn_sft_dataset.py`

### 3.2 当前 `CodeImageTool` 是关键底座

当前 `CodeImageTool` 已具备：

1. 代码安全校验
2. 安全执行环境
3. 图像输入注入
4. helper 注入
5. 外部模型 worker 调用
6. 统一的 `ToolResponse`

这意味着：

- 它很适合复用为 offline executor 的执行底座
- 但不意味着应该直接复用现有 `ToolAgentLoop`

### 3.3 当前外部能力还不完整

当前真正接好的外部能力主要是：

- `paddleocr_vl`
- `grounded_sam2`

GroundedSAM2 当前通过 `_operation` 支持：

- `box`
- `mask`
- `dino_crop`
- `blur_bg`

也就是说，当前还没有一套真正通用的能力级接口：

- `detect`
- `segment`
- `depth`
- `count`

所以后续确实需要补能力层抽象。

### 3.4 当前 tool 文本返回不够 rich

现有在线 `CodeImageTool.execute()` 返回给模型的文字部分比较固定，不足以承担我们 offline 数据生成里的中间语义。

因此 offline pipeline 需要做一层包装：

- 复用 `CodeImageTool`
- 但让运行结果更适合 trajectory 保存和 judge

不过这层包装要尽量轻，不要重新设计一套和现有 pipeline 差别很大的 tool 协议。

---

## 4. 为什么不直接复用现有 ToolAgentLoop

现有 `ToolAgentLoop` 的目标是在线 rollout：

- 单条线性轨迹
- 模型即时生成
- tool 即时执行
- 继续下一轮

而我们现在要做的是：

- planner 先给出多个候选链
- orchestrator 分叉 child trajectories
- judge 离线打分
- frontier 管理
- 最终只导出线性训练样本

所以推荐策略是：

- 复用 `CodeImageTool`
- 新建独立 offline pipeline

而不是把 branching 搜索硬塞进现有 rollout。

---

## 5. 系统总览

建议把系统拆成三层。

### 5.1 能力层

职责：

- 在 `CodeImageTool` 中继续增加 helper
- 屏蔽底层模型差异
- 统一返回 `image / text / meta`

### 5.2 生成编排层

职责：

- trajectory 管理
- planner 调用
- 分叉
- executor 调用
- tool runtime 执行
- judge 打分
- frontier 更新
- 断点恢复

### 5.3 导出层

职责：

- 从轨迹中导出线性多轮训练样本
- 对齐 `LLaMA-Factory` 所需格式

---

## 6. 建议模块

### 6.1 planner

职责：

- 读取当前 trajectory 完整历史
- 判断当前是否可以直接回答
- 如果不能，输出 2 到 3 条候选完整链条

planner 不写代码。

补充：

- planner 不是只在 step 0 调一次。
- 它应在每一轮 executor 执行完成后重新读取最新 trajectory，再规划接下来的候选路线。
- 因此 planner 输出的是“从当前状态继续往后走”的候选路线，而不是一次性锁死未来所有步骤。

### 6.2 executor

职责：

- 读取完整历史
- 读取选中 suggestion
- 只处理当前 step
- 输出：
  - 当前 step 局部 CoT
  - 当前 step 可执行代码

补充：

- executor 不负责把一条 suggestion 一路机械执行到底。
- 它只负责把“当前轮 planner 选中的下一步”真正写成代码并执行。
- 下一步是否继续沿原路线，交给下一轮 planner 再判断。

### 6.3 sandbox runtime

职责：

- 复用 `CodeImageTool`
- 运行 executor 生成的代码
- 生成 tool 返回
- 保存中间 artifact

注意：

- 这里尽量贴近现有 `ToolResponse`
- 不在 runtime 层发明复杂的新语义字段

### 6.4 judge

职责：

- 离线打分
- 不进入主对话 messages

### 6.5 orchestrator

职责：

- 管理 active / parked / answered / failed trajectories
- 控制预算
- 调度 planner / executor / runtime / judge

### 6.6 exporter

职责：

- 把轨迹线性化
- 转成 `LLaMA-Factory` 可用的训练样本

---

## 7. 生成态与训练态分离

### 7.1 生成态

生成态需要支持：

- branching
- artifact 保存
- judge
- 恢复
- 回放

因此生成态必须保留完整 trajectory 记录。

### 7.2 训练态

训练态不需要知道：

- 所有被淘汰的分支
- 全量 judge 日志
- 全量 planner 备选链
- debug 信息

训练态只需要：

- 一条线性的多轮消息
- 对应图像路径
- 必要 metadata

---

## 8. planner 设计

### 8.1 为什么一定保留 planner

如果完全去掉 planner，会出现两个问题：

1. 很难自然地产生 step 0 的全局 CoT。
2. executor 会更容易局部最优，只盯着显眼工具，不利于形成更好的 tool-use 习惯。

### 8.2 planner 在训练态如何体现

建议：

1. planner 完整输出不直接进入训练消息。
2. 只把 planner 的全局链路思考压成 step 0 的 assistant thinking。
3. suggestions、候选链、judge 细节都保存在 trajectory 侧。

### 8.3 正确的运行方式是 rolling replanning

这里必须明确：

- step 0 的 planner 输出只是初始总计划
- 它不是一条执行到底、永不修改的固定脚本

正确循环应该是：

1. planner 读取当前 trajectory
2. 如果还不能回答，就给出当前状态下的候选后续路线
3. orchestrator 选择或分叉这些候选路线
4. executor 只执行当前被选路线的第一步
5. tool 返回的新图和文本写回 trajectory
6. planner 再基于更新后的 trajectory 重新规划

也就是说：

- 已执行前缀是固定历史
- 未执行后缀是可被 planner 重写的

这点对视觉任务尤其重要，因为新图和新文本会显著改变后续最优路线。

---

## 9. 历史可见性策略

V1 默认：

- planner 看全部历史消息和全部历史图
- executor 看全部历史消息和全部历史图
- judge 也能访问全部轨迹产物

先不做 selective replay。

后续如果上下文压力显著，再加压缩。

---

## 10. step 设计原则

一个 step 不是单个 primitive tool，而是一轮有明确子目标的视觉操作。

### 10.1 每个 step 至少包含什么

推荐每个 step 至少包含：

1. `step_id`
2. `step_goal`
3. `capability_plan`
4. `executor_instruction`

### 10.2 `capability_plan` 的含义

`capability_plan` 不是模糊的能力白名单。

它应尽量是明确、少歧义、已定好的路线。

例如推荐：

- `["detect", "crop"]`
- `["detect", "box"]`
- `["enhance_contrast", "ocr"]`
- `["detect", "depth_compare"]`
- `["detect", "count"]`

不推荐：

- `["detect", "crop", "box"]`

因为这会把路线选择再次留给 executor。

### 10.3 image-to-image 约束如何表达

不通过额外字段表达。

而是通过 planner 的总规则约束：

- 每一个 step 都必须是 image-to-image
- 每一步都应产出新的图像结果，供后续继续推理

---

## 11. 多 helper 调用策略

一个 step 允许多次 helper 调用。

例如：

- 先增强，再 OCR
- 先 detect，再 crop
- 先 crop，再 count

但不建议一个 step 同时承担整条链的全部逻辑。

如果一个 step 同时完成：

- 定位
- 消歧
- 比较
- 最终回答

那就过粗了。

---

## 12. 多图返回策略

### 12.1 核心原则

- 每一步必须是 image-to-image。
- 但不强制每一步只能返回一张图。
- 也不强制合并为主图。

planner 可以自由决定：

1. 返回原图加框
2. 返回一个 crop
3. 返回多张局部图

### 12.2 为什么不强制合图

因为如果问题是比较两个物体，planner 有时会认为：

- 原图双框最好
- 两个 crop 更好
- 一个原图加一个局部图也更好

这个选择不应该被 runtime 强行统一。

### 12.3 V1 建议

V1 默认：

- 允许一步返回一张或多张图
- 返回多少张图，由 planner 路线决定
- 全部返回图都保存
- 对话层先按实际返回注入

如果后续验证发现多图显著降低稳定性，再收紧。

---

## 13. Tool return 设计

### 13.1 总原则

offline pipeline 的 tool return 设计尽量贴近现有 CodeVision。

建议保留核心结构：

- `image` 或 `images`
- `text`
- `meta`

### 13.2 `text` 的定位

`text` 不强求是自然语言总结。

更现实的情况是：

- OCR 文本
- OCR tokens
- 结构化 JSON 串
- 简短状态文本

如果没有有价值的文字，也可以为空。

### 13.3 `meta` 的定位

`meta` 主要服务：

- judge
- debug
- export
- 断点恢复

默认不直接进入训练对话。

### 13.4 推荐的 runtime 记录形态

V1 推荐类似：

```json
{
  "success": true,
  "images": [
    {"artifact_id": "img_0", "path": "step_002/output_0.png"},
    {"artifact_id": "img_1", "path": "step_002/output_1.png"}
  ],
  "text": "optional formatted text",
  "meta": {
    "ocr_tokens": [],
    "annotations": [],
    "crop_boxes": []
  },
  "error": null
}
```

这里和现有 ToolResponse 的区别尽量小。

### 13.5 工具调用链 metadata 是否进入对话

默认不进入。

原因：

1. assistant 的 CoT 和 code 已经隐含了调用链。
2. trajectory 侧也会保留 step 记录。
3. 过多 metadata 进入对话会增加噪声。

---

## 14. 图片标识策略

内部不要只靠 `image_index` 做长期标识。

推荐：

- 内部统一用 `artifact_id`
- `image_index` 只作为运行时映射

也就是说：

- offline pipeline 内部维护 `artifact_id`
- executor 真正调用 runtime 时，再编译为临时 `image_index`

---

## 15. planner 输出规范

### 15.1 planner 顶层输出

```json
{
  "can_answer_now": false,
  "global_chain_cot": "先解决人物指代，再解决车辆链条，最后比较远近。",
  "suggestions": []
}
```

### 15.2 suggestion schema 设计原则

不再使用：

- `expected_output_type`
- `primary_image_required`
- `step_success_signal`

而是尽量把 step 约束写在 `capability_plan` 和 `executor_instruction` 里。

### 15.3 suggestion 示例

```json
{
  "suggestion_id": "A",
  "chain_cot": "先做人物链条，再做车辆链条，最后比较深度。",
  "chain": [
    {
      "step_id": "A_1",
      "step_goal": "定位拿伞男人及其后方孩子，并返回适合下一步判断的人物图像结果",
      "capability_plan": ["detect", "crop"],
      "executor_instruction": "这一步固定采用 detect 后 crop 的路线，不要改成 box；代码里应先定位 man、umbrella、child，再基于坐标返回人物局部图。"
    },
    {
      "step_id": "A_2",
      "step_goal": "进一步消除目标孩子歧义",
      "capability_plan": ["segment"],
      "executor_instruction": "返回能帮助判断人物关系的图像结果，不要改回纯 crop。"
    }
  ]
}
```

### 15.4 planner 规则

planner 必须满足：

1. 每条 suggestion 是完整路线。
2. 每个 step 都是 image-to-image。
3. 每个 step 的路线要尽量明确。
4. 不写 fake function name。
5. 不写具体代码。

---

## 16. executor 输出规范

### 16.1 executor 输出内容

executor 只输出两部分：

1. 当前 step 的局部 CoT
2. 当前 step 的 Python code

### 16.2 executor 规则

- 只执行 planner 已确定的路线
- 允许多 helper
- 不重做全局规划
- 非 final answer step 不直接回答最终问题

### 16.3 executor 编码建议

建议代码里显式体现：

- 当前 step 目标
- 当前采用的能力路线
- 如果返回多图，图像顺序是什么

例如：

```python
# step_goal: isolate the bicycle next to the red car
# capability_plan: detect -> crop
det = _call_detect("red car. bicycle.")
crop_res = _call_crop(...)
result = crop_res["image"]
```

---

## 17. step 0 总 CoT

### 17.1 作用

step 0 用于保留：

- 问题拆解
- 全局规划
- 路线选择原则

### 17.2 来源

推荐来源：

- planner 的 `global_chain_cot`
- 经过压缩改写后导入最终第一条 assistant 消息

### 17.3 不放什么

step 0 不应直接包含：

- 全部候选 suggestions
- judge 信息
- prompt engineering 痕迹

CoT 长度上限本轮暂不定死，后续再议。

---

## 18. judge 设计

judge 不进入主对话 messages。

### 18.1 judge 的三层目标

#### 第 1 层：执行有效性

看：

- 代码是否跑通
- 输出图是否有效
- 是否不是空操作或明显退化操作

#### 第 2 层：轨迹质量

看：

- 当前 step 是否提升了 answerability
- 是否减少歧义
- 是否让目标更清楚

#### 第 3 层：弱模型迁移收益

看：

- 原本强模型可做
- 这条轨迹是否让更弱模型也更容易做对

### 18.2 judge 实现建议

建议三段式：

1. cheap filter
2. medium judge
3. committee judge

### 18.3 committee judge 记录什么

至少记录：

- 哪些模型做对了
- 弱模型 solve count
- 简短 note

---

## 19. planner-executor 交替主循环

这一节是本项目的真实主循环。

### 19.1 root 初始化

1. 创建 root trajectory
2. 写入原始问题和原图
3. 初始化 messages

### 19.2 第 0 轮 planner

1. planner 读取 root trajectory
2. 输出：
   - 是否可以直接回答
   - step 0 全局 CoT
   - 当前状态下的 2 到 3 条候选后续路线

### 19.3 分叉与执行

对于每条保留的 suggestion：

1. orchestrator 创建 child trajectory
2. executor 只执行该 suggestion 的第一步
3. runtime 执行代码
4. 新图 / text / meta 追加到 trajectory

### 19.4 再次进入 planner

某条 child trajectory 执行完一个 step 后，不是继续机械执行旧链条的下一步，而是：

1. 重新把完整最新 trajectory 送回 planner
2. planner 基于新图、新 text、新历史，重新给出当前状态下的候选后续路线
3. orchestrator 再决定如何继续扩展

### 19.5 停止条件

满足任一条件时停止：

1. planner 判断当前已经可以直接回答
2. executor 或 runtime 连续错误
3. 命中步数上限
4. 命中预算上限
5. judge 判定该分支不值得继续

### 19.6 这个循环意味着什么

它意味着：

1. planner 是逐轮调用的
2. executor 每次只绑定当前一步
3. 未来路线永远是可修正的
4. trajectory 中保存的是“真实执行历史”，而不是一条永不变化的预定脚本

---

## 20. “全部保留”和“继续扩展”分开

### 19.1 全部保留的含义

全部保留指：

- 所有 trajectory 都落盘
- 所有 trajectory 都可回看
- 所有 trajectory 都可参与后续统计和导出

### 19.2 不等于全部继续展开

如果所有轨迹每轮都继续展开，分支会爆炸。

### 19.3 推荐状态

建议状态包括：

- `running`
- `answered`
- `failed`
- `parked`
- `budget_exhausted`

其中：

- `parked` 表示保留但当前不扩展

---

## 21. trajectory 数据结构

### 20.1 trajectory 顶层建议字段

```json
{
  "sample_id": "gqa_000123",
  "trajectory_id": "gqa_000123__A_r1",
  "parent_trajectory_id": "gqa_000123__root",
  "status": "running",
  "round_idx": 1,
  "step_idx": 1,
  "question": "...",
  "original_image_artifact_id": "orig_0",
  "messages_path": "messages.json",
  "planner_history": [],
  "latest_planner_round_idx": 1,
  "latest_planner_output_path": "round_001/planner_output.json",
  "fork_provenance": {
    "parent_planner_round_idx": 0,
    "parent_suggestion_id": "A"
  },
  "pending_execution": null,
  "steps": [],
  "judge_records": [],
  "final_answer": null,
  "answer_confidence": null,
  "budget": {
    "remaining_rounds": 3,
    "remaining_children": 8
  },
  "last_error": null
}
```

### 20.2 messages 单独存文件

建议：

- `trajectory.json` 做索引
- `messages.json` 单独保存
- 每个 step 有独立目录

### 20.3 不再用 `active_chain` 表示未来固定路线

这里需要明确：

- 不建议再用 `active_chain_path` / `active_chain_position` 这类字段。
- 因为它们很容易误导实现，仿佛一条 trajectory 会把未来整条链锁死。
- 本项目里，trajectory 只保存：
  - 已经真实执行过的历史
  - 最近一轮 planner 的输出快照
  - 这条 child trajectory 是从哪一轮、哪条 suggestion 分叉出来的 provenance

如果需要表示“当前将要执行什么”，建议只保留一个很轻的 `pending_execution`：

- 它只表示当前轮已经选中、但尚未执行的那个 step。
- 一旦该 step 执行完成，`pending_execution` 应清空。
- 下一步是否继续沿原 suggestion，必须重新跑 planner 再决定。

---

## 22. step 数据结构

建议每个 step 至少记录：

```json
{
  "step_idx": 2,
  "planner_round_idx": 1,
  "suggestion_id": "A",
  "suggestion_step_index": 1,
  "step_id": "A_2",
  "step_goal": "消除目标孩子歧义",
  "capability_plan": ["segment"],
  "executor_cot_path": "step_002/executor_cot.txt",
  "executor_code_path": "step_002/code.py",
  "runtime_result_path": "step_002/runtime_result.json",
  "assistant_message_id": "msg_a_002",
  "tool_message_id": "msg_t_002"
}
```

这里的 `planner_round_idx + suggestion_id + suggestion_step_index` 只是 provenance。

- 它表示这个已执行 step 最初来自哪一轮 planner 的哪条 suggestion。
- 它不表示后续步骤必须继续沿同一条 suggestion 机械执行。
- 后续路线永远允许在下一轮 planner 中被重写。

---

## 23. artifact 数据结构

建议图像、文本、结构化结果都走 artifact 管理。

### 22.1 示例

```json
{
  "artifact_id": "img_step002_out0",
  "type": "image",
  "path": "step_002/output_0.png",
  "producer": "executor_runtime",
  "source_step_idx": 2,
  "output_index": 0
}
```

### 22.2 为什么不用 URL 做主键

V1 优先使用：

- 本地路径
- shard 内相对路径

内部主标识统一用 `artifact_id`。

---

## 24. messages 规范

### 23.1 生成态消息顺序

建议顺序：

1. `system`
2. `user`
3. `assistant` step 0 全局 CoT
4. `assistant` step 1 局部 CoT + tool call
5. `tool` step 1 返回
6. `assistant` step 2 局部 CoT + tool call
7. `tool` step 2 返回
8. ...
9. `assistant` 最终 `<answer>`

### 23.2 step 0 是否单独占一条 assistant

建议是。

### 23.3 tool message 长什么样

tool message 可包含：

- 一张或多张图
- 可选文本

例如：

```json
{
  "role": "tool",
  "content": [
    {"type": "image", "path": "step_001/output_0.png"},
    {"type": "image", "path": "step_001/output_1.png"},
    {"type": "text", "text": "{\"ocr_tokens\": [...]}"} 
  ]
}
```

### 23.4 tool message 的 text 是否必需

不是必需。

如果没有有效文字信息，可以只返回图像。

如果有文字，优先保留：

- OCR 文本
- 结构化格式化信息
- 简短状态文本

不要求工具必须自述“我完成了什么”。

---

## 25. 导出到 LLaMA-Factory

### 24.1 总原则

不要直接把生成态 trajectory 原样喂给训练。

应先导出为一份 canonical SFT JSONL，再转成 `LLaMA-Factory` 需要的格式。

### 24.2 canonical 样本示意

```json
{
  "sample_id": "gqa_000123",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>"},
    {"role": "assistant", "content": "<think>...</think><tool_call>...</tool_call>"},
    {"role": "tool", "content": [
      {"type": "image", "path": "step_001/output_0.png"},
      {"type": "text", "text": "...."}
    ]},
    {"role": "assistant", "content": "<think>...</think><answer>yes</answer>"}
  ],
  "tools": [...],
  "metadata": {
    "trajectory_id": "gqa_000123__A1_r3",
    "status": "answered",
    "judge_score": 0.87
  }
}
```

### 24.3 当前导出策略

V1 exporter 先支持：

- 导出全部终止轨迹

后处理时再按：

- `status`
- `judge_score`
- `weak_model_solve_count`

做筛选。

---

## 26. 新 helper 扩展原则

### 25.1 命名按能力

推荐：

- `_call_detect(...)`
- `_call_segment(...)`
- `_call_depth(...)`
- `_call_count(...)`
- `_call_ocr(...)`

不推荐：

- `_call_groundedsam2_box_v2(...)`
- `_call_xxx_model(...)`

### 25.2 返回结构统一

所有 helper 尽量统一返回：

```python
{
    "image": PIL.Image,
    "images": list[PIL.Image],
    "text": str,
    "meta": dict,
}
```

### 25.3 保留 active image 更新机制

当前 `CodeImageTool` helper 会更新 `image / img / draw`。

这个机制建议继续保留，因为它有利于组合式 step 编码。

---

## 27. 推荐目录结构

建议新增独立目录，例如：

```text
offline_sft_pipeline/
├── configs/
├── prompts/
├── schemas/
├── core/
├── runtime/
├── pipelines/
├── outputs/
└── scripts/
```

底层工具能力继续留在 `CodeVision` 里扩展。

---

## 28. 分工建议

### 27.1 你负责

- trajectory schema
- orchestrator
- runtime wrapper
- export
- 断点恢复
- frontier / budget 逻辑

### 27.2 同学负责

- planner / executor / judge prompt
- judge model committee
- 外部模型部署
- helper 对应服务接口

### 27.3 需要共同拍板的接口

1. helper 能力命名
2. planner 输出格式
3. runtime return 格式
4. judge record 格式
5. export sample 格式

---

## 29. 开发阶段建议

### 阶段 A: 扩 helper

先接好：

- `depth`
- `count`

并为每个 helper 做最小 demo。

### 阶段 B: 做 runtime wrapper

目标：

- 复用 `CodeImageTool`
- 跑单 step code
- 保存 images / text / meta

### 阶段 C: 手工 smoke

先不接完整 planner。

用手工给定链条跑通：

- 2 到 3 步
- trajectory 落盘
- artifact 保存
- messages 回放

### 阶段 D: planner / executor 接通

目标：

- root trajectory 自动分叉
- executor 自动执行当前 step

### 阶段 E: judge 接通

目标：

- cheap filter
- committee judge

### 阶段 F: exporter 接通

目标：

- 导出 canonical SFT JSONL
- 再转换成 `LLaMA-Factory` 数据格式

---

## 30. 当前仍未完全定死的点

### 29.1 `depth / count` 的精确定义

例如：

- `depth` 到底返回深度图、比较文本，还是都返回
- `count` 返回计数数字、标注图，还是都返回

### 29.2 弱模型 judge 的统计口径

例如：

- 至少多少个弱模型从错变对才算有效
- 是否区分 model family
- 是否按 solve count / solve rate / uplift 打分

### 29.3 step 0 CoT 的长度控制

这一点本轮暂不定死，后续再议。

---

## 31. 最终结论

本项目当前推荐的实施方向是：

1. 继续把视觉能力封装进 `CodeImageTool`，helper 命名按能力层抽象。
2. 不直接复用现有 `ToolAgentLoop` 做 branching 生成，而是在当前目录下建设独立 offline pipeline。
3. 保留 planner，用它生成 step 0 全局 CoT 和候选工具链。
4. executor 每次只执行当前 step，但一个 step 允许多 helper 调用。
5. 每一步必须是 image-to-image，但不强制只返回一张图，也不强制合并成主图。
6. tool return 尽量贴近现有 `image / text / meta` 思路，文本可选，metadata 默认不进训练对话。
7. planner 应尽量提前做掉路线选择，减少 executor 的歧义空间。
8. 所有 trajectory 都保留，但不是所有 trajectory 都继续扩展。
9. judge 的核心目标是验证轨迹是否提升弱模型的可解性。
10. V1 exporter 先支持导出全部终止轨迹，再在后处理阶段筛选高质量子集。

---

## 32. 下一步最值得先写的文件

基于本文档，下一步最值得先明确的是：

1. `trajectory_schema.json`
2. `planner_output_schema.json`
3. `executor_runtime_result_schema.json`
4. `judge_record_schema.json`
5. `canonical_sft_sample_schema.json`

这 5 份 schema 先定住，后续开发会顺很多。

---

## 33. 一个举例的完整 pipeline 流程

这一节给一个完整但简化的例子，专门说明：

- planner 不是一次性规划完就结束
- executor 每次只执行一步
- 每执行完一步，都要重新把最新 trajectory 送回 planner
- 后续步骤允许被改写

### 33.1 样例问题

问题：

`Is the child behind the man with the umbrella closer to the camera than the bicycle next to the red car?`

原图：

- `original.jpg`

可用能力：

- `detect`
- `crop`
- `box`
- `segment`
- `depth`

### 33.2 Round 0: 初始化 root trajectory

系统创建：

- `trajectory_id = gqa_000123__root`

初始状态：

- `messages` 里只有 user 问题和原图
- `steps = []`
- `planner_history = []`
- `status = running`

### 33.3 Round 0: planner 读取 root

planner 看完原图和问题后，输出：

1. 现在不能直接回答
2. 生成一个 step 0 全局 CoT
3. 给出 2 条候选后续路线

例如：

- suggestion A
  - 先解决人物链条
  - 再解决车辆链条
  - 最后做 depth compare
- suggestion B
  - 先解决车辆链条
  - 再回到人物链条
  - 最后做 depth compare

这时需要注意：

- planner 给出的只是“从当前状态出发的候选路线”
- 不是说未来必须永远沿着 A 或 B 机械执行到底

### 33.4 Round 0: orchestrator 分叉 child trajectories

系统把 root 分成两个 child：

- `gqa_000123__A_r1`
- `gqa_000123__B_r1`

其中：

- `A_r1` 的 provenance 是“来自 root 的 suggestion A”
- `B_r1` 的 provenance 是“来自 root 的 suggestion B”

### 33.5 Round 0: executor 执行 A_r1 的第一步

executor 读取：

- `A_r1` 当前完整 messages
- root 那一轮 planner 的 suggestion A
- suggestion A 的第一个 step

它只执行第一步，例如：

- 先 `detect(man, umbrella, child)`
- 再根据检测结果 `crop` 出人物局部图

运行完成后，runtime 返回：

- `images = [step_001/output_0.png]`
- `text = ""` 或少量结构化文字
- `meta = {...}`

然后系统把以下内容追加进 `A_r1`：

- assistant 的当前 step CoT 和 code
- tool 的返回消息
- 新图 artifact
- 一个 step record

这时 `A_r1` 的状态变成：

- 已经有 1 个真实执行 step
- 拿到了一张更聚焦的人物局部图

### 33.6 Round 0: executor 执行 B_r1 的第一步

同理，`B_r1` 执行的是 suggestion B 的第一步。

例如它先：

- `detect(red car, bicycle)`
- 再 `crop` 出车辆局部区域

执行后，`B_r1` 也得到自己的最新 messages 和中间图。

### 33.7 Round 1: planner 重新读取 A_r1

这是最关键的一步。

此时 planner 看到的已经不是 root，而是：

- 原问题
- 原图
- `A_r1` 第一步 assistant 消息
- `A_r1` 的 tool 返回
- 人物局部新图

所以它会重新判断：

1. 现在能不能直接回答
2. 如果还不能，接下来最好的路线是什么

例如 planner 这时可能发现：

- 人物链条其实已经比较清楚
- 不需要再做原先设想的第二个人物 refine step
- 现在更应该直接去做车辆定位

于是它的新输出可能变成：

- suggestion A1
  - 下一步直接处理 red car 和 bicycle
  - 然后做 depth compare
- suggestion A2
  - 先对人物图做一次 segment
  - 再处理车辆
  - 再做 depth compare

这说明：

- root 轮次里 planner 对 A 的原始后续规划已经被改写了
- 已执行的第一步保留
- 未执行的未来后缀全部允许重写

### 33.8 Round 1: planner 重新读取 B_r1

同理，planner 再看 `B_r1` 时，也会基于车辆局部图重新规划。

例如它可能判断：

- 车辆已经很清楚
- 下一步应该切回人物链条

于是 `B_r1` 的后续路线和 root 时的原始 suggestion B 也可能发生变化。

### 33.9 Round 1: judge 或 selector 决定 frontier

此时系统可以对 `A_r1` 和 `B_r1` 做一次筛选。

例如：

- `A_r1` 的新图更有帮助，保留
- `B_r1` 的路线收益较低，先不继续扩展

那么下一轮 frontier 可能只剩：

- `A_r1`

注意：

- `B_r1` 不一定删除
- 只是“不进入下一轮继续扩展”
- 它仍然可以落盘保存，供后续分析或导出

### 33.10 Round 1: A_r1 再分叉并继续执行一步

假设 `A_r1` 当前轮 planner 给出两个 suggestion：

- `A1`
- `A2`

系统再次分叉：

- `gqa_000123__A1_r2`
- `gqa_000123__A2_r2`

然后：

- `A1_r2` 只执行 suggestion A1 的第一步
- `A2_r2` 只执行 suggestion A2 的第一步

例如：

- `A1_r2` 先做车辆 crop
- `A2_r2` 先做人像 segment

### 33.11 Round 2: 再次重规划，而不是接着跑旧链

假设 `A1_r2` 已经有：

- 人物局部图
- 车辆局部图

这时 planner 再看 `A1_r2`，可能会判断：

- 已经不需要更多局部操作
- 直接做 depth compare 即可

于是新的 suggestion 可能只剩：

- 下一步 depth
- 然后直接 answer

这依然不是“沿着旧链自然走到第三步”，而是：

- planner 读到最新状态后
- 认为最优后续已经变化
- 因此重写后缀

### 33.12 终止条件示例

假设 `A1_r2` 再执行一个 depth step 后，planner 读取到：

- 两个目标物体都已经明确
- depth 结果也足够支持判断

它这时可以输出：

- `can_answer_now = true`
- `direct_answer = yes`

于是该 trajectory 终止，状态变成：

- `status = answered`

如果另一个 trajectory：

- 连续两步执行失败
- 或达到步数上限
- 或 judge 认为无继续价值

则它会终止为：

- `status = failed` / `pruned` / `max_step_reached`

### 33.13 最终导出时怎么处理

生成阶段结束后，系统里可能有多条终止 trajectory：

- `A1_r2` answered
- `A2_r2` pruned
- `B_r1` stopped_early

exporter 在 V1 可以先全部导出并带上状态标签。

如果后处理阶段只选高质量样本，则可能只保留：

- 最终答对
- 工具链清晰
- judge 评价高
- 对弱模型有帮助

的那部分 trajectory。

### 33.14 这个例子最想说明什么

这个 pipeline 的正确理解应该是：

1. root 时 planner 给的是“初始候选路线”。
2. executor 每次只落当前一步，不负责把整条路线跑完。
3. 每执行完一步，新的图和信息都会改变后续最优决策。
4. 因此 planner 必须逐轮重跑。
5. trajectory 保存的是“真实发生过的历史”，不是一条预先写死的脚本。
