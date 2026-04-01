# Offline SFT Pipeline 对齐笔记

日期：2026-03-26  
目的：补充解释当前最容易混淆的设计点，包括 `global_chain_cot` 的语义、轨迹分叉的表示方式、`models.py/store.py` 的职责，以及未来服务化架构对 `CodeImageTool` / Ray / helper 的影响。

配套文档：

- `pipeline_schema_explainer_20260326.md`

---

## 1. `global_chain_cot` 不是“把路线定死”

这一点必须单独讲清楚。

如果只看一个简短例子，很容易把：

```json
"global_chain_cot": "先定位价格标签，再放大或裁剪，再 OCR。"
```

误解成：

- planner 已经把后续路线完全定死
- executor 只是机械执行

这不是我们现在想要的。

### 1.1 正确理解

`global_chain_cot` 的作用更接近：

- 站在“当前这一轮”的全局视角
- 先解释这个问题整体打算怎么解
- 提供更长程的分析路线
- 但不承诺后面绝不会改路

也就是说，它应该：

- 比单个 step 的思考更长
- 比 `suggestion_cot` 更高层
- 允许包含多个可能方向和判断分支

例如更合理的风格应该像：

> 当前问题看起来需要先确认图像方向和可读性，再决定是直接 OCR、先定位局部区域后 OCR，还是先做框选/裁剪来缩小视觉干扰。如果标签本身已经清晰且无方向问题，可能直接读取；如果标签较小、遮挡或混在复杂背景中，更稳妥的路线是先定位候选区域，再返回局部图继续判断，必要时再进入 OCR。只有在新图返回后，才能决定是否继续局部放大，还是已经足够直接作答。

这类 `global_chain_cot` 才符合两个目标：

1. 它能作为“第 0 轮、全轨迹视角”的高层思考。
2. 它不会把路线锁死，仍允许 rolling replanning。

### 1.2 和 `codevision_sft.json` 的关系

根目录的 [codevision_sft.json](/data/home/suchenghao/ToolVision/codevision_sft.json) 里，现有样本更像：

- 每一轮 assistant 先写当前轮 `<think>`
- 如果需要，再跟一个 `<tool_call>`
- 再根据 tool 返回继续下一轮 `<think>`

也就是说，现有数据里的 CoT 更偏“逐轮思考”。

而你现在想加的 `global_chain_cot`，更像：

- 用于生成态的高层规划记录
- 未来可抽成“第 0 轮”的长程 thought

所以这不是冲突，而是多了一层：

- 现有 `codevision_sft.json` 主要体现“逐轮 thought”
- 新 pipeline 还想保留一层“高层全局 thought”

这层信息当前确实主要存在于 `planner_output_schema.json` 的 `global_chain_cot` 里。

---

## 2. 多个 suggestion 和多个 step 是怎么理解的

### 2.1 `suggestions`

`suggestions` 表示：

- 同一轮 planner 提出的多个候选后续路线

例如：

- `s1`: 先框选再裁剪再 OCR
- `s2`: 先直接 OCR，再根据返回结果决定是否局部放大
- `s3`: 先做方向修正，再重新判断

所以：

- suggestion 是“候选路线”
- 不是“都必须执行”

### 2.2 `steps`

每个 suggestion 下面的 `steps` 表示：

- 如果沿这条 suggestion 往后走，未来可能执行的 step 序列

但 executor 每次只执行其中当前的第一步。

原因：

- 新图返回以后，后面的步骤完全可能改写
- 所以后面的 step 更多是 planner 当前视角下的预测计划
- 不是 rigid script

### 2.3 `step_goal` 和 `executor_instruction` 的区别

这两个字段不重复，语义不同：

- `step_goal`
  - 说的是“这一步想达成什么结果”
  - 面向目标

- `executor_instruction`
  - 说的是“让 executor 具体怎么写这一步的代码”
  - 面向执行

举例：

- `step_goal`
  - `定位价格标签并裁出局部区域`

- `executor_instruction`
  - `写代码先用 _call_ground_box 找到最可能的 price tag，再用 _call_dino_crop 返回单个最可信 crop，必要时打印 helper 的文本结果用于后续判断。`

所以：

- `step_goal` 更抽象
- `executor_instruction` 更贴近 prompt 给 executor 的落地要求

### 2.4 这些东西需要单独存吗

建议存。

至少建议存：

- planner 原始输出 JSON
- 被选中的 `suggestion_id`
- 被执行的 `step_id`

原因：

- 后面回放和 debug 时，必须知道“planner 当时想了什么”和“真正执行的是哪条”

这部分主要通过：

- `planner_output_schema.json`
- `trajectory.pending_execution`
- `trajectory.steps`

来承接。

---

## 3. trajectory 分叉后，保留的是树还是三条轨迹

结论：

> 逻辑上是一棵树，存储上是多条 trajectory 记录。

这是最自然的做法。

### 3.1 为什么

因为每次 planner 可能给 2 到 3 条 suggestion。

如果都值得扩展，那么更合理的是：

- 每个 suggestion fork 出一个 child trajectory
- 每个 child trajectory 后续各自独立推进

所以你在磁盘上会看到的是：

- `traj_root`
- `traj_root__s1`
- `traj_root__s2`
- `traj_root__s3`

它们通过下面这些字段串起来：

- `trajectory_id`
- `parent_trajectory_id`
- `fork_provenance.parent_suggestion_id`

### 3.2 父轨迹怎么处理

父轨迹通常不会消失。

它更像：

- 一个历史节点
- 记录“在这一轮被 planner 扩展出了哪些孩子”

如果后续只继续推进孩子，不再直接推进父节点，那么父节点可以进入类似：

- 已展开
- parked
- inactive

这样的内部状态。

当前 schema 里没有专门的 `parked` 状态，但这不一定是问题。

V0.1 完全可以先这样处理：

- 只有叶子 trajectory 是 `running`
- 父 trajectory 不再继续执行，但保留其历史记录

也就是说：

- 树结构主要靠 parent-child 关系表达
- frontier 只维护“当前活跃叶子”

### 3.3 这是不是 schema 之外的后端逻辑

一半是 schema，一半是后端逻辑。

schema 已经提供了足够关键的树结构字段：

- `parent_trajectory_id`
- `fork_provenance`

但：

- 什么时候 fork
- 父节点是否还在 frontier
- 只保留 top-k 还是全保留

这些确实属于 orchestrator / frontier 的后端策略，而不是 schema 本身决定的。

---

## 4. `models.py` 和 `store.py` 到底是做什么的

这两个名字很容易让人误会成“模型推理”。

但这里的 `models.py` 不是指大模型。

### 4.1 `core/models.py`

这里的 `models` 指的是：

- Python 数据模型
- 也就是把 JSON schema 变成 Python 里的类型对象

比如：

- `PlannerOutput`
- `TrajectoryRecord`
- `ExecutorRuntimeResult`
- `JudgeRecord`

为什么要这样做：

1. 代码里不要到处手写裸字典。
2. 读写 JSON 时可以做字段校验。
3. orchestrator / store / exporter 之间传对象更清楚。

换句话说：

> 它不是替代强模型 API，而是替代“全程拿 dict 硬拼”的工程写法。

即便 planner / executor / judge 都是 HTTP 调大模型：

- 请求前还是要组装数据对象
- 响应后还是要解析成结构化对象

这就是 `models.py` 的意义。

### 4.2 `core/store.py`

`store.py` 的职责是：

- 创建 trajectory 目录
- 保存和加载 `trajectory.json`
- 保存和加载 `messages.json`
- 保存 planner/runtime/judge 的落盘文件
- 实现 resume

也就是说：

- `models.py` 解决“对象长什么样”
- `store.py` 解决“对象怎么存、怎么读回来”

---

## 5. planner / executor / judge 除了 prompt，还要写什么

不只是 prompt。

后端至少还要写 4 类东西：

### 5.1 prompt builder

负责把：

- 当前 messages
- 当前图像列表
- 当前 trajectory 摘要
- 可用 helper 描述

组织成发给模型的输入。

### 5.2 model client

负责：

- 调 HTTP API
- 带鉴权、超时、重试
- 读取返回文本

如果以后不同角色用不同模型，这层就更需要独立出来。

### 5.3 parser / validator

负责把返回的文本解析成：

- planner JSON
- executor 代码和 thought
- judge JSON

并校验：

- 字段是否齐
- schema 是否合法
- 失败时怎么重试或 fallback

### 5.4 orchestration glue

负责把这些角色串起来：

- planner 出 suggestion
- executor 只拿当前 step
- runtime 执行
- judge 决定 keep / stop
- frontier 更新

所以 prompt 只是其中一层，不是全部。

---

## 6. 如果未来全部服务化，需不需要“全部推倒”

结论：

> 不要全部推倒，要保留 Tool 协议层，推倒模型承载层。

这也是当前最合理的长期方向。

### 6.1 应该保留的东西

- `CodeImageTool` 作为代码执行与 helper 注入层
- helper 接口名字
  - `_call_ocr_assist`
  - `_call_ground_box`
  - `_call_sam_mask`
  - `_call_dino_crop`
  - `_call_blur_bg`
- helper 统一返回协议
  - `{"image", "images", "text", "meta"}`

### 6.2 应该重做的东西

- `CodeImageTool` 内部承载 OCR / GroundSAM / 其他模型的方式
- 现在这种 actor 直接持有模型对象的方式

更理想的结构是：

- `CodeImageTool`
  - 安全执行代码
  - 注入 helper
  - helper 内部发 HTTP/gRPC 请求

- 模型服务
  - 常驻
  - 占 GPU
  - 做真正推理
  - 暴露稳定 API

---

## 7. 都服务化以后，Ray actor 还有没有价值

有，但价值会收缩。

### 7.1 未来仍然适合保留的 Ray 部分

- `CodeExecutionWorker`
  - 执行用户代码
- 限流 / 重试 / 调度类小 actor
- 如果要做本地缓存代理，也可以留一层轻 actor

### 7.2 未来不应该继续靠 Ray 承担的部分

- 模型常驻
- 模型显存管理
- 模型服务生命周期

这些应该转移到：

- 独立服务进程
- 容器
- systemd / supervisor / k8s 等

所以：

- “代码执行编排”仍然可以用 Ray
- “模型承载”不建议继续靠 Ray actor

---

## 8. `create -> execute -> release` 会不会影响模型常驻

不会影响“独立模型服务常驻”。

原因很简单：

- `create / execute / release` 管的是 tool instance
- 不是 model service instance

它影响的是：

- 当前这次 tool 调用的图片上下文

它不影响的是：

- 外部 HTTP 服务是否还活着
- 外部模型是否还占着 GPU

### 8.1 但它会影响什么

会影响：

- `CodeImageTool` 当前 instance 里的图片状态是否跨多轮保留

更准确地说：

- 单次 tool call 内部链式处理没问题
- 跨多轮 tool call 的图片状态，需要上层自己保存并重新传入

所以未来真正要重新设计的，是：

- 图像状态如何跨多轮保留
- trajectory 如何把新图回写并再次作为可见图片输入

而不是“模型常不常驻”。

---

## 9. 当前最推荐的架构建议

如果你已经决定：

- PP-OCR 换 v5
- GroundSAM 改独立服务
- 后续模型统一走独立部署

那建议正式朝这个方向收敛：

1. 保留 `CodeImageTool`
2. 保留 helper 名字和 helper 协议
3. 把 helper 背后的模型承载方式切成 HTTP/gRPC adapter
4. 让模型常驻完全脱离 tool instance 生命周期
5. Ray 只保留在代码执行池、限流、编排这层

---

## 10. 对当前 schema 的直接建议

基于今天的讨论，我建议：

1. `planner_output_schema.json`
   - 保留 `global_chain_cot`
   - 但 prompt 上要强调它是“高层可改写路线”，不是 rigid plan

2. `capability_plan`
   - 改按真实 helper 名输出

3. `trajectory_schema.json`
   - 现有 parent-child 设计够表达树
   - 不必额外为了“树”再造一份大 schema

4. `runtime_result_schema.json`
   - 继续保留 `meta`
   - 后面适配服务化时最有价值

5. `canonical_sft_sample_schema.json`
   - 当前先视为可选中间层
   - 不作为近期第一优先级

---

## 11. 现在最该冻结的三个东西

1. root sample 的最小字段
2. runtime wrapper 的输入输出接口
3. helper 服务化后的统一返回协议

这三件事一旦定了，后面不管底层模型怎么换，pipeline 主体都更稳。
