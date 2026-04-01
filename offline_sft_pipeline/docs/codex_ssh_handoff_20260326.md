
# Codex SSH 交接文档

日期：2026-03-26

## 1. 先说结论

不要只让新的 Codex 看目录结构。

现在这个仓库虽然目录已经比之前清楚，但“只看目录”还不够，因为真正关键的信息不在目录名里，而在这几类内容里：

1. 这次要做的目标不是改整个 `CodeVision`，而是新建一套独立的 offline branching SFT 生成 pipeline。
2. 要复用 `CodeVision` 里的 `CodeImageTool` 作为 sandbox/runtime 底座，而不是复用现有线性 `ToolAgentLoop`。
3. 当前已经冻结了一部分接口约定，包括 5 份 schema 和目录骨架。
4. 现在最重要的是按约定继续实现，不要重新发明架构。

所以迁移到 SSH 服务器后，应该给 Codex 一份明确的交接说明，而不是只丢一个仓库目录。

---

## 2. 项目目标

当前工作的目标是：

在现有仓库上新增一个 offline、多轮、分支式的 SFT 数据生成系统。

核心循环不是一次性固定计划，而是：

`planner -> executor -> tool runtime -> planner -> ...`

直到：

- planner 判断可以直接回答
- 或轨迹被 judge 淘汰
- 或命中错误 / 步数 / 预算上限

生成阶段保存完整 trajectory。

训练阶段把终止 trajectory 导出成线性的 SFT 数据，供 `LLaMA-Factory` 训练。

---

## 3. 当前仓库里哪些目录重要

仓库根目录当前主要有：

- `CodeVision/`
- `Grounded-SAM-2/`
- `paddleocr_vl_service/`
- `model_store/`
- `offline_sft_pipeline/`

理解方式如下：

### 3.1 `CodeVision/`

这是现有主仓库。

这里最重要的不是整套训练逻辑，而是：

- `CodeImageTool`
- 现有多轮 tool-use message 机制
- 现有 helper / worker / sandbox 实现

新项目会复用其中的 tool runtime 能力。

### 3.2 `Grounded-SAM-2/`

视觉工具后端之一。

主要是检测、框、mask、crop 等相关能力的来源。

### 3.3 `paddleocr_vl_service/`

OCR 相关后端服务。

### 3.4 `model_store/`

本地模型权重和模型资源存放位置。

迁移到 SSH 服务器后，需要重新确认这些模型路径和挂载方式。

### 3.5 `offline_sft_pipeline/`

这是本轮新建的项目目录骨架。

这是后续实现的主工作目录。

后续不要把 branching orchestration 直接硬塞回 `CodeVision` 现有 rollout 逻辑里。

---

## 4. 新 Codex 必读文件

进入 SSH 环境后，先读下面这些文件，不要直接开始写代码。

### 4.1 总设计文档

- `offline_multiturn_sft_pipeline_spec_20260326.md`

这份文档是当前架构共识。

里面已经明确了：

- planner 必须保留
- planner/executor 是交替循环，不是一次性总规划
- trajectory 保存真实执行历史，不保存固定未来脚本
- tool return 尽量贴近 `image / text / meta`
- V1 先导出所有终止 trajectory

### 4.2 一周开工计划

- `offline_sft_pipeline_one_week_plan_20260326.md`

这份文档是当前默认的工程推进顺序。

### 4.3 schema 目录

- `offline_sft_pipeline/schemas/trajectory_schema.json`
- `offline_sft_pipeline/schemas/planner_output_schema.json`
- `offline_sft_pipeline/schemas/executor_runtime_result_schema.json`
- `offline_sft_pipeline/schemas/judge_record_schema.json`
- `offline_sft_pipeline/schemas/canonical_sft_sample_schema.json`

这些 schema 不是最终业务代码，但代表当前已经冻结的数据接口。

### 4.4 `CodeVision` 里需要参考的代码

重点看：

- `CodeVision/verl/tools/code_image_tool.py`
- `CodeVision/verl/experimental/agent_loop/tool_agent_loop.py`
- `CodeVision/verl/utils/dataset/multiturn_sft_dataset.py`

看的目的不是复用整条链，而是理解：

- 当前 tool sandbox 怎么工作
- 当前多轮消息格式怎么承接
- 最终训练数据大概怎么对齐

---

## 5. 当前已经决定的事情

这些点默认已经定了，除非和真实实现强冲突，否则不要重新推翻。

### 5.1 训练和生成分层

- 训练端：`LLaMA-Factory`
- 生成端：新建 offline pipeline
- tool runtime：复用 `CodeImageTool`

### 5.2 planner 不能删

planner 不只是为了给 executor 提示。

它还负责：

- 提供 step 0 的全局 CoT
- 提前做路线选择
- 在每一轮根据最新图像和历史重规划

### 5.3 executor 每次只执行一步

一个 step 里允许多 helper 调用。

但 executor 不负责把整条 suggestion 从头跑到底。

### 5.4 planner 要逐轮重跑

不是：

- step 0 规划整条链
- executor 永远沿固定链条跑

而是：

- 每执行一步
- 新图和新 text 回写 trajectory
- 重新进 planner
- 后缀允许被改写

### 5.5 trajectory 是 offline 索引

它不是直接喂模型的输入对象。

它是后台状态存档，用于：

- 分叉
- 断点恢复
- judge
- export

### 5.6 新 helper 命名按能力，不按后端模型

推荐按能力：

- `detect`
- `crop`
- `box`
- `segment`
- `depth`
- `count`
- `ocr`

不要把训练数据和 prompt 绑死到某个后端模型名。

---

## 6. 当前已经完成到什么程度

当前已经完成的是“文档和接口层面”的启动工作。

已完成：

1. 新项目目录骨架已经建立：
   - `offline_sft_pipeline/configs`
   - `offline_sft_pipeline/prompts`
   - `offline_sft_pipeline/schemas`
   - `offline_sft_pipeline/core`
   - `offline_sft_pipeline/runtime`
   - `offline_sft_pipeline/pipelines`
   - `offline_sft_pipeline/outputs`
   - `offline_sft_pipeline/scripts`
2. 五份 schema 已写好。
3. 一周实施计划已写好。
4. 总设计文档已补充完整样例流程。

尚未完成：

1. `core/models.py`
2. `core/store.py`
3. `runtime` 对 `CodeImageTool` 的包装
4. planner / executor / judge client
5. orchestrator 主循环
6. exporter

所以当前阶段是：

架构和接口已经开头，代码实现基本还没开始。

---

## 7. 新 Codex 到 SSH 后第一件事该做什么

建议按这个顺序：

1. 确认 SSH 服务器上的 repo 根目录路径。
2. 确认 Python 环境和依赖是否齐全。
3. 确认 `CodeVision` 能否在服务器上正常 import。
4. 确认 `CodeImageTool` 依赖的外部服务和模型路径。
5. 先在 `offline_sft_pipeline/` 下实现最小 runtime wrapper。

不要一上来就写 planner 或 exporter。

第一优先级应该是：

先证明“executor 一段代码 -> CodeImageTool -> image/text/meta -> 落盘”这条链能跑通。

---

## 8. SSH 环境下要特别交代的事情

这是最容易漏的部分。

### 8.1 当前文档里的绝对路径是 Windows 风格

当前文档中很多路径写的是：

- `D:\\sdu\\ToolVision\\...`

迁移到 SSH 服务器后：

- 不要继续依赖这些绝对路径
- 一律以“仓库根目录”为基准改成相对路径或服务器绝对路径

### 8.2 先确认哪些服务要单独启动

至少要确认：

- OCR 服务怎么启动
- Grounded-SAM-2 怎么调用
- 模型权重在服务器上的位置

### 8.3 不要默认所有模型都能直接在同一环境里跑

一些 CV 后端可能还是要服务化调用。

所以要区分：

- `offline_sft_pipeline` 自己的 Python 环境
- 外部视觉模型服务环境

### 8.4 注意 SSH 环境通常没有 GUI

所以调试方式要偏向：

- 保存图片到磁盘
- 保存日志
- 写 replay / inspect 脚本

而不是依赖交互式图像查看。

---

## 9. 推荐的新 Codex 起手任务

到 SSH 服务器后，建议直接给 Codex 这个顺序的任务。

### Task 1

阅读并总结：

- `offline_multiturn_sft_pipeline_spec_20260326.md`
- `offline_sft_pipeline_one_week_plan_20260326.md`
- `offline_sft_pipeline/schemas/*.json`

要求它先确认理解，而不是直接编码。

### Task 2

实现：

- `offline_sft_pipeline/core/models.py`

把 5 份 schema 先映射成 Python `Pydantic` 类。

### Task 3

实现：

- `offline_sft_pipeline/core/store.py`

负责 trajectory、messages、planner output、runtime result、judge record 的落盘和加载。

### Task 4

实现：

- `offline_sft_pipeline/runtime/code_image_runtime.py`

先只支持：

- 输入代码
- 输入可见图像
- 执行单步
- 输出 `image / text / meta / error`

### Task 5

写一个 smoke 脚本，手工给一段 executor code，验证单步执行能跑通。

只有这一步通了，后面再接 planner / orchestrator 才有意义。

---

## 10. 不要让新 Codex做的事情

以下行为应该明确禁止或至少提醒：

1. 不要直接重构 `CodeVision` 主 rollout。
2. 不要把 branching 逻辑硬塞进现有 `ToolAgentLoop`。
3. 不要跳过 schema，直接随手写 JSON。
4. 不要一开始就设计过重的数据库或服务架构。
5. 不要在 helper 名称还没冻结时大量写 prompt。
6. 不要假设 planner 是一次性规划。

---

## 11. 建议给新 Codex 的一句话任务说明

可以直接给它下面这段：

> 你现在在 `ToolVision` 仓库里工作，目标不是改整个 `CodeVision`，而是在根目录的 `offline_sft_pipeline/` 下实现一套新的 offline branching SFT 数据生成 pipeline。请先阅读 `offline_multiturn_sft_pipeline_spec_20260326.md`、`offline_sft_pipeline_one_week_plan_20260326.md` 和 `offline_sft_pipeline/schemas/` 下的 5 份 schema，理解后先实现 `core/models.py`、`core/store.py` 和 `runtime/code_image_runtime.py`，不要改现有 `ToolAgentLoop` 的主逻辑。

---

## 12. 如果只给最小交接信息，至少要给这 6 条

如果你不想给一整篇文档，最少也要告诉新 Codex：

1. 主工作目录是 `offline_sft_pipeline/`，不是整个仓库随便改。
2. 目标是 offline branching SFT generation，不是在线 rollout。
3. 复用 `CodeImageTool`，不要复用整条 `ToolAgentLoop`。
4. planner 是逐轮重跑的，不是一步到位。
5. 5 份 schema 已经写好，先按 schema 映射成 Python 类。
6. 第一优先级是先跑通单步 runtime wrapper。

---

## 13. 这份交接文档的用途

这份文档是给“新环境里的 Codex”快速进入状态用的。

它解决的问题不是“目录长什么样”，而是：

- 目标是什么
- 该在哪做
- 哪些决定已经定了
- 哪些地方别乱动
- 第一周先做什么

如果只给目录，不给这些说明，新的 Codex 很容易走偏。

