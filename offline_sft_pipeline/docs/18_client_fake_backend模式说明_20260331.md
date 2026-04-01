# 18 Step 4：client_fake_backend 模式说明

日期：2026-03-31  
状态：已新增  
目的：说明 `run_single_sample_pipeline.py` 新增的 `client_fake_backend` 模式到底在验证什么、为什么这里让 `runtimeWrapper` 真跑、当前已经证明了哪些链路、还剩哪些真实环境阻塞。

---

## 1. 先给结论

当前 `offline_sft_pipeline/scripts/run_single_sample_pipeline.py` 已支持两种模式：

- `--mode scripted`
- `--mode client_fake_backend`

其中 `client_fake_backend` 还支持：

- `--runtime-mode scripted`
- `--runtime-mode code_image_tool`

这两种模式的区别不是“能不能跑 orchestrator”，而是：

- `scripted`：planner / executor / runtime / judge 里有多层 fake，主要用来稳定看目录结构和多轮语义
- `client_fake_backend`：只把 fake 压到 backend 返回层，其他都尽量走真实部件

也就是说，`client_fake_backend` 的目标是回答这个问题：

> 如果 planner / executor backend 能返回合规文本，judge backend 能返回合规分数，那么我们现有的真实 client + orchestrator + store + runtime wiring 能不能跑起来？

---

## 2. 现在两种模式分别怎么接

### 2.1 `scripted`

`scripted` 模式当前接的是：

- `ScriptedPlannerClient`
- `ScriptedExecutorClient`
- `ScriptedRuntime`
- `JudgeClient(backend=ScriptedJudgeBackend)`

这里面 fake 的层级比较高：

- planner client 本身就是 fake
- executor client 本身就是 fake
- runtime 本身也是 fake

它的价值是：

- 稳定复现同一个多轮 branching 结果
- 看目录结构
- 做 orchestrator 回归测试

### 2.2 `client_fake_backend`

`client_fake_backend` 模式当前接的是：

- `PlannerClient(backend=ScriptedTextBackend)`
- `ExecutorClient(backend=ScriptedTextBackend)`
- `JudgeClient(backend=ScriptedJudgeBackend)`
- runtime 由 `--runtime-mode` 决定：
  - `scripted` -> `ScriptedRuntime`
  - `code_image_tool` -> `CodeImageRuntimeWrapper(...)`

这里 fake 只保留在 backend 层：

- `ScriptedTextBackend`
  - 按 request key 返回预先写好的 planner / executor 原始文本
- `ScriptedJudgeBackend`
  - 返回预先写好的分数

而这些都是真实部件：

- `PlannerClient`
- `ExecutorClient`
- `JudgeClient`
- `OrchestratorV01`
- `OfflineTrajectoryStore`
- `CodeImageRuntimeWrapper`

所以这个模式更接近真正要上线的形态。

---

## 3. 为什么这里保留了可切换的 runtime 模式

这个点要明确，不然很容易混乱。

当前新增 `client_fake_backend` 的原始要求是：

> 把 fake 压缩到 Backend 具体调用函数上，其他的都走具体部件，不要 fake。

所以一开始 `client_fake_backend` 是按：

- `runtime = CodeImageRuntimeWrapper`

实现的。

但后续为了避免 runtime 环境成为当前验证阻塞，现在脚本把 runtime 显式拆成两个可选模式：

### 3.1 `--runtime-mode scripted`

这时验证目标是：

- 真实 planner client
- 真实 executor client
- 真实 judge client
- 真实 orchestrator
- 真实 store
- fake runtime

它的用途是：

- 先只验证 `backend -> client -> orchestrator -> store`
- 不让 runtime 环境、Ray、helper 服务成为当前阻塞

### 3.2 `--runtime-mode code_image_tool`

这时验证目标是：

- 除了模型 backend 和 judge backend 之外，其余都走真实部件

它的用途是：

- 在前面链条已经确认没问题后，再继续验证真实 runtime wiring

所以现在更准确的说法不是：

- “client_fake_backend 一定真跑 runtime”

而是：

- “client_fake_backend 默认先不测 runtime；如果需要，再切到真实 runtime 模式继续验证”

---

## 4. 这次实际验证到了什么

### 4.1 已经被证明跑通的链

现在已经新增了测试，验证：

- `ScriptedTextBackend -> PlannerClient -> PlannerOutput`
- `ScriptedTextBackend -> ExecutorClient -> ExecutorStepOutput`
- `JudgeClient(backend=ScriptedJudgeBackend)`
- `OrchestratorV01`
- `OfflineTrajectoryStore`

这条链在回归测试里已经通过：

- `python -m unittest offline_sft_pipeline.tests.test_orchestrator_v01`

所以当前已经能比较有把握地说：

> 在“backend 能返回合规文本/分数”的前提下，planner client / executor client / judge client / orchestrator / store 这一整条链是通的。

### 4.2 还没完全跑穿的层

脚本级 `client_fake_backend --runtime-mode scripted` 已经实际执行过一次。

结果是：

- 能跑到和 scripted demo 一致的完整多轮结果
- 说明当前 `PlannerClient / ExecutorClient / JudgeClient / OrchestratorV01 / Store` 这条链已经被脚本级验证过

另外，脚本级 `client_fake_backend --runtime-mode code_image_tool` 也实际执行过一次。

当前 probe 的结果是：

- planner client 真实跑了
- executor client 真实跑了
- orchestrator 真实跑了
- store 真实落盘了
- runtime 在本机环境里卡在 `CodeImageRuntimeWrapper` 对 Ray 的初始化阶段

实际错误形态是：

- `Timed out waiting for file /tmp/ray/session.../gcs_server_port_...`

所以当前还不能说：

- “这台机器上整条真实 runtime 链已经跑穿”

但可以说：

- “现在主要阻塞已经不在 planner / executor client / orchestrator / store，而是在 runtime 真实环境初始化”

---

## 5. 这说明什么

这次改动把问题边界切得更清楚了。

### 5.1 现在已经基本排除的问题

以下这些不再是主要怀疑点：

- orchestrator 主循环是否能工作
- store 落盘是否能工作
- planner client / executor client 的 prompt-build + parse 是否能工作
- judge client 的 record 组装是否能工作

因为这些现在都已经在：

- 测试里
- 或脚本实际执行里

被跑到过。

### 5.2 现在剩下的主要真实阻塞

当前主要剩三类真实阻塞：

1. planner / executor 的真实模型 API backend 还没接
2. judge 的真实打分 backend 还没接
3. runtime 的真实环境初始化和 helper 服务可用性还需要稳定

所以当前阶段可以更准确地表述成：

> 除了真实 backend 和 runtime 真实环境问题以外，pipeline 主干已经基本顺起来了。

---

## 6. `runtimeWrapper` 现在是不是“必须”真跑

不是。

现在已经显式提供了两种选择：

- `--runtime-mode scripted`
- `--runtime-mode code_image_tool`

所以当前建议的使用方式是：

1. 先跑：
   - `--mode client_fake_backend --runtime-mode scripted`
   - 用来确认 client/orchestrator/store 主干没有问题
2. 再视需要跑：
   - `--mode client_fake_backend --runtime-mode code_image_tool`
   - 用来继续排真实 runtime 环境

---

## 7. 对后续工作的建议

当前比较合理的推进顺序是：

1. 保留 `scripted`
   - 用来稳定看目录结构和 orchestrator 语义
2. 保留 `client_fake_backend`
   - `runtime-mode scripted` 用来证明“只差真实 backend”
   - `runtime-mode code_image_tool` 用来继续验证真实 runtime 环境
3. 后面再分别接：
   - 真实 `ApiTextBackend`
   - 真实 judge score backend
   - 稳定的 runtime/helper 环境

也就是说，当前这两种模式分别回答的是不同问题：

- `scripted`
  - 结构和多轮语义是不是对的
- `client_fake_backend`
  - 如果 backend 合规，真实 client/orchestrator/store/runtime wiring 是不是基本成立

---

## 8. 当前一句话总结

当前新增的 `client_fake_backend` 模式，已经把 fake 压缩到 planner/executor/judge backend 返回层；真实 `PlannerClient`、`ExecutorClient`、`JudgeClient`、`OrchestratorV01`、`OfflineTrajectoryStore` 已经被实际串起来验证。现在剩下的主要阻塞，不再是 pipeline 主干逻辑，而是：

- 真实文本 backend 接线
- 真实 judge backend 接线
- `CodeImageRuntimeWrapper` 所依赖的 runtime / Ray / helper 服务环境
