# MLLM + Tool + GroundSAM2 + OCR 部署基础说明（从当前实现到生产架构）

本文针对你当前项目结构，解释以下问题：

- `worker` 和 `actor` 是什么，区别是什么
- 现在 `ExternalModelWorker` 同时挂 `grounded_sam2` 和 `paddleocr_vl` 到底意味着什么
- “共享资源配额、互相排队”具体是怎么发生的
- 现在结构是不是错的
- 后续引入 MLLM、更多模型后，应该如何演进为可扩展架构

---

## 1. 先看你当前代码的真实结构

当前关键点（`verl/tools/code_image_tool.py`）：

1. `CodeImageTool` 初始化时会创建一个 Ray actor：`ExternalModelWorker`。  
2. `ExternalModelWorker` 内部注册了两个 adapter：
   - `paddleocr_vl`
   - `grounded_sam2`
3. 调用任意 helper（`_call_ground_box/_call_sam_mask/_call_dino_crop/_call_blur_bg/_call_ocr_assist`）都会走：
   - `CodeImageTool._call_external_model(...)`
   - `ExternalModelWorker.infer(model_name, image, kwargs)`
4. 这个 actor 当前是 `max_concurrency=1`，意味着同一时刻只能处理 1 个请求。

因此，`grounded_sam2` 和 `paddleocr_vl` 现在确实是“同一入口、同一 actor 串行调度”。

---

## 2. worker / actor / service 到底是什么

## 2.1 Worker（广义）

`worker` 是通用词，指“干活的执行单元”，可以是线程、进程、容器、节点上的任务。

## 2.2 Ray Actor（你当前在用）

`actor` 是 Ray 里的“有状态远程对象”：

- 会常驻内存（可缓存模型权重）
- 可被远程调用方法
- 可以配置资源（`num_gpus`, `num_cpus`）
- 可设置并发（`max_concurrency`）

## 2.3 Service（独立服务）

`service` 是网络服务（HTTP/gRPC），通常独立进程/容器部署，和调用方通过网络协议通信。  
内部可以不用 Ray，也可以用 Ray Serve。

---

## 3. “共享配额、互相排队”是什么意思

你现在是一个 actor 同时承接两个模型族：

- OCR 请求
- GroundSAM2 请求

因为是同一个 actor 且 `max_concurrency=1`：

- A 请求（比如 OCR）进来后，B 请求（比如 SAM2）只能等
- 如果某一类请求慢，会拖另一类请求的延迟
- 资源上也不能独立扩缩（只能扩整个 actor）

这就是“共享资源配额 + 排队耦合”。

---

## 4. 现在结构是不是错的？

不是错，是“阶段性合理”：

- 优点：
  - 集成简单，调试成本低
  - 通过 actor 缓存模型，避免每次重载权重
  - 对本地 demo 非常友好
- 缺点：
  - 多模型混在一个执行入口，吞吐和隔离性一般
  - 版本/依赖冲突风险上升
  - 后续加模型会越来越难调度

结论：当前实现适合研发验证；生产阶段建议拆分。

---

## 5. 为什么一开始把 GroundSAM2 做成 actor

核心原因是“模型复用”：

1. GroundSAM2 初始化重（权重加载、构图、预处理组件）
2. 每次请求都重新 import+load 会很慢
3. actor 常驻后可复用已加载模型，显著降低请求时延

所以“做成 actor”本身是正确方向，问题不是 actor，而是“多个模型都塞进同一个 actor 串行”。

---

## 6. 你理想中的 pipeline（你说的思路）是否正确

你描述的目标是对的：

1. 单独部署 MLLM（文本/多模态生成）
2. Tool 作为后端图像执行层
3. Tool 再去调用外部模型（GroundSAM2/OCR 等）
4. Tool 返回处理后的图和文本提示给 MLLM继续推理

这就是标准的 Tool-augmented MLLM 架构。

---

## 7. 推荐目标架构（生产）

建议拆成 3 类服务：

1. `MLLM Service`
   - 只做生成和 tool call 决策
2. `Tool Orchestrator Service`
   - 执行 `code_image_tool`
   - 负责路由外部模型调用
3. `Model Services`
   - `GroundSAM2 Service`（独立）
   - `OCR Service`（独立）
   - 未来新增模型继续独立

这样每个模型可独立扩容、升级、分配 GPU、监控。

---

## 8. “单独部署后还需要 worker/actor 吗？”

看你选型：

1. 如果你做“独立 HTTP 服务（FastAPI/Triton/vLLM 等）”
   - 不一定要 Ray actor
   - 但服务内部仍需要 worker 进程/线程池

2. 如果你继续用 Ray 体系
   - 可以每个模型一个 actor（或 actor 池）
   - 或用 Ray Serve 暴露为服务

所以不是“有服务就不要 worker/actor”，而是“worker/actor 是服务内部实现机制”。

---

## 9. GPU 资源怎么安排（实用建议）

## 9.1 单卡研发（你本地 3060）

- 可行：串行验证
- 不建议：MLLM + GroundSAM2 + OCR 同时高并发
- 建议：
  - 先跑 Tool 单链路
  - OCR 尽量走远端服务或 CPU 模式

## 9.2 多卡生产（推荐）

- GPU0..N：MLLM
- GPUx：GroundSAM2
- GPUy（可选）：OCR（或 OCR 走 CPU/别的节点）

目标是让三类负载彼此隔离，避免抢卡。

---

## 10. 迁移路线（从你当前代码平滑演进）

## 阶段 A（当前）

- 单 `ExternalModelWorker` 同时挂 OCR + GroundSAM2
- 用于功能验证

## 阶段 B（建议尽快）

- 拆成两个 actor：
  - `ExternalGroundSAMWorker`
  - `ExternalOCRWorker`
- `CodeImageTool` 内路由到不同 worker

收益：至少解除 OCR 与 GroundSAM2 排队耦合。

## 阶段 C（生产）

- OCR 与 GroundSAM2 都独立服务化（HTTP/gRPC）
- Tool 只做编排与协议适配
- MLLM 服务与 Tool 服务分离

收益：最强的扩展性、隔离性和运维可控性。

---

## 11. 关键接口约定（建议固定）

为了后续换模型不改主流程，建议统一外部模型返回格式：

```json
{
  "images": ["...PIL/encoded..."],
  "text": "summary",
  "meta": {
    "model": "grounded_sam2|paddleocr_vl",
    "operation": "box|mask|dino_crop|blur_bg|ocr",
    "annotations": [],
    "latency_ms": 0
  }
}
```

你当前 `CodeImageTool._call_external_model()` 已基本符合这个方向。

---

## 12. 回答你这轮的直接问题（简版）

1. “现在结构错吗？”  
不是错，研发阶段合理；生产建议拆。

2. “后期加新模型还要写一起吗？”  
不建议。建议独立服务（或至少独立 actor）。

3. “理想是不是 MLLM 单独部署，Tool 后处理图，再回给 MLLM？”  
是，完全正确。

4. “单独部署就不需要 worker/actor 了吗？”  
不是。只是会从“业务层显式 actor”变成“服务内部 worker 机制”。

5. “为什么现在会互相影响？”  
因为同一个 `ExternalModelWorker` 串行承接两个模型请求，天然共享队列和资源。

