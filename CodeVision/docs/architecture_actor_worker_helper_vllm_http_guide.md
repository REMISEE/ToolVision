# CodeVision 模型编排与部署详解（Helper / Worker / Actor / vLLM / HTTP）

本文针对你提的 5 个问题，按“概念 -> 当前代码 -> 架构选择 -> GPU 编排 -> 迁移路线”展开。

---

## 0. 一句话总览

你当前系统已经是一个“可用的工具编排雏形”：

- MLLM 负责决策（要不要调工具、调哪个工具）
- `CodeImageTool` 负责执行工具代码
- 工具内部通过一个 Ray actor（`ExternalModelWorker`）路由到 GroundSAM2 / OCR adapter

问题不在“能不能跑”，而在“后续模型增多后怎么避免串行排队和资源耦合”。

---

## 1. 先把术语彻底理顺

## 1.1 Helper 是什么

在你项目里，helper 是注入到“可执行代码沙箱”里的 Python 函数，例如：

- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`
- `_call_ocr_assist`

作用：给 MLLM 一组“高层能力函数”，避免它自己拼底层模型调用细节。

## 1.2 Worker 是什么

`worker` 是泛称，不是特定技术名。  
可理解为“干活的执行单元”（线程/进程/远程进程都可叫 worker）。

你代码里有两个“worker概念”：

1. `CodeExecutionWorker`：执行用户代码（沙箱执行池）
2. `ExternalModelWorker`：执行外部模型推理路由

## 1.3 Actor 是什么

`actor` 是 Ray 的具体机制：`@ray.remote class ...`

特点：

- 常驻有状态（模型可常驻内存）
- 可声明资源（`num_gpus`, `num_cpus`）
- 可并发控制（`max_concurrency`）

在你的代码里，`ExternalModelWorker` 本质上就是 Ray actor（虽然类名叫 Worker）。

## 1.4 Service 是什么

独立网络服务（HTTP/gRPC），通常单独进程/容器部署。  
Tool 通过网络调用它，不直接 import 它的推理代码。

---

## 2. 你当前项目里“谁调用谁”

下面是当前真实链路（简化）：

1. MLLM 产生 tool call（`code`, `image_index`, `description`）
2. `CodeImageTool.execute(...)` 执行这段 code
3. code 中调用 helper（如 `_call_sam_mask(...)`）
4. helper 调 `self._call_external_model(...)`
5. `self._call_external_model(...)` 调 Ray actor：
   - `ExternalModelWorker.infer(model_name, image, kwargs)`
6. actor 里按 `model_name` 选 adapter：
   - `grounded_sam2 -> GroundedSAM2Adapter`
   - `paddleocr_vl -> PaddleOCRVLAdapter`
7. adapter 输出统一结构：
   - `images`
   - `text`
   - `meta`
8. helper 返回给代码执行环境，`CodeImageTool` 把结果图返回给上层 MLLM

你可以把它理解成：

- helper = SDK 层
- ExternalModelWorker(actor) = 路由层 + 生命周期层
- adapter = 模型实现层

---

## 3. 你问的关键点：为什么会“共享配额、互相排队”

因为现在是“一个 actor 承接两类模型”：

- OCR 请求
- GroundSAM2 请求

且当前 `ExternalModelWorker` 是 `max_concurrency=1`。  
结果就是：

- 任一请求在跑，另一请求只能等
- OCR 慢会拖 GroundSAM2，反之亦然
- 资源扩展粒度粗（只能扩整个 actor）

这就是你听到的“共享队列 + 资源耦合”。

---

## 4. 现在结构是错的吗？

不是错，是“研发阶段可接受，生产阶段需演进”。

优点：

- 实现简单，调试快
- actor 常驻可复用模型，避免重复加载
- 对本地验证非常实用

短板：

- 模型多时排队明显
- 各模型资源隔离弱
- 依赖冲突风险上升

---

## 5. 阶段B拆成两个 Actor 难吗？能平滑过渡吗？

结论：难度中低，可平滑过渡。

## 5.1 为什么说不难

你对上层 helper 的接口已经稳定。  
只需改 Tool 内部路由，不改 MLLM 调用格式。

## 5.2 平滑方案

1. 保持 helper 不变（`_call_ground_box/_call_ocr_assist`）
2. 新增两个 actor：
   - `ExternalGroundSAMWorker`
   - `ExternalOCRWorker`
3. `CodeImageTool._call_external_model(...)` 根据 `model_name` 路由到不同 actor
4. 先灰度（保留旧 `ExternalModelWorker` 兜底），验证后下线旧入口

这样对训练数据、tool schema、上层 pipeline 都是无感升级。

---

## 6. GroundSAM2 已经“耦合进仓库”，还能拆吗？

能拆，且“代码耦合”不等于“部署耦合”。

你现在把 Grounded-SAM-2 repo 放本地，只是代码组织方式。  
部署时完全可以：

- MLLM 单独占一张 GPU
- Tool + GroundSAM2 先共用另一张 GPU（阶段性方案）
- OCR 单独服务（可其他卡或 CPU）

后续再把 GroundSAM2 也拆独立服务，不影响 helper 协议。

---

## 7. Actor 部署新模型 vs HTTP/gRPC 部署新模型

## 7.1 Ray Actor 方式

优点：

- 同进程生态，开发快
- 直接传 Python 对象（PIL/np）方便
- 模型常驻复用自然
- 资源声明简洁（`num_gpus=1`）

缺点：

- 与 Ray 生态绑定
- 跨语言/跨团队集成弱于 HTTP
- 服务治理（鉴权、网关、SLA）需要自己补

## 7.2 HTTP/gRPC 方式

优点：

- 边界清晰，强解耦
- 易运维、易监控、易限流
- 语言无关，团队协作好

缺点：

- 需处理序列化/协议/超时/重试
- 图片传输要编码（bytes/base64/url）
- 本地调试门槛略高

## 7.3 选型建议

- 研发早期：Actor 更快
- 生产长期：HTTP/gRPC 更稳
- 常见做法：内部先 Actor，成熟后服务化

---

## 8. HTTP 传图片会不会损耗很大？

会有损耗，但通常可控，关键看你怎么传。

## 8.1 常见传输方式

1. `multipart/form-data` 传 JPEG/PNG bytes（推荐）
2. JSON + base64（方便但体积膨胀约 33%）
3. 只传对象存储 URL（大图/批量场景更优）

## 8.2 粗略量级

- 1080p RGB 原始数据约 6MB
- JPEG 常见 100KB~800KB
- 同机/同内网 HTTP 往返通常是毫秒级到几十毫秒级

对你这种工具链路，模型推理耗时通常远大于传输耗时。  
因此 HTTP 不是不能用，关键是：

- 压缩格式选好（JPEG/PNG）
- 限制图像分辨率
- 做好超时和重试

---

## 9. PaddleOCR-VL1.5 应该用 actor 还是 vLLM？

先分清两层：

1. `vLLM` 是 LLM/VLM 推理引擎
2. `actor/service` 是你系统里的编排与承载方式

所以不是二选一。组合方式可以是：

- PaddleOCR-VL1.5 作为独立服务
- 服务内部使用 vLLM 后端（如果官方推荐且收益明显）
- Tool 仍通过 HTTP 调这个 OCR 服务

也可以短期先 actor 包装 OCR，但长期建议 OCR 服务化（你已经有 `vl_rec_server_url` 设计）。

---

## 10. 未来有很多模型时怎么设计（你的 4+4 场景）

你说的场景：

- 4 个 CV 小模型（GroundSAM、Depth 等）
- 4 个约 1B 的大模型

建议架构：

1. `MLLM Cluster`（vLLM）
   - 每个大模型一个 vLLM 实例（或按业务分组）
2. `CV Model Cluster`
   - 每个 CV 模型一个独立服务（早期可先一个模型一个 actor）
3. `Tool Orchestrator`
   - 统一路由到各模型服务
   - 统一返回 `{images,text,meta}`
4. `Gateway + Queue + Metrics`
   - 限流、重试、熔断、追踪

核心原则：

- 模型服务“独立扩缩”
- Tool 协议“统一不变”
- MLLM 只关心“该调用哪个工具”

---

## 11. Actor 可以独占 GPU 吗？能开很多个吗？

可以。

在 Ray 中：

- `num_gpus=1`：该 actor 启动时预留 1 张 GPU 资源
- 你可以开多个 actor，各自 `num_gpus=1`，前提是集群总 GPU 足够
- 也可用小数（如 `0.5`）做共享，但不推荐给重模型

## 11.1 Tool 和 actor 如何通信

通过 Ray 远程调用：

- `actor_handle.method.remote(...)`
- `ray.get(...)` 取结果

图片可以直接传 PIL/np（Ray 会序列化）。  
这和 HTTP 传 bytes 是两种通信机制。

---

## 12. 多个 vLLM 大模型怎么安排 GPU 和通信

## 12.1 GPU 分配

常见三种：

1. 一模型一GPU（简单直观）
2. 一模型多GPU张量并行（更大模型）
3. 多模型共享GPU（不推荐高负载场景）

## 12.2 通信方式

Tool/Orchestrator 通过 HTTP 调 vLLM（OpenAI 兼容接口）。  
你只需维护“模型名 -> endpoint”路由表。

## 12.3 典型路由策略

- `task=reasoning` -> MLLM_A endpoint
- `task=ocr_vl` -> OCR endpoint
- `task=grounding` -> GroundSAM endpoint

---

## 13. 结合你当前状态的落地建议（按优先级）

1. 先做阶段B：OCR actor 与 GroundSAM actor 拆开（最小改动，收益立竿见影）。
2. OCR 优先服务化（你已有 `vl_rec_server_url`，改动最小）。
3. GroundSAM 先保持 actor（稳定后再决定是否服务化）。
4. 新增 CV 模型不要再塞回同一个 actor，最少也要“一模型一 actor”。
5. 大模型统一走 vLLM endpoint，不要和 Tool 进程混跑。

---

## 14. 最后给你一个决策速记

- 追求“开发快” -> Actor
- 追求“长期稳、可运营” -> HTTP/gRPC 服务
- LLM/VLM 推理引擎优先考虑 vLLM
- CV 工具模型优先考虑独立服务或独立 actor
- 不要把所有模型堆进同一个执行单元

