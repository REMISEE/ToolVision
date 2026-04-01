# Actor / Worker / Service 常见问题（面向当前 CodeVision 实现）

本文专门回答你这 5 个问题，并尽量用“类/对象”的方式解释。

---

## Q1. `CodeExecutionWorker` 也是 actor 吗？到底什么是 actor？

是的，按你当前代码，`CodeExecutionWorker` 和 `ExternalModelWorker` 都是通过 `@ray.remote` 启动的远程执行单元（actor 用法）。

可把它理解成：

- **普通 Python 类**：只在当前进程内创建对象，生命周期跟当前进程走。  
- **Ray actor**：一个“远程常驻对象”，在 Ray 管理的进程里活着，你通过句柄远程调用它的方法。

类比：

1. 普通类像“本地对象”  
   `obj = MyClass(); obj.foo()`
2. actor 像“远程对象”  
   `h = MyClass.remote(); h.foo.remote()`

所以可以说：

- `actor` 不是新语法，而是“类实例化位置和调用方式”变成了远程。
- `worker` 是泛称，`actor` 是 Ray 里的具体实现机制。

你代码中的线索：

- `@ray.remote class ExternalModelWorker`  
- `@ray.remote(...) class TokenBucketWorker`
- `ray.remote(CodeExecutionWorker).options(...).remote(...)`

---

## Q2. `max_concurrency=1` 改大能解决问题吗？

能缓解一部分排队，但不是根治，且有副作用。

## 2.1 什么时候有帮助

如果一个 actor 里是轻量任务，或模型支持并行推理，调大并发可提升吞吐。

## 2.2 你当前场景的问题

你这个 actor 里挂了多个模型（GroundSAM2 + OCR），并且都可能是重推理：

- 并发调大后，会出现更重的显存竞争
- 可能 OOM 或抖动（延迟更不稳定）
- OCR 和 GroundSAM2 依然共享同一个队列入口（逻辑耦合仍在）

## 2.3 结论

- `max_concurrency` 是“调度参数优化”，不是“架构解耦”。  
- 真正解耦要拆 actor（至少按模型拆）。

---

## Q3. “每个模型一个 actor”是不是等于开很多终端、很多显卡？

不是“很多终端”，而是“很多远程进程/实例”，由 Ray 调度。

可以把它理解成：

- 你声明了多个模型实例（actor）
- Ray 按你给的资源约束（GPU/CPU）把它们放到合适节点/显卡

终端只是你看日志/启动命令的入口，不是部署实体本身。

---

## Q4. `num_gpus=1` / `0.5` 是什么含义？可以自定义把多个模型放一张卡吗？

可以自定义，但要分清“调度声明”和“真实显存占用”。

## 4.1 `num_gpus=1`

- 表示该 actor 申请 1 个 GPU 资源配额
- Ray 会给它分配可见 GPU（通常通过 `CUDA_VISIBLE_DEVICES`）

## 4.2 `num_gpus=0.5`

- 表示“逻辑上”两个 actor 可共享同一张 GPU 配额
- 但真实是否能共存，取决于两个模型显存占用和峰值行为

## 4.3 能否把多个模型放同一卡

可以，但要你自己评估：

- 常驻显存之和
- 峰值显存（推理瞬时）
- 并发情况下的波动

建议：

- 重模型（GroundSAM2、大 OCR/VLM）优先独占或低并发共享
- 小模型可尝试共享
- 先压测再上线

---

## Q5. 服务化（HTTP/gRPC）和 actor 的关系是什么？为什么常说先 actor 后服务化？

不是互斥关系，而是两层不同问题：

1. **actor**：进程内/集群内“怎么承载模型实例”
2. **服务化**：系统间“怎么通信与治理”（HTTP/gRPC、鉴权、限流、SLA）

## 5.1 只用 actor 的特点

- 快速开发
- Python 对象直接传，接线简单
- 但跨团队、跨语言、生产治理不够标准化

## 5.2 服务化的特点

- 清晰边界，易运维
- 通常传 JSON + 二进制图片（或 URL）
- 增加了协议和网络层成本，但可控

## 5.3 为什么建议“先 actor，后服务化”

因为这是最稳的演进路径：

1. 先把功能和模型效果跑通（actor 快）
2. 再把稳定能力抽成服务（治理更强）

不是说小模型不能一直 actor，而是：

- 当模型数量、调用方、并发、SLA 增加时，服务化优势会明显超过开发便利性。

---

## 6. 你当前最务实的架构建议（结合现状）

你现在可以这样做：

1. 先保持 GroundSAM2 现状（不重构 repo）
2. 把 OCR 从同 actor 拆出去（独立 actor 或独立 HTTP 服务）
3. MLLM 单独部署（建议 vLLM）
4. Tool 做编排层，统一调用入口不变

这就是“平滑过渡”：

- 上层 helper 不改
- 下层部署逐步解耦

---

## 7. 你最容易混淆的一句话（记住这个就够）

`worker` 是角色名，`actor` 是 Ray 的实现方式，`service` 是系统边界方式。  
它们不是同一维度，不冲突，可以组合使用。

---

## 8. 补充：HTTP 传图到底会不会很亏

如果用 `multipart/form-data` 直接传压缩图（JPEG/PNG），在内网通常可接受。  
对你这类视觉模型，推理时间往往远大于传输时间。  
真正要关注的是：

- 统一压缩策略
- 合理超时/重试
- 可观测性（日志、延迟、失败率）

---

## 9. 对应你代码里可直接观察的位置

- `ExternalModelWorker` 创建参数（`max_concurrency`, `num_gpus`, `num_cpus`）  
  在 `CodeImageTool.__init__` 中配置
- `ExternalModelWorker.infer()` 内部 adapter 路由  
  可看到 `grounded_sam2` / `paddleocr_vl`
- `CodeExecutionWorker` 作为代码执行池  
  由 `init_code_execution_pool` 初始化

这些都在：

- `verl/tools/code_image_tool.py`

