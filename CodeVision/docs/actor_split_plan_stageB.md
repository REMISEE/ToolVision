# Stage B：ExternalModelWorker 拆分方案（GroundSAM2 / OCR）

## 1. 目标

- 把当前“单一 `ExternalModelWorker` 承接所有外部模型”的结构，拆成“按模型域分开”的 actor。
- 第一阶段只拆两路：
  - `GroundSAM2` 独立 actor
  - `PaddleOCR-VL` 独立 actor
- 保持上层 helper 接口不变：
  - `_call_ground_box/_call_sam_mask/_call_dino_crop/_call_blur_bg/_call_ocr_assist` 不改签名。

## 2. 现状问题

- OCR 与 GroundSAM2 共享一个 actor 队列，容易互相排队。
- 资源只能整块配置，无法按模型单独调 `num_gpus/num_cpus/max_concurrency`。
- 后续加模型会持续扩大单 actor 的耦合度。

## 3. 目标结构

- `CodeImageTool` 维护两个 actor handle：
  - `self.ocr_model_worker`
  - `self.grounded_sam2_worker`
- `_call_external_model(model_name, ...)` 按 `model_name` 路由到对应 actor。
- 每个 actor 只注册自己负责的 adapter。

## 4. 代码改动清单（最小改动）

文件：`verl/tools/code_image_tool.py`

1. 新增两类 actor（可直接复用现有 adapter 逻辑）
   - `OCRModelWorker`：仅支持 `paddleocr_vl`
   - `GroundSAM2ModelWorker`：仅支持 `grounded_sam2`

2. 保留现有 `ExternalModelWorker` 作为兼容兜底，不立即删除。

3. `CodeImageTool.__init__` 增加模式开关
   - `external_worker_mode: "single" | "split"`
   - 默认先保守设为 `single`（避免破坏存量）
   - 新环境可设 `split`

4. `CodeImageTool.__init__` 在 `split` 模式下创建两个 actor
   - OCR actor 读取 OCR 专属资源配置
   - GroundSAM2 actor 读取 GroundSAM2 专属资源配置

5. `_call_external_model` 路由规则
   - `model_name == "paddleocr_vl"` -> OCR actor
   - `model_name == "grounded_sam2"` -> GroundSAM2 actor
   - 未命中时抛明确错误或回落单 actor（按配置）

6. 增加日志
   - 打印请求路由到哪个 actor
   - 打印 actor 初始化参数（name/gpu/cpu/concurrency）

## 5. 配置草案（向后兼容）

在现有配置上新增：

```yaml
external_worker_mode: split

external_workers:
  paddleocr_vl:
    name: code-image-ocr-worker
    num_gpus: 0
    num_cpus: 2
    max_concurrency: 4
  grounded_sam2:
    name: code-image-groundsam-worker
    num_gpus: 1
    num_cpus: 2
    max_concurrency: 1
```

兼容规则：

- 若 `external_worker_mode` 缺失，默认 `single`，走旧逻辑。
- 若设为 `split` 但 `external_workers` 缺失，回落到旧参数并给 warning。

## 6. 灰度与回滚

灰度步骤：

1. 先上线代码，但保持 `external_worker_mode=single`。
2. 在测试环境切 `split`，跑 demo 与回归集。
3. 观察稳定后，再在生产切 `split`。

回滚策略：

- 一键改回 `external_worker_mode=single`，无需回滚代码。

## 7. 验证清单

功能验证：

- `basic_pil` 正常。
- `ground_box/sam_mask/dino_crop/blur_bg` 正常。
- `ocr_assist` 在 OCR 依赖可用时正常。

路由验证：

- OCR 请求只出现在 OCR actor 日志。
- GroundSAM2 请求只出现在 GroundSAM2 actor 日志。

并发验证：

- 并发压测下，OCR 不再阻塞 GroundSAM2 请求。

## 8. 风险与规避

风险：

- 资源配置不当导致 actor 起不来或争抢 GPU。
- actor 名冲突（`get_if_exists=True` 下复用旧实例）。

规避：

- 为两个 actor 使用不同 name。
- 切配置前先 `ray stop --force` 清理旧实例。
- 首次切 split 时降低并发，先保守运行。

## 9. 后续扩展（Stage C）

- 新增模型不再加入同一个 actor。
- 优先“一个模型一个 actor”或“一个模型一个服务”。
- 当依赖冲突明显时，直接服务化（HTTP/gRPC）并由 Tool 路由调用。

