# 3 CodeImageTool 重构与服务化规划

日期：2026-03-26  
状态：规划稿  
目的：围绕你已经确认的方向，规划 `CodeImageTool` 的下一阶段重构：替换 PP-OCR、拆出 GroundSAM 服务、加入 depth / count，并把接口收敛成更适合 offline pipeline 的形式。

---

## 1. 目标

当前你要做的不是把 `CodeImageTool` 整个推倒。

真正要重构的是：

- `CodeImageTool` 背后的模型承载方式
- helper 到底怎么接服务
- helper 返回怎么更适合 offline pipeline

目标状态：

1. PP-OCR 替换为新的服务版本
2. GroundSAM 彻底独立服务化
3. 新增 depth 和 count 能力
4. helper 接口保持稳定
5. `CodeImageTool` 更像“安全代码执行 + helper 注入 + 服务调用编排层”
6. 让 runtime wrapper 可以直接消费更丰富、更结构化的返回

---

## 2. 当前基线

当前仓库里的 [code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py) 里，外部能力主要是：

- `PaddleOCRVLAdapter`
- `GroundedSAM2Adapter`
- `ExternalModelWorker`
- `OCRModelWorker`
- `GroundedSAM2ModelWorker`

也就是说，当前的外部模型还是主要靠：

- adapter
- Ray actor
- `CodeImageTool._call_external_model(...)`

来完成。

这条链对“先跑起来”很有用，但对你现在要做的长期服务化不够理想。

---

## 3. 建议保留什么

建议明确保留：

1. `CodeImageTool` 作为安全代码执行入口
2. helper 注入机制
3. helper 在 executor 代码中的使用方式
4. helper 层统一返回协议
   - `{"image", "images", "text", "meta"}`
5. `create -> execute -> release` 这套 tool instance 生命周期

这里要保留的不是“旧模型实现”，而是“上层调用方式”。

---

## 4. 建议重构什么

建议重点重构：

1. OCR adapter 的具体实现
2. GroundSAM adapter 的具体实现
3. `_call_external_model(...)` 的路由方式
4. helper 内部服务调用返回的标准化逻辑
5. `ToolResponse.text` / runtime 可观测信息

尤其是：

- 不再让 Ray actor 持有真正的大模型推理实例
- 改成 helper 发请求到独立服务

---

## 5. 目标架构

建议收敛成下面这层：

### 5.1 代码执行层

由 `CodeImageTool` 负责：

- 安全代码执行
- 注入 helper
- 管理当前 tool instance 的图片上下文

### 5.2 helper 调用层

由 helper 负责：

- 参数整理
- 选择服务端点
- 调用远端服务
- 统一返回 `image/images/text/meta`

### 5.3 模型服务层

由独立服务负责：

- 常驻
- GPU 占用
- 推理
- 健康检查
- 重试 / 扩缩容

---

## 6. 新能力集合建议

短期建议 helper 集合收敛成：

- `_call_ocr_assist`
- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`
- `_call_depth_assist`
- `_call_count_assist`

这里先不强制改旧名字。

原因：

- 旧 prompt 和旧使用习惯里已经有这些 helper 名
- 新 pipeline 的 `capability_plan` 也已经决定先按真实 helper 名输出

---

## 7. 具体改造方向

### 7.1 OCR

目标：

- 用新的 PP-OCR 服务替换现有 PaddleOCR-VL adapter

建议：

- 保留 `_call_ocr_assist(...)` 名字
- 把实际实现改成调用新的 OCR 服务
- `text` 返回 OCR 原始文本 / tokens
- `meta` 保存：
  - `model`
  - `tokens`
  - `boxes`
  - `confidence`
  - `service_latency_ms`

### 7.2 GroundSAM

目标：

- 不再在 tool 进程里持有 GroundSAM 推理

建议：

- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`

这些 helper 内部全部改成请求独立 GroundSAM 服务。

### 7.3 depth

目标：

- 增加深度辅助能力

建议 helper：

- `_call_depth_assist(...)`

最小可用返回：

- `text`: 深度排序 / 比较结论
- `meta`: 原始深度图、相对深度值、区域统计

### 7.4 count

目标：

- 增加计数能力

建议 helper：

- `_call_count_assist(...)`

最小可用返回：

- `text`: 计数结果文本
- `meta`: 检测框、计数依据、置信度

---

## 8. 对 Ray 的处理建议

未来不建议再让 Ray actor 承担：

- OCR 模型常驻
- GroundSAM 模型常驻
- depth / count 模型常驻

Ray 未来主要保留在：

1. `CodeExecutionWorker`
2. 代码执行限流
3. 轻量调度、缓存、重试代理

也就是说：

- 模型常驻归服务层
- 代码执行归 Tool / Ray 层

---

## 9. 对 `create / execute / release` 的判断

这套生命周期不用推倒。

因为它控制的是：

- 当前 tool instance 的图片上下文

它不控制的是：

- 远端模型服务是否常驻

所以在服务化架构里：

- 远端服务一直活着
- `create / execute / release` 只影响本次 tool 调用的上下文状态

---

## 10. 对接口的改造建议

为了更适合 offline pipeline，建议后续改造后做到：

1. helper 返回稳定的 `image/images/text/meta`
2. `CodeImageTool.execute()` 对外能暴露更真实的 tool 文本信息
3. `meta` 里保留更多结构化结果
4. runtime wrapper 能更容易拿到 helper 调用顺序
5. `ToolResponse` 和 `runtime_result` 之间的转换尽量轻

也就是说，不是重新发明一套 tool 协议，而是让现有协议更可落盘、更可回放、更适合 judge。

---

## 11. 推荐实施顺序

建议按这个顺序推进：

### Phase 0

先冻结：

- helper 名
- helper 返回协议
- 远端服务推荐协议

### Phase 1

重写 `CodeImageTool` 的外部调用层：

- 从“actor 持模型”
- 改成“adapter 调服务”

### Phase 2

先替换 OCR 服务。

原因：

- OCR 最容易直接验证 text 返回是否正确
- 也最影响 pipeline 的中间语义质量

### Phase 3

把 GroundSAM 全部迁到独立服务。

### Phase 4

加入 depth / count。

### Phase 5

补充 pipeline 友好的 runtime 可观测信息：

- 更真实的 `text`
- 更完整的 `meta`
- helper 调用观测

---

## 12. 推荐的代码改动范围

后续真正实现时，优先会碰这些位置：

- [code_image_tool.py](/data/home/suchenghao/ToolVision/CodeVision/verl/tools/code_image_tool.py)
- [code_image_tool_config.yaml](/data/home/suchenghao/ToolVision/CodeVision/recipe/codevision/config/code_image_tool_config.yaml)
- `offline_sft_pipeline/runtime/`
- `offline_sft_pipeline/core/`

如果服务层独立出去，还会新增：

- OCR service client
- GroundSAM service client
- depth service client
- count service client

---

## 13. 当前最值得先冻结的点

现在最值得冻结的不是“每个模型用什么框架”。

更值得冻结的是：

1. helper 名字
2. helper 返回协议
3. runtime wrapper 输入输出
4. 服务化后 `CodeImageTool` 仍然只是上层编排层

这四点一旦定了，底层模型后续怎么换都不会把 pipeline 打碎。

---

## 14. 一句话版本

你现在不需要推倒 `CodeImageTool`，你需要推倒的是“`CodeImageTool` 内部承载模型的方式”，把它改造成面向 offline pipeline 的稳定 helper 编排层。
