# CodeImageTool 无 MLLM Demo 运行分析（2026-03-04）

本文基于你实际运行成功的一次命令输出，解释：

- 这次 demo 走了什么流程
- 每一步的意义是什么
- 结果如何判断为“通过”
- 它和 `demo_code_image_tool_external.py` 的区别
- 这个 demo 是否有价值，以及下一步怎么测 external demo

---

## 1. 本次运行信息（事实记录）

执行命令：

```bash
python recipe/codevision/demo_code_image_tool_no_mllm.py \
  --image /mnt/d/sdu/CodeVision/tmp_demo_input.png \
  --bbox-xyxy "250,150,390,210" \
  --out-dir outputs/code_image_tool_no_mllm \
  --output-name run_b.png
```

关键输出（你日志中的核心字段）：

- `Started a local Ray instance`
- `[info] total_input_images=1`
- `[info] bbox_xyxy=(250, 150, 390, 210)`
- `[info] reward=0.0`
- `[info] metrics={'success': True, 'message': 'Code executed successfully', 'processed_image_index': 0, 'total_images': 1}`
- `[info] saved=outputs/code_image_tool_no_mllm/run_b.png`

输出文件：

- [run_b.png](/D:/sdu/CodeVision/outputs/code_image_tool_no_mllm/run_b.png)
- 分辨率：`140 x 60`（与 bbox 宽高一致）

---

## 2. 这次到底验证了什么

这次验证的是 **CodeImageTool 基础执行链路**，不依赖 MLLM，也不依赖 OCR/GroundedSAM2：

1. 本地 Ray 初始化成功（执行池可用）
2. `CodeImageTool.create()` 能接收输入图
3. 代码沙箱能执行裁剪代码
4. `tool.execute()` 返回 `ToolResponse(image/text)` 结构正常
5. 输出图保存成功

对应代码入口：

- [demo_code_image_tool_no_mllm.py](/D:/sdu/CodeVision/recipe/codevision/demo_code_image_tool_no_mllm.py#L85)
- [demo_code_image_tool_no_mllm.py](/D:/sdu/CodeVision/recipe/codevision/demo_code_image_tool_no_mllm.py#L112)
- [demo_code_image_tool_no_mllm.py](/D:/sdu/CodeVision/recipe/codevision/demo_code_image_tool_no_mllm.py#L115)

---

## 3. 流程解释（按脚本执行顺序）

1. 解析参数  
   接收 `--image`、`--bbox-xyxy`、`--out-dir`、`--output-name` 等。

2. 生成要执行的代码  
   你这次没传 `--code-file`，因此脚本根据 bbox 自动生成一段 `image.crop(...)` 代码。

3. 创建工具实例  
   `build_tool_config()` 里 `enable_external_model_functions=False`，所以明确不走外部模型分支。

4. 执行一次工具调用  
   `tool.execute(...)` 收到 `{code, description, image_index}`，在安全环境中运行代码。

5. 保存结果图  
   将 `response.image[0]` 保存到目标路径，并打印 reward/metrics。

---

## 4. 每个关键指标的意义

1. `reward=0.0`  
   对这类成功执行路径是正常值（非错误惩罚）。

2. `metrics.success=True`  
   代表执行成功，不是语法/越界/返回类型错误。

3. `message='Code executed successfully'`  
   说明沙箱代码执行完成并产出合法图像。

4. `processed_image_index=0`  
   多图模式下说明处理的是第 0 张图。

---

## 5. 与 `demo_code_image_tool_external.py` 的区别

`no_mllm` demo（当前跑的）：

- 目标：验证基础 pipeline（创建实例、执行代码、返回图像）
- 默认关闭外部模型
- 适合你现在“先不接模型”阶段

`external` demo：

- 目标：验证 helper 与外部模型联动（OCR/Grounded）
- 可以测试 `_call_ocr_assist`、Grounded 相关函数等
- 依赖外部模型环境是否就绪

文件：

- [demo_code_image_tool_no_mllm.py](/D:/sdu/CodeVision/recipe/codevision/demo_code_image_tool_no_mllm.py)
- [demo_code_image_tool_external.py](/D:/sdu/CodeVision/recipe/codevision/demo_code_image_tool_external.py)

---

## 6. 这个 demo 是不是“纯 toy，没意义”

不是 toy，它是 **必要的分层验证**：

1. 先证明“工具层”无问题（你已完成）
2. 再引入“模型层”排查依赖/权重/服务问题

这样一旦外部模型报错，你可以确认不是 `CodeImageTool` 基础链路导致。

---

## 7. 现在可以去测 external demo 吗

可以，建议按两步：

1. 先跑 external demo 的“基础模式”（禁用外部模型）  
   目的是确认 external 脚本本身可执行、参数链路正常。

2. 再逐个打开模型能力（先 Grounded，再 OCR，最后串联）  
   目的是把问题范围收敛到具体模型侧。

建议先执行：

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/sdu/CodeVision/tmp_demo_input.png \
  --disable-external \
  --out-dir outputs/code_image_tool_external_smoke
```

如果你愿意，下一步我可以基于你当前 `demo_code_image_tool_external.py` 的实际 case 列表，给你一份“逐 case 预期结果对照表”（哪个应成功、哪个在禁用外部时应失败、失败文案应该是什么）。

