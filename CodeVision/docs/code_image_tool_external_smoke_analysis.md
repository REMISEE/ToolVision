# CodeImageTool External Demo（`--disable-external`）运行分析

日期：2026-03-04  
运行环境：WSL + conda `cvtool`  
命令：

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/sdu/CodeVision/tmp_demo_input.png \
  --disable-external \
  --out-dir outputs/code_image_tool_external_smoke
```

---

## 1. 这次测试的目标是什么

本次测试不是验证 PaddleOCR/GroundedSAM2 推理效果，而是验证：

1. `demo_code_image_tool_external.py` 脚本本身可执行。  
2. `CodeImageTool` 在 external demo 里的 case 编排、参数注入、输出保存流程正常。  
3. 在 `--disable-external` 模式下，基础 case 应成功、依赖外部模型的 case 应报“外部函数被禁用”，并且错误行为可控。

这属于“联调前 smoke test”。

---

## 2. 日志逐段解释

### 2.1 `Started a local Ray instance`

说明 Ray 启动成功，工具执行池可用。  
这一步通过后，`tool.create()/tool.execute()` 才能正常走 actor 执行链路。

### 2.2 `FutureWarning ... RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO`

这是 Ray 的前瞻性警告，不是失败。  
当前不影响本次测试结果，可忽略。

### 2.3 开头打印了一段 JSON schema

这是工具 schema 内容（`name=code_image_tool`，参数 `code/description/image_index`）。  
它表明 demo 正常构建了 tool schema，并把 schema 传给了 `CodeImageTool`。

---

## 3. 各 case 结果解读（逐项）

本次 external demo 共跑了 7 个 case：

1. `basic_pil`  
   - 结果：`reward=0.0`，`success=True`，输出图保存成功  
   - 解释：该 case 不依赖外部模型，仅做基础图像处理，符合预期成功。

2. `ocr_assist`  
3. `ground_box`  
4. `sam_mask`  
5. `dino_crop`  
6. `blur_bg`  
7. `focus_alias`
   - 共同结果：`reward=-0.05`，`success=False`，`Code execution error: External model functions are disabled.`  
   - 解释：这些 case 都需要 external helper（如 `_call_ocr_assist`、`_call_ground_box` 等）。你传了 `--disable-external`，所以全部失败是预期行为，不是 bug。

---

## 4. 这次结果说明了什么

结论：**结果完全符合预期，说明脚本和工具链是健康的。**

你已经验证了两件关键事：

1. external demo 脚本在当前环境可跑（Ray、参数、case 执行、输出路径都正常）。  
2. 外部模型开关生效（禁用时 helper 明确拒绝执行，而不是随机报错）。

---

## 5. 和无 MLLM demo 的关系

你目前已完成两层验证：

1. `demo_code_image_tool_no_mllm.py`：验证基础 pipeline（create/execute/crop/output）  
2. `demo_code_image_tool_external.py --disable-external`：验证 external demo 框架与 case 编排

下一层才是“真实模型层”：

3. external demo 在 `enable_external` 下逐个打开模型能力（Grounded -> OCR -> 串联）。

---

## 6. 为什么失败 case 反而是好现象

`--disable-external` 的语义是“只保留基础能力，不允许外部模型 helper”。  
在这个模式下：

- `basic_pil` 成功：证明基础路径正常。  
- helper case 统一失败且文案一致：证明开关控制精确，错误可预期。

这比“偶发成功偶发失败”更好，因为行为是确定的。

---

## 7. 下一步怎么测真实 external 能力

建议顺序：

1. 先只测 Grounded（本地进程内）  
   - 准备好 `sam2`/`grounding_dino` 安装和 checkpoint 绝对路径  
   - 不启 OCR 服务，先让 `ground_box/sam_mask` 跑通

2. 再只测 OCR（服务端）  
   - 起 PaddleOCR-VL `vllm-server`  
   - 验证 `ocr_assist` case

3. 最后测组合链路  
   - `ground_box -> ocr_assist` 或 `sam_mask -> dino_crop -> ocr_assist`

建议命令（下一步）：

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/sdu/CodeVision/tmp_demo_input.png \
  --out-dir outputs/code_image_tool_external_real \
  --device cuda \
  --sam2-checkpoint <abs_path_to_sam2_ckpt> \
  --sam2-model-config <abs_path_to_sam2_yaml> \
  --grounding-dino-config <abs_path_to_gdino_cfg> \
  --grounding-dino-checkpoint <abs_path_to_gdino_ckpt> \
  --external-worker-name code-image-external-model-worker-demo-v2
```

注：先不加 OCR 服务相关参数也可以，先把 Grounded 路径跑通。

---

## 8. 一句话总评

这次日志不是“失败”，而是 **预期成功地验证了 external 开关和脚本结构**。  
你现在可以进入“真实模型接入验证”阶段。

