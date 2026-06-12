# GroundSAM2 联调运行文档（基于当前 ToolVision 目录）

适用目录结构：

```text
/mnt/d/sdu/ToolVision/
  ├─ CodeVision/
  └─ Grounded-SAM-2/
```

本文目标：在“昨天已跑通基础 demo”的基础上，验证 GroundSAM2 已接入 `code_image_tool`。

---

## 1. 结论先说

1. 现在 `API` 对接已经完成，不需要再改 Adapter 才能跑 GroundSAM2 基础能力。  
2. GroundSAM2 相关 helper 已在 `tool` 内接通并路由到统一 adapter。  
3. 你现在只需要在 `CodeVision` 目录运行 demo，并保证 4 个 GroundSAM2 路径可访问。

相关代码位置：

- GroundSAM2 operation 路由：  
  [code_image_tool.py](/D:/sdu/ToolVision/CodeVision/verl/tools/code_image_tool.py#L501)
- helper 到 operation 的映射：  
  [code_image_tool.py](/D:/sdu/ToolVision/CodeVision/verl/tools/code_image_tool.py#L764)
- external model 配置（YAML）：  
  [code_image_tool_config.yaml](/D:/sdu/ToolVision/CodeVision/recipe/codevision/config/code_image_tool_config.yaml#L23)

---

## 2. 前置检查（在 WSL 终端）

```bash
conda activate cvtool
cd /mnt/d/sdu/ToolVision/CodeVision

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import sam2; import importlib; importlib.import_module('groundingdino'); print('sam2/gdino import ok')"
```

检查关键文件：

```bash
test -f ../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt && echo ok_sam_ckpt
test -f ../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml && echo ok_sam_cfg
test -f ../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py && echo ok_gdino_cfg
test -f ../Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth && echo ok_gdino_ckpt
```

---

## 3. 直接运行 demo（启用 external）

> 注意：不加 `--disable-external`，才会跑 GroundSAM2 helper。

```bash
cd /mnt/d/sdu/ToolVision/CodeVision

python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/sdu/ToolVision/CodeVision/tmp_demo_input.png \
  --device cuda \
  --external-worker-name code-image-external-model-worker-demo-gs2-v1 \
  --out-dir outputs/code_image_tool_external_grounded
```

说明：

- `--external-worker-name` 建议每次改配置时换新名字，避免复用旧 Ray actor。
- 如果你还没部署 PaddleOCR 服务，`ocr_assist` case 失败是预期的，不影响 GroundSAM2 case 验证。

---

## 4. 预期结果判定

`demo_code_image_tool_external.py` 内置 case 顺序大致是：

1. `basic_pil`
2. `ocr_assist`
3. `ground_box`
4. `sam_mask`
5. `dino_crop`
6. `blur_bg`
7. `focus_alias`

在“仅 GroundSAM2 已就绪、OCR 服务未起”的场景下，预期：

1. `basic_pil`：成功  
2. `ocr_assist`：失败（预期）  
3. `ground_box`：成功  
4. `sam_mask`：成功  
5. `dino_crop`：成功  
6. `blur_bg`：成功  
7. `focus_alias`：成功（这是 `_call_ground_box` 的兼容别名）

输出图应在：

```text
outputs/code_image_tool_external_grounded/
```

重点检查：

- `02_ground_box.png`
- `03_sam_mask.png`
- `04_dino_crop.png`
- `05_blur_bg.png`
- `06_focus_alias.png`

---

## 5. 常见报错与定位

1. `Cannot import Grounded SAM2 dependencies`  
   说明 `sam2/grounding_dino/torchvision` 仍有安装问题，回到安装步骤排查。

2. `Missing Grounded SAM2 config fields`  
   说明 YAML/参数里四个路径字段缺失。

3. `No such file or directory`（checkpoint/config）  
   说明路径和当前 `cwd` 不一致。你必须从 `CodeVision` 目录启动。

4. OCR case 报连接失败  
   这是 PaddleOCR 服务未启动导致，与 GroundSAM2 不冲突。

---

## 6. 现在还要不要改 Adapter？

当前阶段不需要，为了“跑通 GroundSAM2 demo”已足够。  
仅在你有以下新需求时才改 Adapter：

1. 想切到 HuggingFace GroundingDINO API（替换本地 `grounding_dino` 包）  
2. 想新增返回字段（例如额外导出全部 masks/多 crop 元数据）  
3. 想改变 operation 协议（例如新增 `segment_only`/`track`）

---

## 7. 推荐下一步

1. 先按本文跑通 GroundSAM2 case。  
2. 再单独起 PaddleOCR 服务，补齐 `ocr_assist`。  
3. 最后验证组合链路：`ground_box -> ocr_assist`。
