# CodeImageTool 外部模型改造文档

## 1. 本次改动概述

在原 `code_image_tool` 的基础上，新增了可在 `code` 字段中直接调用的外部模型函数，并保持原有能力兼容：

- 原能力保留：
  - PIL / numpy / cv2 任意多行处理；
  - `code/description/image_index` 协议不变。
- 新能力新增：
  - OCR：`_call_ocr_assist(...)`
  - GroundedSAM2：
    - `box`：`_call_ground_box(...)`
    - `mask`：`_call_sam_mask(...)`
    - `dino_crop`：`_call_dino_crop(...)`
    - `blur_bg`：`_call_blur_bg(...)`
  - 兼容别名：`_call_focus(...)`（等价于 `_call_ground_box(...)`）

## 2. 代码改动点

## 2.1 `verl/tools/code_image_tool.py`

新增/扩展：

- `BaseExternalModelAdapter`
- `PaddleOCRVLAdapter`
- `GroundedSAM2Adapter`（支持 box/mask/dino_crop/blur_bg）
- `ExternalModelWorker`（Ray actor，统一管理外部模型）

`GroundedSAM2Adapter` 关键变化：

- 先跑 GroundingDINO 得框；
- `mask/blur_bg/dino_crop(based_on=mask)` 会实际调用 `sam2_predictor.predict(...)`；
- `multimask_output` 现在真正参与推理逻辑（会在多 mask 里取最佳）。

`CodeImageTool._create_safe_globals(...)` 注入函数：

- `_call_ocr_assist(...)`
- `_call_ground_box(...)`
- `_call_sam_mask(...)`
- `_call_dino_crop(...)`
- `_call_blur_bg(...)`
- `_call_focus(...)`（兼容）

所有 helper 返回统一结构：

```python
{
  "image": PIL.Image,
  "images": list[PIL.Image],
  "text": str,
  "meta": dict,
}
```

并且都会更新当前执行环境中的 `image/img/draw`，便于链式多步处理。

## 2.2 `recipe/codevision/config/code_image_tool_config.yaml`

已同步：

- 外部模型 worker 配置；
- PaddleOCR-VL 服务配置；
- GroundedSAM2 路径与阈值配置；
- `code` 字段说明更新，加入所有 helper 的签名和示例。

> 你要求的“路径写死”保留在 YAML 默认值中。

## 2.3 `recipe/codevision/demo_code_image_tool_external.py`

新增详细注释版本 demo，覆盖 7 个 case：

1. `basic_pil`
2. `ocr_assist`
3. `ground_box`
4. `sam_mask`
5. `dino_crop`
6. `blur_bg`
7. `focus_alias`

## 3. 如何运行 demo

## 3.1 只测基础链路（不依赖外部模型）

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image <图片路径或URL> \
  --disable-external
```

## 3.2 测 OCR（需先启动 PaddleOCR-VL 服务）

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image <图片路径或URL> \
  --vl-rec-server-url http://127.0.0.1:8080/v1
```

## 3.3 测 GroundedSAM2（需权重和依赖）

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image <图片路径或URL> \
  --device cuda \
  --sam2-checkpoint ./checkpoints/sam2.1_hiera_large.pt \
  --sam2-model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --grounding-dino-config grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounding-dino-checkpoint gdino_checkpoints/groundingdino_swint_ogc.pth
```

## 4. 环境配置指南（基于原 repo + 当前改造）

下面给一套从零可复现流程（Linux/CUDA 场景）。

## 4.1 基础环境（原 repo）

```bash
conda create -n codevision python=3.10 -y
conda activate codevision

pip install --upgrade pip setuptools wheel
pip install -r requirements-runtime.txt
```

如果你只做工具层验证，至少需要：

- `ray`
- `qwen_vl_utils`
- `Pillow`
- `numpy`

## 4.2 PyTorch / torchvision（按 CUDA 版本）

示例（CUDA 12.1）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

请按你的驱动/CUDA 实际版本替换源。

## 4.3 PaddleOCR-VL

1) 启动服务（推荐容器）：

```bash
docker run \
  --rm \
  --gpus all \
  --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8080 --backend vllm
```

2) 保证 YAML 中 `vl_rec_server_url` 指向实际服务地址。

## 4.4 GroundedSAM2

安装（通常按官方仓库）：

- `sam2`
- `grounding_dino`
- 以及与 torch 匹配的 `torchvision`

准备文件（对应 YAML 默认路径）：

- `./checkpoints/sam2.1_hiera_large.pt`
- `configs/sam2.1/sam2.1_hiera_l.yaml`
- `grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- `gdino_checkpoints/groundingdino_swint_ogc.pth`

## 4.5 Ray actor 复用注意事项

`ExternalModelWorker` 使用 `get_if_exists=True`：

- 同名 worker 已存在时会复用旧实例；
- 改了配置不生效时，请：
  - 改 `external_model_worker_name`，或
  - 重启 Ray。

## 5. 常见问题

1) `_call_sam_mask` 没有效果  
- 检查 SAM2 权重/配置路径是否存在；
- 检查 `sam2` 和 `grounding_dino` 是否正确安装。

2) 只想跑 OCR，不想占 GPU  
- demo 默认 `--external-worker-num-gpus 0.0`；
- 生产配置可单独给 OCR worker 一个资源配置。

3) `dino_crop` 想按 mask 裁剪  
- 设置 `based_on="mask"`；
- 可配合 `multimask_output=True`。

## 6. 后续扩展方式

新增模型时只需：

1. 继承 `BaseExternalModelAdapter` 实现 `initialize/infer`
2. 在 `ExternalModelWorker.adapter_registry` 注册
3. 在 YAML 补配置 + 在 schema 文案补函数说明
