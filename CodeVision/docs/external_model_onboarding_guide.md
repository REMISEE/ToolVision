# CodeImageTool 外部模型接入与本地联调指南（WSL2 / Windows）

本文面向当前仓库 `code_image_tool` 的外部模型能力，目标是把以下两条链路稳定接上：

- `PaddleOCR-VL-1.5`（通过 `vllm-server` 方式调用）
- `Grounded-SAM-2`（本地 Python 进程内推理）

同时覆盖你当前最关心的几点：

- 是否要先不把 `paddleocr` 放进基础依赖
- Grounded-SAM-2 仓库里很多目录（dataset/training）到底用不用
- 不接 MLLM 时如何独立验证 pipeline
- Windows + WSL2 的推荐操作路径

---

## 1. 推荐部署形态（先统一认知）

当前代码的两条外部通路不是同一种模式：

1. OCR：`PaddleOCRVLAdapter` 里调用 `PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=...)`  
   也就是**客户端 + 服务端**模式。  
   客户端在 `CodeVision` 进程里，服务端是独立的 vLLM 服务进程。

2. Grounded-SAM-2：`GroundedSAM2Adapter` 直接 import `sam2` 和 `grounding_dino`，并在本进程里加载权重。  
   也就是**本地进程内推理**模式。

这意味着最稳妥的本地方案是：

- `CodeVision`（客户端）环境：运行你的工具和 demo，包含 `ray + paddleocr + grounded-sam2 依赖`
- `PaddleOCR-VL`（服务端）环境：单独起 vLLM 服务（可以同机同卡）

---

## 2. `requirements` 怎么管理更合理

### 2.1 结论（建议）

如果你当前阶段是“先测基础功能 + 逐步接模型”，建议把依赖分层：

- 基础层（必装）：`requirements-runtime.txt` 保留通用依赖
- 外部模型层（可选安装）：按需安装 OCR / Grounded-SAM-2

也就是说，你可以暂时不强制要求所有人都安装 `paddleocr`，但你自己要测 OCR 时仍需安装。

### 2.2 为什么这样做

- `paddleocr + vllm` 路径较重，且容易和其他包产生版本耦合
- Grounded-SAM-2 需要编译/安装额外组件，适合按需安装
- 先把基础工具链跑通，再叠加模型，定位问题更快

---

## 3. Windows + WSL2 推荐工作流

### 3.1 是否必须 WSL2

- 纯逻辑测试（不跑外部模型）可在 Windows 原生做
- 真正接 Grounded-SAM-2，建议在 WSL2（Ubuntu）做，成功率更高

### 3.2 关键原则

- 不要混用 Windows Anaconda 环境和 WSL 环境
- 在 WSL 里单独安装 Miniconda/Conda，并在 WSL 环境里创建 `conda env`

---

## 4. 在 WSL2 建立可用环境（推荐步骤）

以下命令在 WSL 终端执行：

```bash
# 0) 进入项目
cd /mnt/d/sdu/CodeVision

# 1) 创建环境
conda create -n cvtool python=3.10 -y
conda activate cvtool

# 2) 基础工具链
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-runtime.txt

# 3) 若你按 PyTorch 官网方式已装 torch/torchvision，可跳过；否则补齐
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) 校验
python -c "import ray, PIL, cv2; print('base ok')"
python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

---

## 5. Grounded-SAM-2 接入（本地进程内）

## 5.1 你到底需要它仓库里的哪些内容

你只需要：

- `sam2` 包（推理代码）
- `grounding_dino` 包（检测代码）
- `checkpoints/` 与 `gdino_checkpoints/` 权重
- 配置文件（如 `configs/sam2.1/sam2.1_hiera_l.yaml`、`GroundingDINO_SwinT_OGC.py`）

`dataset/`、`training/`、标注相关目录都不是你当前推理接入必需项。

### 5.2 安装步骤（WSL）

```bash
cd ~
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

# 官方推荐安装方式
python -m pip install -e .
python -m pip install --no-build-isolation -e grounding_dino

# 下载权重
cd checkpoints && bash download_ckpts.sh
cd ../gdino_checkpoints && bash download_ckpts.sh
cd ..
```

### 5.3 本地先跑官方 demo（强烈建议）

```bash
cd ~/Grounded-SAM-2
python grounded_sam2_local_demo.py
```

如果官方 demo 都没过，不要先回头怀疑 `CodeVision` 接口层。

### 5.4 接到你项目时最关键的路径处理

当前 `code_image_tool_config.yaml` 的默认路径是相对路径，通常会失效。  
请改成 WSL 绝对路径，例如：

```yaml
external_models:
  grounded_sam2:
    device: "cuda"
    sam2_checkpoint: "/home/<you>/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_config: "/home/<you>/Grounded-SAM-2/configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_dino_config: "/home/<you>/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint: "/home/<you>/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
```

---

## 6. PaddleOCR-VL-1.5 接入（客户端 + 服务端）

### 6.1 先区分两侧

1. 客户端（CodeVision 侧）
- 需要 `paddleocr` 包
- 通过 `vl_rec_server_url` 调服务

2. 服务端（独立进程）
- 需要安装 `genai_server` 依赖（vLLM 或 sglang）
- 启动 `paddleocr genai_server` 暴露 OpenAI 风格 `/v1`

### 6.2 最小可行命令（官方 CLI 路径）

在“服务端环境”里：

```bash
python -m pip install "paddleocr[doc-parser]"
paddleocr install_genai_server_deps vllm
paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8118
```

客户端配置对应：

```yaml
external_models:
  paddleocr_vl:
    vl_rec_backend: "vllm-server"
    vl_rec_server_url: "http://127.0.0.1:8118/v1"
```

### 6.3 是否需要和 CodeVision 同进程

不需要。推荐分开终端：

- 终端 A：启动 OCR 服务
- 终端 B：运行 CodeVision / demo 调用 `_call_ocr_assist`

---

## 7. 无 MLLM 条件下如何分阶段测试

你现在不接 MLLM，建议按以下顺序：

### 阶段 A：纯基础链路

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/path/to/test.png \
  --disable-external \
  --out-dir outputs/code_image_tool_demo_smoke
```

说明：`--disable-external` 会关闭外部模型 worker，仅验证图像代码执行基础能力。

### 阶段 B：仅 Grounded-SAM-2

1. 先确保 `grounded_sam2_local_demo.py` 可跑  
2. 再运行你项目 demo，并传入 Grounded 绝对路径参数

### 阶段 C：仅 PaddleOCR-VL

1. 先确认服务端可访问：`curl http://127.0.0.1:8118/v1/models`  
2. 再跑项目 demo 里的 OCR case

### 阶段 D：二者串联

目标样例：先 `_call_focus()` 再 `_call_ocr_assist(image_obj=...)`，确认链式结果。

---

## 8. 常见问题与排障

1. `ModuleNotFoundError: ray`  
   说明基础依赖未装全，先补装 `requirements-runtime.txt`。

2. Grounded 路径报错（找不到 checkpoint/config）  
   基本是相对路径问题，统一改绝对路径。

3. GPU OOM（3060）  
   先只跑一个模型，优先小模型；不要同时做 OCR 服务高负载 + Grounded 推理压测。

4. 配置改了却像没生效  
   当前 external worker 使用固定 name + `get_if_exists=True`。  
   修改配置后建议：
   - 更换 `external_model_worker_name`，或
   - 重启 Ray 进程。

---

## 9. 你现在可以直接执行的最小行动清单

1. 在 WSL 建 `cvtool` 环境并安装基础依赖。  
2. 跑 `--disable-external`，确认基础链路。  
3. 克隆并安装 Grounded-SAM-2，先跑官方 local demo。  
4. 把 Grounded 路径改成绝对路径，回到 CodeVision 验证 `_call_focus`。  
5. 单独起 PaddleOCR-VL 服务，改 `vl_rec_server_url`，验证 `_call_ocr_assist`。  
6. 最后做两者串联。

---

## 参考资料（官方）

- Grounded-SAM-2 GitHub：<https://github.com/IDEA-Research/Grounded-SAM-2>
- PaddleOCR-VL 最新使用教程：<https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html>
- PaddleX 中 PaddleOCR-VL（含 server/client 配置）：<https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.html>

