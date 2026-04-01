# CodeImageTool 本地运行指引（GroundedSAM2 + PaddleOCR-VL）

## 1. 当前推荐架构
- `GroundedSAM2` 走 Ray actor（`runtime_env: cvtool`，占 GPU）。
- `PaddleOCR-VL` 走独立服务（`paddleocr genai_server`，占 GPU）。
- `paddleocr_vl` actor 只做调用与封装（`runtime_env: ocr`，`num_gpus: 0`）。

## 2. 已固化到 YAML 的配置
文件：[code_image_tool_config.yaml](D:/sdu/ToolVision/CodeVision/recipe/codevision/config/code_image_tool_config.yaml)

- `external_worker_mode: "split"`
- `external_workers.grounded_sam2.runtime_env: "cvtool"`
- `external_workers.paddleocr_vl.runtime_env: "ocr"`
- `external_models.paddleocr_vl.vl_rec_backend: "vllm-server"`
- `external_models.paddleocr_vl.vl_rec_server_url: "http://127.0.0.1:8118/v1"`

## 3. 每次登录后先做（cvtool 终端）
```bash
conda activate cvtool
export RAY_DISABLE_DASHBOARD=1
export RAY_TMPDIR=/tmp/ray
mkdir -p /tmp/ray
ray stop --force
```

说明：
- `RAY_DISABLE_DASHBOARD=1`：避免 dashboard 相关启动问题。
- `RAY_TMPDIR=/tmp/ray`：避免路径权限/Windows 挂载路径问题。
- `ray stop --force`：清理旧 actor，确保新配置生效。

## 4. OCR 服务端启动（ocr 终端）
```bash
conda activate ocr
paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --host 0.0.0.0 --port 8118
```

## 5. Demo 最简运行命令（cvtool 终端）
```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image /mnt/d/sdu/ToolVision/CodeVision/vstar_bench_input2.png \
  --focus-prompt "car." \
  --out-dir outputs/Vstar_Test2
```

说明：
- 当前 demo 默认值已对齐：
  - `--external-worker-mode=split`
  - `--vl-rec-server-url=http://127.0.0.1:8118/v1`
  - `--grounded-sam2-worker-runtime-env=cvtool`
  - `--paddleocr-worker-runtime-env=ocr`

## 6. 必填参数
- `--image`：输入图像路径。

## 7. 建议显式填写（虽有默认）
- `--focus-prompt`：决定 Grounding 检测目标。
- `--out-dir`：结果输出目录，便于多轮对比。

## 8. 重要但已有默认值（通常不用每次传）
- `--device` 默认 `cuda`
- `--box-threshold` 默认 `0.35`
- `--text-threshold` 默认 `0.25`
- `--paddleocr-worker-num-gpus` 默认 `0.0`
- `--grounded-sam2-worker-num-gpus` 默认 `1.0`

## 9. worker name 是否每次都要换
- 不需要每次换，只要每次运行前执行 `ray stop --force`。
- 如果不执行 `ray stop --force`，旧 actor 可能被复用（`get_if_exists=True`），导致你以为配置改了但实际没生效。
- 如果你不想停 Ray，就改成新名字强制新 actor。

## 10. 常见故障快速判断
- OCR 报 `No module named paddleocr`：
  - 检查 `paddleocr_vl` actor 是否跑在 `runtime_env: ocr`。
- OCR 连不上服务：
  - 检查 `ocr` 终端服务是否在 `8118` 正常启动。
  - 检查 `vl_rec_server_url` 是否一致。
- GroundedSAM2 报 CUDA 不可用：
  - 检查 `cvtool` 环境 `torch.cuda.is_available()`。
  - 检查 Ray actor GPU 配额与当前机器 GPU 占用。
