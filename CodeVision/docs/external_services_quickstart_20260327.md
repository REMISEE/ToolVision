# External Services Quickstart

日期：2026-03-27

cd /data/home/suchenghao/ToolVision/CodeVision

  OCR_PORT=8090 \
  OCR_DEVICE=gpu:0 \
  GROUNDEDSAM2_PORT=8085 \
  GROUNDEDSAM2_CUDA_VISIBLE_DEVICES=2 \
  DEPTH_PORT=8086 \
  DEPTH_CUDA_VISIBLE_DEVICES=3 \
  COUNTGD_PORT=8087 \
  COUNTGD_CUDA_VISIBLE_DEVICES=0 \
  bash scripts/launch_external_services.sh restart

  如果只起一个服务：

  OCR_PORT=8090 OCR_DEVICE=gpu:0 bash scripts/launch_external_services.sh start ocr
  GROUNDEDSAM2_PORT=8085 GROUNDEDSAM2_CUDA_VISIBLE_DEVICES=2 bash scripts/launch_external_services.sh start groundedsam2
  DEPTH_PORT=8086 DEPTH_CUDA_VISIBLE_DEVICES=3 bash scripts/launch_external_services.sh start depth
  COUNTGD_PORT=8087 COUNTGD_CUDA_VISIBLE_DEVICES=0 bash scripts/launch_external_services.sh start countgd
  
## 1. 当前结论

`CodeImageTool` 现在默认走 service 模式：

- OCR: `paddleocr` HTTP service
- GroundSAM: `groundedsam2` HTTP service
- Depth: `depth` HTTP service
- Count: `countgd` HTTP service

主配置文件：

- `recipe/codevision/config/code_image_tool_config.yaml`

其中旧 worker 配置仍然保留，但只作为回退路径。

不建议现在立刻删除旧 worker 段，原因是：

1. 现在已经验证通过的是主链路 smoke case
2. 更大规模训练 / eval 还没完整跑一轮
3. 保留回退路径的成本很低

建议做法是：

- 默认用 service
- worker 段先保留
- 等你确认线上/批量任务也稳定后，再删

## 2. 一键启动

新增统一启动脚本：

```bash
bash scripts/launch_external_services.sh start
```

常用命令：

```bash
bash scripts/launch_external_services.sh start
bash scripts/launch_external_services.sh status
bash scripts/launch_external_services.sh restart groundedsam2
bash scripts/launch_external_services.sh restart depth
bash scripts/launch_external_services.sh restart ocr
bash scripts/launch_external_services.sh restart countgd
bash scripts/launch_external_services.sh stop
```

日志与 PID 默认在：

- `outputs/service_logs/`
- `outputs/service_pids/`

## 3. 可直接写死的参数

这些一般可以直接在脚本或配置里写死：

- OCR host: `0.0.0.0`
- OCR port: `8080`
- OCR pipeline: `OCR`
- GroundSAM host: `0.0.0.0`
- GroundSAM port: `8081`
- Depth host: `0.0.0.0`
- Depth port: `8082`
- CountGD host: `0.0.0.0`
- CountGD port: `8083`
- GroundSAM `default_text_prompt`: `object.`
- GroundSAM `box_threshold`: `0.35`
- GroundSAM `text_threshold`: `0.25`
- Depth `default_text_prompt`: `object.`
- Depth `box_threshold`: `0.35`
- Depth `text_threshold`: `0.25`
- CountGD `default_confidence_thresh`: `0.23`
- CountGD `default_visualize`: `heatmap`
- GroundSAM checkpoint / config 路径
- Depth Pro checkpoint / root 路径
- Depth service should point to the existing `groundedsam2` HTTP service for `ground_depth`
- CountGD checkpoint / config 路径
- `request_timeout`: `180`

这些值更像部署默认值，不需要频繁改。

## 4. 建议保留为可调参数

这些更适合用环境变量覆盖，而不是彻底写死：

- `OCR_ENV`
- `GROUNDEDSAM2_ENV`
- `DEPTH_ENV`
- `COUNTGD_ENV`
- `OCR_DEVICE`
- `GROUNDEDSAM2_CUDA_VISIBLE_DEVICES`
- `GROUNDEDSAM2_DEVICE`
- `DEPTH_CUDA_VISIBLE_DEVICES`
- `DEPTH_DEVICE`
- `DEPTH_GROUNDEDSAM2_BASE_URL`
- `COUNTGD_CUDA_VISIBLE_DEVICES`
- `COUNTGD_DEVICE`
- GroundSAM 模型规模对应的 checkpoint / config
- Depth Pro 环境 / checkpoint / service base_url
- CountGD checkpoint / config / text encoder

原因：

- 环境名会随机器变化
- GPU 号会随机器/任务变化
- GroundSAM 模型大小可能切 tiny / small / large

## 5. 默认环境变量

统一启动脚本支持这些环境变量：

```bash
OCR_ENV=paddleocr
OCR_HOST=0.0.0.0
OCR_PORT=8080
OCR_PIPELINE=OCR
OCR_DEVICE=gpu:0
OCR_CUDA_VISIBLE_DEVICES=0
OCR_USE_HPIP=0
OCR_HPI_CONFIG=

GROUNDEDSAM2_ENV=groundedsam2
GROUNDEDSAM2_HOST=0.0.0.0
GROUNDEDSAM2_PORT=8081
GROUNDEDSAM2_DEVICE=cuda
GROUNDEDSAM2_CUDA_VISIBLE_DEVICES=1
GROUNDEDSAM2_DEFAULT_TEXT_PROMPT=object.
GROUNDEDSAM2_BOX_THRESHOLD=0.35
GROUNDEDSAM2_TEXT_THRESHOLD=0.25
GROUNDEDSAM2_SAM2_CHECKPOINT=../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt
GROUNDEDSAM2_SAM2_MODEL_CONFIG=../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
GROUNDEDSAM2_GDINO_CONFIG=../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
GROUNDEDSAM2_GDINO_CHECKPOINT=../Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth

DEPTH_ENV=depth-pro
DEPTH_HOST=0.0.0.0
DEPTH_PORT=8082
DEPTH_DEVICE=cuda
DEPTH_CUDA_VISIBLE_DEVICES=2
DEPTH_ROOT=../ml-depth-pro-main
DEPTH_CHECKPOINT_PATH=checkpoints/depth_pro.pt
DEPTH_CACHE_SIZE=8
DEPTH_REQUEST_TIMEOUT=180
DEPTH_DEFAULT_TEXT_PROMPT=object.
DEPTH_GROUNDEDSAM2_BASE_URL=http://127.0.0.1:8081
DEPTH_BOX_THRESHOLD=0.35
DEPTH_TEXT_THRESHOLD=0.25

COUNTGD_ENV=countgd
COUNTGD_HOST=0.0.0.0
COUNTGD_PORT=8083
COUNTGD_DEVICE=cuda
COUNTGD_CUDA_VISIBLE_DEVICES=3
COUNTGD_ROOT=CountGD
COUNTGD_CONFIG_PATH=config/cfg_fsc147_vit_b.py
COUNTGD_PRETRAIN_MODEL_PATH=checkpoints/checkpoint_fsc147_best.pth
COUNTGD_TEXT_ENCODER_TYPE=checkpoints/bert-base-uncased
COUNTGD_DEFAULT_CONFIDENCE_THRESH=0.23
COUNTGD_DEFAULT_VISUALIZE=heatmap
COUNTGD_HEATMAP_SIGMA=5.0
```

示例：把 OCR 固定到 GPU 0，GroundSAM 固定到 GPU 2，Depth 固定到 GPU 3，CountGD 固定到 GPU 1：

```bash
OCR_DEVICE=gpu:0 \
GROUNDEDSAM2_CUDA_VISIBLE_DEVICES=2 \
DEPTH_CUDA_VISIBLE_DEVICES=3 \
COUNTGD_CUDA_VISIBLE_DEVICES=1 \
bash scripts/launch_external_services.sh restart
```

## 6. 推荐下一步

1. 先用统一脚本起服务
2. 再用 `codevision` 环境跑 OCR-only 和 external demo
3. 如果训练/评测链路也稳定，再删旧 worker 段
