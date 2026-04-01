# GroundedSAM2 服务化接入 CodeVision

日期：2026-03-27

## 1. 结论

Grounded-SAM-2 不像 PaddleOCR 那样有现成的官方 OCR-style HTTP API。

因此这里的做法是：

1. 保留 `CodeImageTool` 里的 GroundSAM helper 接口不变
2. 把当前本地持模推理逻辑迁到独立 `runner.py`
3. 自己包一层轻量 HTTP 服务
4. `CodeImageTool` 在 service 模式下改为调这个 GroundSAM HTTP client

## 2. 为什么不直接复用官方 repo 自带 backend

官方仓库里有 backend 和 Docker 文件，但那套主要是交互式视频 demo：

- `Grounded-SAM-2/demo/backend/server/app.py`
- `Grounded-SAM-2/backend.Dockerfile`

它的重点是：

- `/healthy`
- `/propagate_in_video`
- GraphQL

不是你当前 `CodeImageTool` 需要的这条能力链：

- `text_prompt -> grounding -> mask/crop/blur`

所以更稳的方式是：

- 推理核心复用官方 repo 的本地 image demo 写法
- 服务协议自己定义

## 3. 当前服务目录

GroundSAM2 现在拆到了：

```text
CodeVision/verl/external_services/groundedsam2/
  __init__.py
  client.py
  codec.py
  runner.py
  service_app.py
  service_launcher.py
```

职责如下：

- `runner.py`
  - 常驻持模
  - 负责 GroundingDINO + SAM2 推理
  - 负责 box/mask/dino_crop/blur_bg
- `service_app.py`
  - 提供 Flask HTTP 接口
- `client.py`
  - 给 `CodeImageTool` service mode 调用
- `codec.py`
  - 处理 base64 / PIL 编解码
- `service_launcher.py`
  - 启动服务入口

## 4. 服务接口

当前服务提供：

- `GET /healthy`
- `POST /infer`

请求体核心字段：

- `file`
- `operation`
- `text_prompt`
- `box_threshold`
- `text_threshold`
- `multimask_output`
- `mask_alpha`
- `draw_box_on_mask`
- `based_on`
- `detection_index`
- `max_crops`
- `padding`
- `blur_radius`

返回体统一为：

```json
{
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "images": ["<base64 image>"],
    "text": "...",
    "meta": {}
  }
}
```

## 5. 如何启动服务

推荐直接用独立启动脚本，这样不会经过 `verl.__init__`，也就不要求服务环境额外安装 `tensordict` 这类训练侧依赖。

在 Grounded-SAM-2 相关依赖已装好的环境里启动：

```bash
python scripts/launch_groundedsam2_service.py \
  --host 0.0.0.0 \
  --port 8081 \
  --device cuda \
  --sam2-checkpoint ../Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-model-config ../Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --grounding-dino-config ../Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounding-dino-checkpoint ../Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth
```

这个服务环境至少还需要 HTTP 层依赖，例如 `flask` 和 `Pillow`。

另外，Grounded-SAM-2 里这份 GroundingDINO 代码与 `transformers 5.x` 不兼容。
建议在服务环境里固定：

```bash
pip install "transformers==4.33.2"
```

如果已经装过较新的版本，先降级再重启服务。

如果你已经在完整 `verl` 环境里，也可以继续用：

```bash
python -m verl.external_services.groundedsam2.service_launcher ...
```

但这条命令会先导入 `verl` 包本身，因此服务环境需要能满足 `verl` 的顶层依赖。

如果你想单独占卡，优先用 `CUDA_VISIBLE_DEVICES`；设备参数请写 `cuda` 或 `cuda:<index>`，不要写 `gpu:<index>`：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/launch_groundedsam2_service.py --device cuda ...
```

## 6. CodeVision 怎么切到 service 模式

配置片段：

```yaml
external_call_mode: "service"

external_services:
  grounded_sam2:
    base_url: "http://127.0.0.1:8081"
    request_timeout: 180
```

这时：

- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`

都会通过 `GroundedSAM2ServiceAdapter` -> `GroundedSAM2HTTPClient` 走 HTTP。

旧 worker 路径仍然保留，方便回退。

## 7. 快速联调

可以直接用 demo 脚本验证：

```bash
python recipe/codevision/demo_code_image_tool_external.py \
  --image <your_image> \
  --cases ground_box,sam_mask \
  --external-call-mode service \
  --grounded-sam2-base-url http://127.0.0.1:8081
```

如果 OCR 服务还没起，不要带 `ocr_assist` case。

## 8. 推荐后续顺序

1. 先验证 `ground_box`
2. 再验证 `sam_mask`
3. 再验证 `dino_crop`
4. 再验证 `blur_bg`
5. 全部稳定后，再考虑删旧 `GroundedSAM2ModelWorker`
