# DLC 工具服务 Runbook

本文档记录 2026-06-15 当前可用的 DLC 工具服务，以及后续 rollout DLC job 应该如何调用。

## 当前状态

**重点：当前可用的工具 DLC job 是这个。**

```text
Job ID:      dlcwv66tm4r5zxyp
Job name:    cv-tool-services-2gpu-v3
Workspace:   240810 / pai_a100_workspace
Resource:    quotaev2tl4w6aw0 / a100
Pod IP:      172.17.2.38
Status:      Running
GPU:         2
Ports:       18080-18093
```

已经验证：

```text
18080-18083 open
18090-18093 open

replica 0 warmup ok: OCR / GroundedSAM2 / Depth / CountGD
replica 1 warmup ok: OCR / GroundedSAM2 / Depth / CountGD
```

## 两组工具 URL

**重点：后续 rollout job 不要再用 DSW 工具 IP，直接用下面这个 DLC pod IP。**

Replica 0:

```bash
export OCR_BASE_URL=http://172.17.2.38:18080
export GROUNDEDSAM2_BASE_URL=http://172.17.2.38:18081
export DEPTH_BASE_URL=http://172.17.2.38:18082
export COUNTGD_BASE_URL=http://172.17.2.38:18083
```

Replica 1:

```bash
export OCR_BASE_URL=http://172.17.2.38:18090
export GROUNDEDSAM2_BASE_URL=http://172.17.2.38:18091
export DEPTH_BASE_URL=http://172.17.2.38:18092
export COUNTGD_BASE_URL=http://172.17.2.38:18093
```

**重点：一个 rollout DLC job 尽量只打一组工具。**

建议：

```text
shard A -> replica 0 / 18080-18083
shard B -> replica 1 / 18090-18093
```

不要两个大 shard 同时打同一组端口，除非确认工具延迟可接受。

## 使用前检查

**重点：每次启动大 rollout 前先检查 job 还在 Running。Pod IP 不是永久地址。**

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

dlc_pai get job dlcwv66tm4r5zxyp \
  -w 240810 \
  --show_detail \
  -r cn-wulanchabu \
  -e pai-dlc.cn-wulanchabu.aliyuncs.com
```

如果 pod 重启或 job 重提，重新拿 URL：

```bash
bash scripts/dlc_tool_urls_from_job.sh <TOOL_DLC_JOB_ID>
```

检查端口：

```bash
TOOL_IP=172.17.2.38

for p in 18080 18081 18082 18083 18090 18091 18092 18093; do
  timeout 2 bash -lc "</dev/tcp/${TOOL_IP}/${p}" && echo "$p open" || echo "$p closed"
done
```

Warmup replica 0:

```bash
python3 scripts/warmup_external_services.py all \
  --ocr-host 172.17.2.38 --ocr-port 18080 \
  --groundedsam2-host 172.17.2.38 --groundedsam2-port 18081 \
  --depth-host 172.17.2.38 --depth-port 18082 \
  --countgd-host 172.17.2.38 --countgd-port 18083 \
  --timeout-s 90
```

Warmup replica 1:

```bash
python3 scripts/warmup_external_services.py all \
  --ocr-host 172.17.2.38 --ocr-port 18090 \
  --groundedsam2-host 172.17.2.38 --groundedsam2-port 18091 \
  --depth-host 172.17.2.38 --depth-port 18092 \
  --countgd-host 172.17.2.38 --countgd-port 18093 \
  --timeout-s 90
```

## 提交 rollout job 示例

Replica 0 示例：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://172.17.2.38:18080 \
GROUNDEDSAM2_BASE_URL=http://172.17.2.38:18081 \
DEPTH_BASE_URL=http://172.17.2.38:18082 \
COUNTGD_BASE_URL=http://172.17.2.38:18083 \
EVAL_PARQUET=/path/to/shard.parquet \
EXP_NAME=mut_rollout8_stream_shardXX_t0p7 \
JOB_NAME=cv-mut-rollout8-stream-XX \
STREAM_VALIDATION_DUMP=True \
bash scripts/submit_dlc_toolvision_mut_rollout8.sh
```

Replica 1 示例：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://172.17.2.38:18090 \
GROUNDEDSAM2_BASE_URL=http://172.17.2.38:18091 \
DEPTH_BASE_URL=http://172.17.2.38:18092 \
COUNTGD_BASE_URL=http://172.17.2.38:18093 \
EVAL_PARQUET=/path/to/shard.parquet \
EXP_NAME=mut_rollout8_stream_shardYY_t0p7 \
JOB_NAME=cv-mut-rollout8-stream-YY \
STREAM_VALIDATION_DUMP=True \
bash scripts/submit_dlc_toolvision_mut_rollout8.sh
```

## 重新部署工具服务

**重点：这个服务不是永久平台服务。DLC job 停掉、失败、pod 重建后，需要重新确认 IP。**

提交命令：

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

JOB_NAME=cv-tool-services-2gpu-v4 \
TOOL_REPLICA_COUNT=2 \
TOOL_REPLICA_GPUS=0,1 \
WORKER_GPU=2 \
WORKER_CPU=32 \
WORKER_MEMORY=300Gi \
WORKER_SHARED_MEMORY=300Gi \
bash scripts/submit_dlc_tool_services.sh
```

脚本默认参数：

```text
WORKSPACE_ID=240810
RESOURCE_ID=quotaev2tl4w6aw0
VPC_ID=vpc-0jl5rpw5qokp6p2ettip6
SWITCH_ID=vsw-0jlmr9rjzed093yr9c0kz
SECURITY_GROUP_ID=sg-0jl0pd5qaerdj75wmred
customPortList=18080-18093
SKIP_WARMUP=1
```

`SKIP_WARMUP=1` 是故意的：DLC job 先把两组端口都拉起来，warmup 由我们拿到 IP 后手动做。

## 关键区别

**DLC 工具服务和本地 DSW 工具服务的核心区别：**

```text
DSW 本地工具:
  - 跟 DSW 生命周期绑定
  - 通常只有一组端口 18080-18083
  - 资源少，可能成为 rollout bottleneck

DLC 工具服务:
  - 跟 DLC job 生命周期绑定
  - 当前有两组端口 18080-18083 / 18090-18093
  - 后续 DLC rollout job 通过 pod IP 调用
  - pod IP 可能变化，必须在使用前确认
```

底层工具实现没有换，仍然走：

```bash
scripts/launch_external_services.sh
```

所以 OCR / GroundedSAM2 / Depth / CountGD 的默认 checkpoint、threshold、pipeline 和本地部署一致。

## 已踩过的坑

**重点：`requirements.txt not found` 不是失败。**

日志里看到：

```text
WARN: ./requirements.txt not found, skip installing requirements.
```

可以忽略。依赖来自镜像和已有 conda env。

**重点：`pai-common/alpine:3.10-multi-plat` 不是我们的 worker 镜像。**

如果 EnvPreparing 阶段报：

```text
Failed to pull image pai-common/alpine:3.10-multi-plat
```

这是 PAI 平台内部辅助镜像拉取失败，不是 `WORKER_IMAGE` 写错。我们的 worker 镜像仍是：

```text
dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/torcheasyrec:1.1.0-pytorch2.10.0-gpu-py311-cu129-ubuntu22.04
```

这种情况通常直接重提一次。

**重点：不要在工具 DLC 启动过程中改它正在执行的脚本。**

现在 submit 脚本已经规避这个问题：启动时会把 entrypoint 复制到 `/tmp/dlc_tools_entrypoint.sh` 再执行，避免 CPFS 文件运行中被改动导致 shell EOF。

## 判断是否可用

只要满足这三条，就可以给 rollout job 用：

```text
1. DLC job Status=Running
2. 18080-18083 和 18090-18093 都 open
3. 两组 warmup_external_services.py all 都 ok
```

当前 `dlcwv66tm4r5zxyp / 172.17.2.38` 已满足这三条。
