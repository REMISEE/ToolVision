# DLC Tool Services

This note records how to host CodeVision external tools in a long-running DLC job and let later rollout DLC jobs call them by HTTP.

## Status

The current PAI setup verified from DSW is:

- Region: `cn-wulanchabu`
- DLC endpoint: `pai-dlc.cn-wulanchabu.aliyuncs.com`
- DLC CLI: `dlc_pai` or `/etc/dsw/runtime/export_bin/dlc`
- Workspace: `240810` (`pai_a100_workspace` / `Innovator_Coder`)
- Resource quota: `quotaev2tl4w6aw0` (`a100`)
- Aliyun resource group: `rg-aek7nnd27i6hanq`
- VPC: `vpc-0jl5rpw5qokp6p2ettip6`
- VSwitch: `vsw-0jlmr9rjzed093yr9c0kz`
- Security group: `sg-0jl0pd5qaerdj75wmred`
- CPFS mount: `cpfs://cpfs-298fffb575a502fe.cn-wulanchabu/ptc-29f47d9393ad2b16/exp-29f2869e7d984aa6/::/mnt/cpfs`

`rg-aek7nnd27i6hanq` is the Alibaba Cloud resource group id. The current `dlc submit pytorchjob` commands do not pass it directly; the effective scheduling target is controlled by `WORKSPACE_ID` and `RESOURCE_ID`.

The submit script is:

```bash
scripts/submit_dlc_tool_services.sh
```

The URL helper is:

```bash
scripts/dlc_tool_urls_from_job.sh
```

## Persistence

This is not a permanent platform service. It is persistent only while the DLC job is `Running`.

- If the DLC job is running, the tool processes stay alive via `scripts/dlc_tools_entrypoint.sh`.
- If the DLC job fails, is stopped, or the pod is recreated, the services stop.
- If the pod is recreated, its pod IP can change. Re-run `scripts/dlc_tool_urls_from_job.sh <job_id>` and update rollout job URLs.
- Logs and pid files are written under CPFS, so they persist after the container exits:
  - `outputs/dlc_tool_services/replica_0/logs`
  - `outputs/dlc_tool_services/replica_1/logs`

For rollout jobs, treat the tool DLC job as an external dependency: start it first, warm it up, then submit rollout jobs using its current pod IP and ports.

The DLC entrypoint defaults `SKIP_WARMUP=1` during startup. This is intentional: replicas are started serially, and internal warmup can block replica 1 behind a slow replica 0 warmup. Start the DLC job first, then run the explicit warmup commands below from DSW.

The submit script copies `scripts/dlc_tools_entrypoint.sh` to `/tmp/dlc_tools_entrypoint.sh` inside the pod before executing it. This avoids a running DLC job reading a partially edited CPFS script if the repo is patched while tools are starting.

## Default Deployment

Submit one DLC job with two GPUs and two independent tool replicas:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

JOB_NAME=cv-tool-services-2gpu \
TOOL_REPLICA_COUNT=2 \
TOOL_REPLICA_GPUS=0,1 \
WORKER_GPU=2 \
WORKER_CPU=32 \
WORKER_MEMORY=300Gi \
WORKER_SHARED_MEMORY=300Gi \
bash scripts/submit_dlc_tool_services.sh
```

The script exposes these ports through DLC service settings:

```text
replica 0: 18080-18083
replica 1: 18090-18093
```

Internally each replica runs all four tools on one GPU:

| Replica | GPU | OCR | GroundedSAM2 | Depth | CountGD |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 18080 | 18081 | 18082 | 18083 |
| 1 | 1 | 18090 | 18091 | 18092 | 18093 |

The DLC submit settings include:

```text
createSvcForAllWorkers=true,customPortList=18080-18093
```

This exposes the listed ports inside the PAI VPC. It does not make the tools a public internet API.

## Get Tool URLs

After submission, get the job id from DLC output. Then print usable environment variables:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

bash scripts/dlc_tool_urls_from_job.sh <TOOL_DLC_JOB_ID>
```

The output will look like:

```bash
# replica 0
export OCR_BASE_URL=http://<tool_ip>:18080
export GROUNDEDSAM2_BASE_URL=http://<tool_ip>:18081
export DEPTH_BASE_URL=http://<tool_ip>:18082
export COUNTGD_BASE_URL=http://<tool_ip>:18083

# replica 1
export OCR_BASE_URL=http://<tool_ip>:18090
export GROUNDEDSAM2_BASE_URL=http://<tool_ip>:18091
export DEPTH_BASE_URL=http://<tool_ip>:18092
export COUNTGD_BASE_URL=http://<tool_ip>:18093
```

Use one replica per rollout job when possible. The system does not automatically load-balance across replicas.

## Warmup

Verify replica 0:

```bash
python3 scripts/warmup_external_services.py all \
  --ocr-host <tool_ip> --ocr-port 18080 \
  --groundedsam2-host <tool_ip> --groundedsam2-port 18081 \
  --depth-host <tool_ip> --depth-port 18082 \
  --countgd-host <tool_ip> --countgd-port 18083 \
  --timeout-s 60
```

Verify replica 1:

```bash
python3 scripts/warmup_external_services.py all \
  --ocr-host <tool_ip> --ocr-port 18090 \
  --groundedsam2-host <tool_ip> --groundedsam2-port 18091 \
  --depth-host <tool_ip> --depth-port 18092 \
  --countgd-host <tool_ip> --countgd-port 18093 \
  --timeout-s 60
```

## Use From Rollout DLC Jobs

Pass one replica's URLs explicitly when submitting a rollout job:

```bash
OCR_BASE_URL=http://<tool_ip>:18080 \
GROUNDEDSAM2_BASE_URL=http://<tool_ip>:18081 \
DEPTH_BASE_URL=http://<tool_ip>:18082 \
COUNTGD_BASE_URL=http://<tool_ip>:18083 \
EVAL_PARQUET=/path/to/shard.parquet \
EXP_NAME=mut_rollout8_stream_shardXX_t0p7 \
JOB_NAME=cv-mut-rollout8-stream-XX \
STREAM_VALIDATION_DUMP=True \
bash scripts/submit_dlc_toolvision_mut_rollout8.sh
```

For another rollout job, use replica 1 ports:

```bash
OCR_BASE_URL=http://<tool_ip>:18090 \
GROUNDEDSAM2_BASE_URL=http://<tool_ip>:18091 \
DEPTH_BASE_URL=http://<tool_ip>:18092 \
COUNTGD_BASE_URL=http://<tool_ip>:18093 \
...
```

## Difference From DSW Local Deployment

Both paths use the same underlying service launcher:

```bash
scripts/launch_external_services.sh
```

That means the tool implementations, default checkpoints, thresholds, and service behavior are the same unless environment variables override them.

Main differences:

| Item | DSW local | DLC tool job |
|---|---|---|
| Entry script | `scripts/start_dsw_tool_services.sh` | `scripts/dlc_tools_entrypoint.sh` |
| Default replicas | 1 | 2 |
| Default GPUs | all tools share GPU 0 | replica 0 uses GPU 0, replica 1 uses GPU 1 |
| Ports | `18080-18083` | `18080-18083` and `18090-18093` |
| Logs | `outputs/dsw_tool_services/logs` | `outputs/dlc_tool_services/replica_<id>/logs` |
| PID files | `outputs/dsw_tool_services/pids` | `outputs/dlc_tool_services/replica_<id>/pids` |
| Lifetime | tied to DSW instance/processes | tied to DLC job lifetime |
| Access | DSW/DLC can call DSW IP if network permits | later DLC jobs call tool pod IP inside PAI VPC |
| Scaling | usually one shared tool group | multiple port groups in one DLC job |

The important behavior difference is lifetime and address stability:

- DSW IP is convenient while that DSW is alive, but small DSW resources can bottleneck.
- DLC tool service has more resources and is cleaner for production rollout, but the pod IP must be queried after the job starts and can change if the job restarts.

## Parameter Consistency

The tool-level defaults are inherited from `scripts/launch_external_services.sh` in both modes:

- OCR pipeline: `recipe/codevision/config/ocr_no_doc_preprocessor.yaml`
- GroundedSAM2 checkpoint/config defaults under `ToolVision/Grounded-SAM-2`
- Depth checkpoint default: `checkpoints/depth_pro.pt`
- CountGD checkpoint default: `checkpoints/checkpoint_fsc147_best.pth`
- GroundedSAM2 default thresholds:
  - box threshold: `0.35`
  - text threshold: `0.25`
- CountGD default confidence threshold: `0.23`

The DLC script intentionally changes only deployment shape:

- two replicas instead of one
- second port group starts at `18090`
- each replica pins all four tools to one GPU
- startup skips internal warmup by default; warmup is run explicitly after the pod IP is known
- service ports are exposed through DLC `customPortList`

So for model/tool behavior, it should match local DSW deployment. For throughput and concurrency, DLC has two independent tool groups and should handle two rollout jobs more cleanly.

## Notes

- If a rollout job fails to reach tools, first check that the tool DLC job is still `Running`.
- Re-run `scripts/dlc_tool_urls_from_job.sh <TOOL_DLC_JOB_ID>` if the pod may have restarted.
- Re-run warmup before starting large rollout shards.
- Avoid sending many rollout jobs to the same replica unless tool latency is acceptable.
