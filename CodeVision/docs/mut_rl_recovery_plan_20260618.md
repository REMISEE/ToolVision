# MUT RL Recovery Plan 2026-06-18

## Files Used

- `saves/ToolVisionRL/mutv1_a/rollout_generations/*.jsonl`
- `saves/ToolVisionRL/mutv2/rollout_generations/*.jsonl`
- `wandb/run-20260616_124315-mkb0mhdw/files/output.log` (`mutv1_a`)
- `wandb/run-20260617_075946-l3h2geji/files/output.log` (`mutv2`)
- `saves/ToolVisionRL/mutv1_a/tool_config.runtime.yaml`
- `saves/ToolVisionRL/mutv2/tool_config.runtime.yaml`
- Analysis outputs in `outputs/analysis/mut_runs_20260618/`

## What Happened

`mutv1_a` used tool replica 0:

```text
http://172.17.0.142:18080-18083
```

`mutv2` used tool replica 2:

```text
http://172.17.0.142:18100-18103
```

`mutv1_a` has two clear tool connection-failure windows:

```text
step 76-106   2026-06-17 13:38-18:38 UTC   2026-06-17 21:38-2026-06-18 02:38 CST
step 141-163  2026-06-18 03:47-08:00 UTC   2026-06-18 11:47-16:00 CST
```

At step 76 the output contains:

```text
HTTPConnectionPool(host='172.17.0.142', port=18081/18082/18083)
Failed to establish a new connection: [Errno 111] Connection refused
```

So the step-75 neighborhood issue is mostly:

- step 75 itself: no connection-refused evidence; likely batch/source difficulty and sampling noise.
- step 76 onward: real tool outage; this matches the reported around-9pm China time failure.

`NumTurns` did not become zero because the policy still emitted valid `<tool_call>` blocks. The service call failed after the tool call was made, then the agent received an error observation and often continued to answer. `valid_tool_call_count` counts syntactically valid tool calls, not successful HTTP tool execution.

`mutv2` is not mainly a tool-outage failure. It collapsed into no-tool behavior:

```text
step 1:   NumTurns 1.31, R_mut 0.303
step 40:  NumTurns 0.08, R_mut 0.018
step 75+: NumTurns approximately 0, R_mut approximately 0
```

The early tool outage likely accelerated the no-tool collapse, but after step 60 the main problem is policy/reward dynamics, not tool connectivity.

## Checkpoints

Both runs saved every 10 steps.

```text
mutv1_a latest checkpoint: global_step_160
mutv2   latest checkpoint: global_step_170
```

Recommendation:

- Do not resume `mutv2` as-is.
- For `mutv1_a`, the scientifically cleaner resume point is `global_step_70`, before the first large tool outage.
- Resuming from `global_step_160` is technically possible but keeps weights trained through two tool-failure windows.

## Batch Size 128

The 16-GPU jobs did not look GPU-memory bound. Observed W&B memory fields:

```text
max_memory_allocated_gb: about 115-119
max_memory_reserved_gb: about 124
cpu_memory_used_gb: about 500-690
```

The risk for `TRAIN_BSZ=128` is not primarily GPU memory. It doubles prompt count per step, so it increases:

- generated trajectories per step (`128 * rollout8 = 1024`)
- Ray/CPU memory pressure
- tool-server request pressure
- per-step wall time

Recommendation: run a 20-step probe first:

```text
TRAIN_BSZ=128
TRAIN_MINI_BSZ=32
TOTAL_TRAINING_STEPS=20
```

## Tool DLC Deployment On Specific Nodes

`dlc_pai submit pytorchjob --help` exposes:

```text
--allow_nodes nodeA,nodeB
--deny_nodes nodeC,nodeD
```

`scripts/submit_dlc_tool_services.sh` now supports:

```text
ALLOW_NODES=<comma-separated-node-names>
DENY_NODES=<comma-separated-node-names>
```

Submit a 2-GPU / 4-replica tool service job on a specified node:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

ALLOW_NODES=<NODE_NAME> \
JOB_NAME=cv-tool-services-2gpu-4replica-0618 \
TOOL_REPLICA_COUNT=4 \
TOOL_REPLICA_GPUS=0,0,1,1 \
TOOL_PORT_BASE=18080 \
TOOL_PORT_STRIDE=10 \
WORKER_GPU=2 \
WORKER_CPU=32 \
WORKER_MEMORY=300Gi \
WORKER_SHARED_MEMORY=300Gi \
bash scripts/submit_dlc_tool_services.sh
```

After it starts:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

TOOL_REPLICA_COUNT=4 \
bash scripts/dlc_tool_urls_from_job.sh <TOOL_DLC_JOB_ID>
```

Warm up all four replicas:

```bash
TOOL_IP=<TOOL_DLC_POD_IP>

for base in 18080 18090 18100 18110; do
  python3 scripts/warmup_external_services.py all \
    --ocr-host "${TOOL_IP}" --ocr-port "${base}" \
    --groundedsam2-host "${TOOL_IP}" --groundedsam2-port "$((base + 1))" \
    --depth-host "${TOOL_IP}" --depth-port "$((base + 2))" \
    --countgd-host "${TOOL_IP}" --countgd-port "$((base + 3))" \
    --timeout-s 180
done
```

## Training Commands

Before submitting training jobs, keep existing secret exports in the shell:

```bash
export WANDB_API_KEY=...
export LLM_JUDGE_BASE_URL=...
export LLM_JUDGE_MODEL_NAME=...
export LLM_JUDGE_API_KEY=...
```

The launcher also accepts `OFFLINE_SFT_QWEN_BASE_URL`, `OFFLINE_SFT_QWEN_MODEL`, and `OFFLINE_SFT_QWEN_API_KEY`; `submit_dlc_gspo_direct_full.sh` maps them to `LLM_JUDGE_*`.

### 1. Recommended `mutv1` Clean Resume From Step 70

Use one dedicated tool replica. Example below uses replica 0.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://<TOOL_IP>:18080 \
GROUNDEDSAM2_BASE_URL=http://<TOOL_IP>:18081 \
DEPTH_BASE_URL=http://<TOOL_IP>:18082 \
COUNTGD_BASE_URL=http://<TOOL_IP>:18083 \
JOB_NAME=cv-mut1-resume70 \
EXP_NAME=mutv1_resume70_0618 \
RESUME_MODE=resume_path \
RESUME_FROM_PATH=./saves/ToolVisionRL/mutv1_a/global_step_70 \
TRAIN_BSZ=64 \
TRAIN_MINI_BSZ=32 \
MAX_NUM_SEQS=32 \
SAVE_FREQ=10 \
bash scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh
```

### 2. Technical Resume From Latest Step 160

This keeps the noisy updates from tool-failure windows, so use only if you want continuity over cleanliness.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://<TOOL_IP>:18080 \
GROUNDEDSAM2_BASE_URL=http://<TOOL_IP>:18081 \
DEPTH_BASE_URL=http://<TOOL_IP>:18082 \
COUNTGD_BASE_URL=http://<TOOL_IP>:18083 \
JOB_NAME=cv-mut1-resume160 \
EXP_NAME=mutv1_a \
RESUME_MODE=auto \
TRAIN_BSZ=64 \
TRAIN_MINI_BSZ=32 \
MAX_NUM_SEQS=32 \
SAVE_FREQ=10 \
bash scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh
```

### 3. `mut1-128bs` Probe

Use a new experiment name and a dedicated tool replica.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://<TOOL_IP>:18090 \
GROUNDEDSAM2_BASE_URL=http://<TOOL_IP>:18091 \
DEPTH_BASE_URL=http://<TOOL_IP>:18092 \
COUNTGD_BASE_URL=http://<TOOL_IP>:18093 \
JOB_NAME=cv-mut1-128-probe \
EXP_NAME=mutv1_128bs_probe20_0618 \
RESUME_MODE=disable \
TRAIN_BSZ=128 \
TRAIN_MINI_BSZ=32 \
TOTAL_TRAINING_STEPS=20 \
MAX_NUM_SEQS=32 \
SAVE_FREQ=10 \
bash scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh
```

If stable for 20 steps, rerun without `TOTAL_TRAINING_STEPS=20`:

```bash
EXP_NAME=mutv1_128bs_0618
```

### 4. `mut1-v04`

The v04 SFT model path is:

```text
/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04
```

Do not resume v03 checkpoints into v04. Start from v04 SFT.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://<TOOL_IP>:18100 \
GROUNDEDSAM2_BASE_URL=http://<TOOL_IP>:18101 \
DEPTH_BASE_URL=http://<TOOL_IP>:18102 \
COUNTGD_BASE_URL=http://<TOOL_IP>:18103 \
JOB_NAME=cv-mut1-v04 \
EXP_NAME=mutv1_v04_0618 \
MODEL_PATH=/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v04 \
RESUME_MODE=disable \
TRAIN_BSZ=64 \
TRAIN_MINI_BSZ=32 \
MAX_NUM_SEQS=32 \
SAVE_FREQ=10 \
bash scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh
```

## Recommendation

Immediate order:

1. Redeploy tool DLC on the specified node with 2 GPU / 4 replicas.
2. Warm up all replicas.
3. Run `mutv1_resume70_0618` first.
4. Run `mutv1_128bs_probe20_0618` as a short probe, not full.
5. Run `mutv1_v04_0618` after v03 resume/probe starts, using a separate tool replica.

Do not continue `mutv2` without changing reward/initialization. It has already learned the no-tool shortcut.
