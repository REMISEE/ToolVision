# MUT v2 RL Runbook 2026-06-17

## Goal

Run GSPO on a cleaner MUT mixture:

- Stable MUT gets positive tool reward.
- Weak is split by no-tool counterfactual success.
- Regular gets a small tool-use cost instead of a negative MUT weight.
- Every 64-prompt training batch has fixed difficulty composition.

## Environment

Use:

```text
/mnt/cpfs/delinmao/envs/codevision_new
```

Do not use the old shared environment:

```text
/mnt/cpfs/delinmao/envs/codevision
```

The old environment was modified in place and fails the Qwen3-VL/verl import
chain with `AutoModelForVision2Seq`.  See
`docs/codevision_env_recovery_20260617.md` for the verification notes.

## Data

Input:

```text
outputs/analysis/mut_v1_20260616/mut_v1_train.parquet
```

Generated output:

```text
outputs/analysis/mut_v2_20260617/mut_v2_train_balanced.parquet
outputs/analysis/mut_v2_20260617/mut_v2_train_summary.json
```

Build command:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
python3 recipe/codevision/tools/build_mut_v2_train.py
```

Class split before balancing:

```text
mut                 8,464
weak_clean          2,825   # old weak with mut_v1_NTC == 0
hard_regular        3,661   # old weak with mut_v1_NTC >= 1
regular_9_15       10,000
total              24,950
```

Balanced-order parquet:

```text
rows_out           30,144
batch_size             64
blocks                471
```

Every consecutive 64 rows are one intended batch:

```text
regular_9_15          28
hard_regular          10
mut                   20
weak_clean             6
```

This requires:

```text
DATA_SHUFFLE=False
```

The builder oversamples some rows to fill balanced blocks.  It preserves the
original uid in `extra_info.mut_v2_original_uid` and writes a unique training
uid into `extra_info.uid`.

## Reward

Current mode:

```text
TOOL_REWARD_MODE=mut_clean
```

Formula:

```text
R_total = R_acc
        + 0.2 * R_protocol
        + mut_weight * R_mut
        - regular_tool_penalty * I(used_tool)
        - 0.05 * max(0, NumTurns - 6)
```

Where:

```text
R_mut = 1 if answer is correct AND used_tool AND tool_exec_error_count == 0 else 0
R_protocol = R_fmt * I(tool JSON is legal)
```

Class labels:

```text
mut             mut_weight=0.5   regular_tool_penalty=0.00
weak_clean      mut_weight=0.2   regular_tool_penalty=0.00
hard_regular    mut_weight=0.0   regular_tool_penalty=0.00
regular_9_15    mut_weight=0.0   regular_tool_penalty=0.05
```

Do not encode regular cost as a negative `mut_weight`.  Negative `mut_weight`
would only punish correct tool use, not all unnecessary tool use.

## Code Changes

Data builder:

```text
recipe/codevision/tools/build_mut_v2_train.py
```

Reward/logging:

```text
recipe/codevision/rewards/router.py
verl/experimental/agent_loop/agent_loop.py
verl/trainer/ppo/metric_utils.py
verl/trainer/ppo/ray_trainer.py
```

Launcher:

```text
recipe/codevision/qwen3_vl_gspo_direct.sh
scripts/submit_dlc_gspo_direct_full.sh
scripts/submit_dlc_gspo_mut_v2_t07_cap2048_mns32.sh
```

New metrics expected in future W&B runs:

```text
reward/R_protocol
reward/R_mut
reward/MutWeight
reward/P_regular_tool
reward/P_turn_overuse
reward/R_total
reward/NumTurns
```

Existing active DLC jobs will not hot-load these logging changes.

## Submit Command

Use one tool replica per RL job.  Example for tool job `dlc1xa5hqisrget5`
replica 0:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

export WANDB_API_KEY='<your_wandb_key>'
export OFFLINE_SFT_QWEN_BASE_URL='<your_offline_sft_qwen_base_url>'
export OFFLINE_SFT_QWEN_MODEL='<your_offline_sft_qwen_model>'
export OFFLINE_SFT_QWEN_API_KEY='<your_offline_sft_qwen_api_key>'

OCR_BASE_URL=http://172.17.0.142:18080 \
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18081 \
DEPTH_BASE_URL=http://172.17.0.142:18082 \
COUNTGD_BASE_URL=http://172.17.0.142:18083 \
JOB_NAME=cv-mutv2 \
EXP_NAME=mutv2 \
bash scripts/submit_dlc_gspo_mut_v2_t07_cap2048_mns32.sh
```

If these env vars are already exported in your shell, the shorter command is:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

OCR_BASE_URL=http://172.17.0.142:18080 \
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18081 \
DEPTH_BASE_URL=http://172.17.0.142:18082 \
COUNTGD_BASE_URL=http://172.17.0.142:18083 \
JOB_NAME=cv-mutv2 \
EXP_NAME=mutv2 \
bash scripts/submit_dlc_gspo_mut_v2_t07_cap2048_mns32.sh
```

Replica alternatives:

```bash
# replica 1
OCR_BASE_URL=http://172.17.0.142:18090
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18091
DEPTH_BASE_URL=http://172.17.0.142:18092
COUNTGD_BASE_URL=http://172.17.0.142:18093

# replica 2
OCR_BASE_URL=http://172.17.0.142:18100
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18101
DEPTH_BASE_URL=http://172.17.0.142:18102
COUNTGD_BASE_URL=http://172.17.0.142:18103

# replica 3
OCR_BASE_URL=http://172.17.0.142:18110
GROUNDEDSAM2_BASE_URL=http://172.17.0.142:18111
DEPTH_BASE_URL=http://172.17.0.142:18112
COUNTGD_BASE_URL=http://172.17.0.142:18113
```

## Expected Run Shape

Defaults in the v2 launcher:

```text
TRAIN_BSZ=64
N_RESP_PER_PROMPT=8
effective trajectories per step=512
MAX_NUM_SEQS=32
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.95
ROLLOUT_MAX_TOKENS_PER_TURN=2048
SAVE_FREQ=10
VAL_BEFORE_TRAIN=False
TEST_FREQ=-1
TOTAL_EPOCHS=1
DATA_SHUFFLE=False
```

With 30,144 rows and batch size 64:

```text
total train steps = 471
```

## Early Monitoring

Do not judge from aggregate `reward/R_acc` alone.  Watch:

```text
reward/R_acc
reward/R_total
reward/R_mut
reward/P_regular_tool
reward/NumTurns
reward/R_fmt
reward/P_turn_overuse
```

First stop conditions:

```text
R_fmt < 0.95 for several steps
NumTurns keeps rising beyond 2.5 without R_acc recovery
P_regular_tool rises but regular accuracy does not stabilize
tool execution errors spike
```

If regular tool use remains too high, next knob is:

```text
regular_9_15 regular_tool_penalty: 0.05 -> 0.08
```

If weak_clean is too noisy, next knob is:

```text
weak_clean mut_weight: 0.2 -> 0.1
```
