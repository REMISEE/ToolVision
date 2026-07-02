# Step Answerability Reward

This is an incremental RL reward path. Existing reward modes are unchanged.

## Mode

Use `TOOL_REWARD_MODE=mut_clean_step_v1` to enable the new aggregation path.
The step judge itself is still off by default:

```bash
STEP_REWARD_ENABLE=False
```

When the judge service is deployed, turn it on with an OpenAI-compatible endpoint:

```bash
TOOL_REWARD_MODE=mut_clean_step_v1 \
STEP_REWARD_ENABLE=True \
STEP_JUDGE_BASE_URL=http://<ip>:<port> \
STEP_JUDGE_MODEL=<model-name> \
STEP_JUDGE_API_KEY_ENV=STEP_JUDGE_API_KEY
```

`STEP_JUDGE_BASE_URL` may be the service root, `/v1`, or the full
`/v1/chat/completions` URL.

The current default judge prompt is `STEP_JUDGE_PROMPT_MODE=context`. It sends
the rollout context up to the scored state: original prompt, images, assistant
thinking/tool calls, tool responses, and a text copy of the tool schema. The
judge is instructed not to call tools and to answer directly in
`<answer>...</answer>`. The old state-snapshot prompt can still be used with:

```bash
STEP_JUDGE_PROMPT_MODE=snapshot
```

Context limits are configurable:

```bash
STEP_JUDGE_MAX_CONTEXT_CHARS=60000
STEP_JUDGE_MAX_IMAGES=8
STEP_JUDGE_MAX_OBSERVATION_CHARS=12000
```

## Reward

Base reward is the current MUT-clean reward. Current 128bs MUT-v1 data has
`P_regular_tool = 0`; the field is only kept for old parquet compatibility.

```text
R_base = R_acc + 0.2 * R_protocol + mut_weight * R_mut - P_regular_tool - 0.05 * max(0, NumTurns - 6)
```

The step judge scores answerability before tools and after each tool step:

```text
V0 = score(original context, direct-answer request)
Vt = score(rollout context after tool step t, direct-answer request)
gain_t = max(0, Vt - best_previous - tau)
R_step_raw = min(cap, sum(gain_t over valid steps))
R_step = step_weight * step_gate * R_step_raw
R_total = R_base + R_step
```

The default `step_gate` is `1.0`; set `STEP_REWARD_USE_MUT_WEIGHT=True` to use
the older `mut_weight` gate.

Defaults:

```text
step_weight = 0.2
tau = 0.1
cap = 0.5
step_judge_num_judgments = 1
step_judge_aggregation = mean
step_judge_prompt_mode = context
```

Invalid tool steps and judge failures get no positive step gain. Repeating many
small steps should not accumulate reward unless a step improves answerability
above the previous best by more than `tau`; the total step reward is capped.

## Logged Fields

Rollout samples include:

```text
step_answerability_v0
step_answerability_scores
step_answerability_valid
step_answerability_records
R_base_total
R_step_raw
R_step
StepScoredCount
StepValidCount
StepBestScore
```

## DLC Judge Service

The recommended deployment is a separate long-running DLC job, same shape as
the existing tool-service job. The RL job only receives `STEP_JUDGE_BASE_URL`
and `STEP_JUDGE_MODEL`.

Submit the committee judge service:

```bash
bash scripts/submit_dlc_step_judge_committee_service.sh
```

The default layout is:

- GPU0: Qwen3-VL-2B
- GPU1: Qwen3-VL-4B
- GPU2: Qwen3-VL-8B
- GPU3-6: Qwen3-VL-32B, TP=4
- GPU7: Qwen3-VL-8B test endpoint
- API: qwen3.6-plus and qwen3.5-397b-a17b by default

Local vLLM defaults use `--max-model-len 32768`. Increase
`JUDGE_MAX_MODEL_LEN_SMALL` / `JUDGE_MAX_MODEL_LEN_32B` if the context probe
shows long-trajectory failures, and reduce `STEP_JUDGE_MAX_IMAGES` if image
tokens dominate.

After the job starts, get the RL environment values:

```bash
bash scripts/step_judge_url_from_job.sh <judge_job_id_or_name>
```

Then launch RL with the printed exports plus:

```bash
TOOL_REWARD_MODE=mut_clean_step_v1
STEP_REWARD_ENABLE=True
STEP_JUDGE_NUM_JUDGMENTS=1
STEP_JUDGE_AGGREGATION=mean
```

For lower judge variance, run repeated judgments per state:

```bash
STEP_JUDGE_NUM_JUDGMENTS=2
STEP_JUDGE_AGGREGATION=mean
```

This calls the same configured judge endpoint twice for each baseline/step
state and averages rule-scored correctness. It is useful when the judge request
uses nonzero sampling temperature or the serving stack has nondeterminism.

## Probe Script

Use the standalone probe before long RL runs:

```bash
PYTHONPATH=/mnt/cpfs/delinmao/ToolVision/CodeVision \
python3 scripts/probe_step_judge_context.py \
  --judge-host <judge-pod-ip> \
  --num-judgments 2 \
  --jsonl-out outputs/step_judge_context_probe/probe.jsonl
```

Analyze saved rollout committee records without the old 122B member:

```bash
PYTHONPATH=/mnt/cpfs/delinmao/ToolVision/CodeVision \
python3 scripts/probe_step_judge_context.py \
  --records-only \
  --rollout-records saves/ToolVisionRL/<exp>/rollout_generations \
  --drop-member-regex 122b
```
