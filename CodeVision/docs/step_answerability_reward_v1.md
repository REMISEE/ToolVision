# Step Answerability Reward v1

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

## Reward

Base reward is the current MUT-clean reward. Current 128bs MUT-v1 data has
`P_regular_tool = 0`; the field is only kept for old parquet compatibility.

```text
R_base = R_acc + 0.2 * R_protocol + mut_weight * R_mut - P_regular_tool - 0.05 * max(0, NumTurns - 6)
```

The step judge scores answerability before tools and after each tool step:

```text
V0 = score(original image + question)
Vt = score(state after tool step t)
gain_t = max(0, Vt - best_previous - tau)
R_step_raw = min(cap, sum(gain_t over valid steps))
R_step = step_weight * mut_weight * R_step_raw
R_total = R_base + R_step
```

Defaults:

```text
step_weight = 0.2
tau = 0.1
cap = 0.5
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

Submit one judge replica:

```bash
JUDGE_MODEL_PATH=/path/to/judge/model \
JUDGE_MODEL_NAME=step-judge \
JUDGE_REPLICA_COUNT=1 \
JUDGE_REPLICA_GPUS=0 \
bash scripts/submit_dlc_step_judge_service.sh
```

The default backend is:

```bash
vllm serve "${JUDGE_MODEL_PATH}" --host 0.0.0.0 --port "${JUDGE_PORT}"
```

If the actual deployment uses another framework, override the server command:

```bash
JUDGE_SERVER_CMD='swift deploy --model "$JUDGE_MODEL_PATH" --host 0.0.0.0 --port "$JUDGE_PORT" --infer_backend vllm' \
JUDGE_MODEL_PATH=/path/to/judge/model \
bash scripts/submit_dlc_step_judge_service.sh
```

After the job starts, get the RL environment values:

```bash
bash scripts/step_judge_url_from_job.sh <judge_job_id_or_name>
```

Then launch RL with the printed exports plus:

```bash
TOOL_REWARD_MODE=mut_clean_step_v1
STEP_REWARD_ENABLE=True
```
