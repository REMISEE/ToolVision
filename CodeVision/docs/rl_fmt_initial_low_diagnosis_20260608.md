# RL format reward initial-low diagnosis

Date: 2026-06-08

## Current conclusion

Eval-stage `format_reward` and RL-stage `R_fmt` are intended to be the same signal.

- The reward manager computes `format_reward` in `verl/workers/reward_manager/uvtr.py` via `UVTRRewardManager._compute_format_reward`.
- RL tool reward reads that exact field from `output.non_tensor_batch["format_reward"]` in `verl/experimental/agent_loop/agent_loop.py`.
- `R_total` then uses `format_reward_weight * R_fmt`.

So the main gap is not two different format scorers. The gap is generation regime:

- SFT eval/validation uses greedy validation generation: `temperature=0`, `do_sample=false`, `n=1`.
- RL train rollout uses stochastic rollout: `temperature=1.0`, `top_p=1.0`, `do_sample=true`, `n=8`.
- The format checker is strict: output must start with `<think>`, have balanced think tags, exactly one final `<answer>...</answer>`, and no answer inside a think block. Any extra text after answer, missing close tag, duplicate answer, or malformed tool-loop continuation gets 0.

This explains why eval can show format reward around 0.8-0.99 while RL step-0 sampled rollout can start around 0.2-0.3.

## 1-step guard result

Run inspected:

`saves/ToolVisionRL/qwen3vl8b_gspo_final_v1_sftclean_clean015_nood_fmtguard_1step/rollout_generations/1.jsonl`

Summary:

- Rows: 512
- Mean `format_reward`: 0.2578125
- Recomputed strict format mean: 0.2578125
- Stored `format_reward` exactly matches recomputation, so this is not a metric propagation artifact.
- `global_step_1` checkpoint exists.

Failure breakdown:

- `ok`: 132 / 512 = 25.8%
- `think_mismatch`: 231 / 512 = 45.1%
- `answer_count_0`: 118 / 512 = 23.0%
- `answer_count_multi`: 22 / 512 = 4.3%
- `answer_not_last`: 9 / 512 = 1.8%

Tool/no-tool split:

- no-tool rollouts: 100 / 116 valid format = 86.2%
- tool-using rollouts: 32 / 396 valid format = 8.1%

This localizes the problem: the base SFT model can usually emit the final `<think>...</think><answer>...</answer>` shape when it does not use tools, but the tool loop causes widespread format collapse.

Observed pattern:

- Many tool rollouts end with `<tool_call>...</tool_call>` and never produce a final answer.
- Many tool rollouts produce repeated or malformed JSON/tool-call structures.
- Several outputs repeat `<answer>` many times or continue generating after a valid answer.
- Output lengths are abnormal after tool use: median 2.3k chars, p90 55k chars, p99 500k+ chars, max 870k chars.

Root cause candidate found in generation code:

- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` used `max_tokens = max_model_len - len(prompt_ids)` for each assistant turn.
- The tool loop did not pass a per-turn cap.
- Therefore a post-tool assistant turn could generate until near model context length, not until a reasonable per-turn response budget.
- With no stop condition on `</answer>`, tool-turn generation can run away and then get truncated, which makes strict format reward 0.

Fix added:

- `ToolAgentLoop` now passes per-turn `max_tokens`.
- vLLM async server now respects request-level `max_tokens` by taking `min(model_remaining, requested_max_tokens)`.
- Launchers expose `ROLLOUT_MAX_TOKENS_PER_TURN`, default `2048`.

## What was wrong in the previous RL run

The 2026-06-02 run config had:

- `trainer.save_freq=400`
- `trainer.test_freq=-1`
- `trainer.val_before_train=False`
- `trainer.rollout_data_dir=null`
- `trainer.validation_data_dir=null`

Observed result:

- No `global_step_*` checkpoint exists under `saves/ToolVisionRL`.
- No local rollout JSONL directory was configured.
- Only console/W&B train-generation logging was available, so only sparse sampled examples could be inspected.

This means the previous run did not preserve enough local artifacts to debug format failure after the fact.

## Changes made now

Updated launchers:

- `recipe/codevision/qwen3_vl_gspo_direct.sh`
- `recipe/codevision/qwen3_vl_gspo.sh`
- `scripts/submit_dlc_gspo_direct_full.sh`

New behavior:

- Default `SAVE_FREQ=50`.
- Default `MAX_ACTOR_CKPT_TO_KEEP=5`, `MAX_CRITIC_CKPT_TO_KEEP=5`.
- Default `ROLLOUT_DATA_DIR=${SAVE_DIR}/rollout_generations`.
- DLC submit layer now forwards save/rollout settings instead of overriding save frequency back to 400.
- Rollout sampling is explicit and overrideable:
  - `ROLLOUT_TEMPERATURE`, default `1.0`
  - `ROLLOUT_TOP_P`, default `1.0`
  - `ROLLOUT_DO_SAMPLE`, default `True`

Defaults preserve the previous full-run sampling behavior. The added knobs are for diagnosis and controlled reruns.

Additional bug hardening:

- `verl/experimental/agent_loop/agent_loop.py` now keeps the union of all `reward_extra_info` keys instead of only the first sample's keys.
- If `format_reward` is missing from a sample's reward extra info, agent-loop postprocess recomputes it from the decoded response with the same strict format rules as `UVTRRewardManager`.
- This prevents a missing `format_reward` field from silently turning `R_fmt` into 0 in tool reward aggregation.

## Recommended immediate execution

Run a 1-step guard first. This is not a separate experiment; it is a launch preflight before burning a full run.

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

JOB_NAME=codevision_gspo_direct_fmt_guard_1step \
EXP_NAME=qwen3vl8b_gspo_final_v1_sftclean_clean015_nood_fmtguard_1step \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=1 \
LOG_TRAIN_FREQ=1 \
LOG_TRAIN_GENERATIONS=64 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
bash scripts/submit_dlc_gspo_direct_final_before_newdata.sh
```

Acceptance:

- `rollout_generations` exists under the save dir.
- At least one `global_step_*` checkpoint exists.
- W&B/console `format_reward` or `R_fmt` is not unexpectedly low.
- If `R_fmt` is still around 0.2-0.3, inspect the saved rollout JSONL and classify failures before full run.

If the 1-step guard confirms the issue is only high-temperature malformed tags, run a low-temperature confirmation:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

JOB_NAME=codevision_gspo_direct_fmt_guard_1step_t07 \
EXP_NAME=qwen3vl8b_gspo_final_v1_sftclean_clean015_nood_fmtguard_t07_1step \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=1 \
LOG_TRAIN_FREQ=1 \
LOG_TRAIN_GENERATIONS=64 \
ROLLOUT_TEMPERATURE=0.7 \
ROLLOUT_TOP_P=0.95 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
bash scripts/submit_dlc_gspo_direct_final_before_newdata.sh
```

If low temperature fixes `R_fmt` but original temperature does not, the full run should use the lower-temperature rollout. This changes exploration but is a real fix; post-processing malformed outputs before reward is not recommended because it hides the policy behavior.

## Temperature ramp option

The current VERL config does not provide an in-run rollout temperature schedule. Use staged DLC jobs with the same `EXP_NAME` / `SAVE_DIR` and `resume_mode=auto`:

1. Run steps 1-50 with `ROLLOUT_TEMPERATURE=0.7`, `ROLLOUT_TOP_P=0.95`.
2. Resume to step 100 with `ROLLOUT_TEMPERATURE=0.85`, `ROLLOUT_TOP_P=0.95`.
3. Resume beyond step 100 with `ROLLOUT_TEMPERATURE=1.0`, `ROLLOUT_TOP_P=1.0` only if `R_fmt` is stable.

This is safer than adding a new dynamic scheduler inside rollout because each stage leaves an inspectable checkpoint and rollout JSONL.

## Full run command

Same run shape as the previous final-before-newdata run, with checkpoint/rollout retention fixed:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision

JOB_NAME=codevision_gspo_direct_final_v1_fmtfixed \
EXP_NAME=qwen3vl8b_gspo_final_v1_sftclean_clean015_nood_fmtfixed \
SAVE_FREQ=50 \
MAX_ACTOR_CKPT_TO_KEEP=8 \
MAX_CRITIC_CKPT_TO_KEEP=3 \
LOG_TRAIN_FREQ=10 \
LOG_TRAIN_GENERATIONS=32 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
bash scripts/submit_dlc_gspo_direct_final_before_newdata.sh
```

If 1-step shows that lower-temperature rollout is required, add:

```bash
ROLLOUT_TEMPERATURE=0.7 \
ROLLOUT_TOP_P=0.95 \
```

## Post-run eval

Evaluate checkpoints at least at:

- `global_step_50`
- `global_step_100`
- `global_step_150`
- last step

The previous training looked acceptable around 100+ steps but did not preserve checkpoints. With `SAVE_FREQ=50`, this run should leave enough checkpoints for regression analysis.

## Current blocker for launching from this shell

This shell currently has no `WANDB_API_KEY`, `LLM_JUDGE_BASE_URL`, or `LLM_JUDGE_MODEL_NAME` environment variables. The DLC submit script will fail preflight without them when W&B and judge are enabled. The tool service URLs are generated by `scripts/dsw_tool_urls.sh` at submit time.
