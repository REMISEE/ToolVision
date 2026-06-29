# RL Model Eval Scores

Date: 2026-06-23

This note records the current MUT RL eval scores and fixes the naming convention so later comparisons do not mix up `v03`, `v04`, `base`, and `base_pipeline`.

## Model Lineage

The MUT v1 RL runs discussed here are initialized from v03, not v04.

- RL base model / v03:
  `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-mix200-simple-notool-sp3-v03`
- MUT v1 128bs checkpoint step 60, merged HF:
  `/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/merged_hf/mutv1_128bs_global_step_60`
- MUT v1 128bs checkpoint step 140, merged HF:
  `/mnt/cpfs/delinmao/ToolVision/CodeVision/saves/ToolVisionRL/merged_hf/mutv1_128bs_global_step_140`

Evidence:

- `scripts/submit_dlc_gspo_mut_v1_t07_cap2048_mns32.sh` defaults `MODEL_PATH` to v03.
- `scripts/submit_dlc_gspo_mut1_128bs_0618.sh` launches the 128bs run using the same v03 default path.
- Eval manifests for `mutv1_128bs_s60_merged_*` and `mutv1_128bs_s140_merged_*` point to the merged HF checkpoints above.

Do not use v04 as the baseline for these MUT v1 results. v04 is a separate SFT checkpoint for later experiments.

## Naming

- `base`: Qwen3-VL-8B-Thinking original model baseline from `/mnt/cpfs/delinmao/docs/20260510_toolvision_final_summary.md`.
- `base_pipeline`: SFT model trained from strong-model trajectory data. It is not Qwen3-VL-8B-Thinking base.
- `v03`: `sft-mix200-simple-notool-sp3-v03`, the actual initialization model for MUT v1.
- `mut1 s60` / `mut1 s140`: MUT v1 128bs RL checkpoints at global step 60 / 140.

## Metric Definition

`macro6` means the simple average over these six non-FSC benchmarks:

1. ChartQA
2. V*
3. CountBench
4. OCRBench
5. HRBench4K
6. HRBench8K

It excludes FSC147 because FSC147 is normally reported as MAE in the 2026-05-10 summary, and it excludes ArxivQA because the older base/v03 evals did not include ArxivQA.

## Main Comparison

For the six non-FSC benchmarks, the table uses official-like score / accuracy. Higher is better.
The CountBench base number is the refreshed local no-tool lmms-eval result from 2026-06-26; the other base numbers still come from the older baseline summary.

| dataset | Qwen3VL8B Thinking base | v03 | mut1 s60 | mut1 s140 | s140 - v03 | s140 - base |
|---|---:|---:|---:|---:|---:|---:|
| ChartQA | 0.8860 | 0.8420 | 0.8620 | 0.8724 | +0.0304 | -0.0136 |
| V* | 0.7750 | 0.7696 | 0.7801 | 0.8115 | +0.0419 | +0.0365 |
| CountBench | 0.8839 | 0.8697 | 0.8819 | 0.8758 | +0.0061 | -0.0081 |
| OCRBench | 0.8190 | 0.7320 | 0.7480 | 0.7890 | +0.0570 | -0.0300 |
| HRBench4K | 0.7240 | 0.7425 | 0.8063 | 0.8137 | +0.0712 | +0.0897 |
| HRBench8K | 0.6810 | 0.6950 | 0.7350 | 0.7550 | +0.0600 | +0.0740 |
| macro6 | 0.7948 | 0.7751 | 0.8022 | 0.8196 | +0.0444 | +0.0248 |

Interpretation:

- MUT v1 s140 is clearly above v03 on this six-benchmark average: `+4.44 pp`.
- MUT v1 s140 is only modestly above Qwen3VL8B Thinking base on macro6: `+2.48 pp` with the refreshed CountBench base.
- It is still below Thinking base on ChartQA, CountBench, and OCRBench, but the CountBench gap is small after the refreshed direct-answer eval: `-0.81 pp`.
- The strongest gains are on V*, HRBench4K, HRBench8K, and FSC147.

## CountBench Base Sanity Eval

Refreshed local no-tool direct-answer CountBenchQA-491 result for `Qwen3-VL-8B-Thinking`:

| model | task | samples | acc / exact_match | correct | primary output path |
|---|---|---:|---:|---:|---|
| Qwen3-VL-8B-Thinking | `tv_countbench_local` | 491 | 0.8839 | 434 | `/mnt/cpfs/delinmao/lmms-eval/logs/base_qwen3vl8bthinking_countbench_lmms_t0_len32k_gen4096/models__Qwen3-VL-8B-Thinking/20260626_143457_results.json` |

Run config:

- model path: `/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking`
- backend: `lmms_eval --model vllm_generate`
- conda env: `/mnt/cpfs/delinmao/envs/codevision`
- task/data: `tv_countbench_local`, 491 rows from `vikhyatk/CountBenchQA` test
- vLLM args: `max_model_len=32768`, `max_new_tokens=4096`, `max_num_seqs=64`, `data_parallel_size=8`, `tensor_parallel_size=1`
- batch size: `32`

Why this replaces the older report-only CountBench base number:

- The old table used `0.9150` from `/mnt/cpfs/delinmao/docs/20260510_toolvision_final_summary.md`; that was not rerun through the current no-tool lmms-eval CountBench task.
- A first DLC attempt with `max_model_len=8192` was invalid for this dataset: two samples exceeded the context limit, with prompt lengths `8878` and `16267`.
- A second attempt with `max_model_len=16384` but `max_new_tokens=16` completed all 491 samples but scored `0.0` because the Thinking model was truncated before producing the final number; sample outputs started with text such as `Got it, let's count...`.
- The final `len32k_gen4096` run produced mostly numeric outputs (`488/491`) and is the current comparable no-tool direct-answer baseline for CountBenchQA-491.

## FSC147

The 2026-05-10 summary reports FSC147 as MAE, lower is better. Current MUT eval metrics primarily store relative score, so the MUT MAE below is parsed from `diagnostics/metadata.jsonl` by comparing `final_answer` and `ground_truth`.

| dataset | Qwen3VL8B Thinking base MAE | v03 MAE from old summary | mut1 s60 parsed MAE | mut1 s140 parsed MAE |
|---|---:|---:|---:|---:|
| FSC147-val | 39.59 | 12.43 | 10.30 | 10.87 |
| FSC147-test | 45.56 | 15.77 | 12.47 | 11.21 |

Current relative score from metrics:

| dataset | v03 relative | mut1 s60 relative | mut1 s140 relative |
|---|---:|---:|---:|
| FSC147-val | 0.8628 | 0.8952 | 0.9004 |
| FSC147-test | 0.8617 | 0.8964 | 0.8993 |

## ArxivQA

ArxivQA was added after the old base/v03 summaries. The old `base` and `v03` rows therefore do not have ArxivQA numbers yet.

Current MUT v1 ArxivQA holdout result, after fixing the letter-choice scorer offline:

| dataset | mut1 s60 | mut1 s140 |
|---|---:|---:|
| ArxivQA holdout 2000 | 0.7155 | 0.7315 |

The raw logged ArxivQA metrics around `0.15-0.16` are a scorer bug: option metadata is stored as a dict, and the generic multiple-choice scorer collapses the valid letter range. The fixed letter-only rescore is the number to use for now.

Recommended next action:

- Run ArxivQA holdout for v03.
- Run ArxivQA holdout for Qwen3-VL-8B-Thinking base if we want to compare against the old `base` column.
- Patch the ArxivQA multiple-choice scorer before treating future online metrics as authoritative.

## Result Files

Current MUT eval outputs:

- `saves/CodeVision/mutv1_128bs_s60_merged_allbench_8gpu_g1_*_t0/metrics.json`
- `saves/CodeVision/mutv1_128bs_s140_merged_allbench_8gpu_g1_*_t0/metrics.json`
- `outputs/analysis/mut1_eval_20260623/mutv1_128bs_eval_weighted_summary_arxivqa_fixed.tsv`
- `outputs/analysis/mut1_eval_20260623/arxivqa_rescore_letter.json`

Older baseline summary:

- `/mnt/cpfs/delinmao/docs/20260510_toolvision_final_summary.md`

Refreshed CountBench base sanity output:

- `/mnt/cpfs/delinmao/lmms-eval/logs/base_qwen3vl8bthinking_countbench_lmms_t0_len32k_gen4096/models__Qwen3-VL-8B-Thinking/20260626_143457_results.json`
- `/mnt/cpfs/delinmao/lmms-eval/logs/base_qwen3vl8bthinking_countbench_lmms_t0_len32k_gen4096/models__Qwen3-VL-8B-Thinking/20260626_143457_samples_tv_countbench_local.jsonl`

v03 eval outputs:

- `saves/CodeVision/mix200_sft_sp3_v03_first8_*`
