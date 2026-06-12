# lmms-eval Rerun Plan for 06-02 pass16

## Immediate Finding

The current rollout16 outputs have obvious visual-path symptoms, not just scorer noise:

- `gqa`: 20,000 samples, mean `correct_count/16 = 32.47%`, `45.64%` all-zero.
- `textvqa`: 10,000 samples, mean `correct_count/16 = 8.16%`, `82.24%` all-zero.
- `fsc147`: 1,286 samples, mean `correct_count/16 = 1.03%`, `90.05%` all-zero.

Examples are very direct:

- FSC147 counting rows often answer `<answer>0</answer>` for all 16 generations even when the target is 83 or 10.
- OCR/Doc/Chart rows in the wider pass16 dump sometimes output unrelated generic tables or unrelated markdown text.

This is consistent with a bad rollout/image loading path or sample-image mismatch. It is not credible as Qwen3-VL-8B visual ability.

## Migration Decision

Use lmms-eval for inference only, then use ToolVision scorer v2 for final scoring/bucketing.

Built-in lmms-eval tasks are useful only as baseline:

- `gqa_lite` / `gqa`: sanity check local model + lmms image path.
- `textvqa_val` / `textvqa_val_lite`: sanity check only; our 06-02 TextVQA rows are train question ids.
- `fsc147`: sanity check only; our same-sample subset is from the local FSC147 eval parquet.

For the actual 06-02 replacement, use same-sample external tasks:

- `tv_pass16_gqa`
- `tv_pass16_textvqa`
- `tv_pass16_fsc147`

These read JSONL exported from the original pass16 parquet and open real image files through lmms-eval.

## Files Added

- `recipe/codevision/tools/prepare_lmms_pass16_rerun.py`
  - Exports same-sample JSONL.
  - Materializes GQA/TextVQA images from local benchmark parquet bytes.
  - Reuses FSC147 file paths.
  - Supports `real`, `blank`, and `shuffled` control modes.

- `lmms_tasks/toolvision_pass16_0602/*.yaml`
  - External lmms-eval tasks for real/blank/shuffled GQA, TextVQA, FSC147.

- `recipe/codevision/tools/convert_lmms_samples_to_pass16.py`
  - Converts lmms-eval `--log_samples` output into pass16-like parquet with `pred_texts_json`.

- `recipe/codevision/run_lmms_pass16_rerun_0602.sh`
  - Staged launcher for preflight, smoke, control, full, and conversion.

## Commands

Run from the CodeVision repo:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
```

Preflight built-in lmms GQA baseline:

```bash
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh preflight
```

Same-sample smoke:

```bash
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh prepare_smoke
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh smoke
```

Control:

```bash
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh prepare_control
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh control
```

Full rerun:

```bash
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh prepare_full
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh full
bash recipe/codevision/run_lmms_pass16_rerun_0602.sh convert
```

Defaults:

- model: `/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Instruct`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- processes: `8`
- lmms repo: `/mnt/cpfs/delinmao/lmms-eval`
- output: `/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun`

## Acceptance

Smoke:

- Every logged sample must have exactly 2 `filtered_resps`.
- `input_media` must be present.
- GQA/TextVQA/FSC147 outputs must be non-empty and not collapse to constant `0`.

Control:

- GQA real primary accuracy >= blank primary accuracy + 15 percentage points.
- TextVQA real primary accuracy >= blank primary accuracy + 20 percentage points.
- FSC147 real mean relative score >= blank mean relative score + 20 percentage points.

Full:

- Every sample has 16 responses.
- Convert produces `pass16_like.parquet`.
- ToolVision scorer v2 produces `rerun_0602_by_source.csv`.

## Notes

The default conda activation should be:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate lmms-eval
```

Avoid `conda run ... python - <<EOF` for this environment; it produced misleading missing-module errors during checks. Direct activation and the env's absolute Python both showed `accelerate` and `loguru` are installed.
