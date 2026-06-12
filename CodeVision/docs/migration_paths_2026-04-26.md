# ToolVision Path Migration Notes - 2026-04-26

This records the hardcoded path migration after moving the workspace to `/mnt/cpfs/delinmao`.

## New Defaults

- `WORKSPACE_ROOT`: `/mnt/cpfs/delinmao`
- `TOOLVISION_ROOT`: `${WORKSPACE_ROOT}/ToolVision`
- ToolVision CodeVision repo: `${WORKSPACE_ROOT}/ToolVision/CodeVision`
- Top-level CodeVision repo: `${WORKSPACE_ROOT}/CodeVision`
- Original CodeVision repo: `${WORKSPACE_ROOT}/CodeVision_orig`
- Benchmarks: `${WORKSPACE_ROOT}/Benchmarks`
- SFT checkpoint: `${WORKSPACE_ROOT}/outputs/qwen3vl_sft/full`
- Base Qwen3-VL checkpoint: `${WORKSPACE_ROOT}/models/Qwen3-VL-8B-Thinking`
- Slurm logs: `${WORKSPACE_ROOT}/logs`
- Ray temp dirs: `${WORKSPACE_ROOT}/ray_tmp/...`

All shell scripts below keep these values overrideable with environment variables.

## Conda Defaults

The migrated machine exposes conda at `/opt/conda`, so scripts now also search:

- `${CONDA_SH_PATH}`
- `/mnt/public/apps/miniconda3/etc/profile.d/conda.sh`
- `/opt/conda/etc/profile.d/conda.sh`
- `${HOME}/miniforge3/etc/profile.d/conda.sh`

Default env names were changed from old absolute paths to the env names available on this machine:

- `CODEVISION_ENV=toolvision`
- `OCR_ENV=paddleocr`
- `GROUNDEDSAM2_ENV=groundedsam2`
- `DEPTH_ENV=depth-pro`
- `COUNTGD_ENV=countgd`

## Files Changed

- `scripts/launch_external_services.sh`
  - Added `WORKSPACE_ROOT` and `TOOLVISION_ROOT`.
  - Made `ROOT_DIR`, log dir, pid dir, service envs, GroundedSAM2 files, Depth root, and CountGD root overrideable.
  - Replaced old `/mnt/users/maodelin-20251119/...` defaults.

- `scripts/slurm_tools_eval_all_a100_3gpu.sbatch`
  - Migrated Slurm log paths, repo root, benchmark root, model path, Ray temp dir, and CodeVision env.
  - Added `/opt/conda` conda fallback.
  - Preserved the 3-GPU binding: first 2 GPUs for model, third GPU for external tools.

- `scripts/slurm_vstar_tools_eval_a100_1gpu_probe.sbatch`
- `scripts/slurm_vstar_tools_eval_a100_3gpu.sbatch`
- `scripts/slurm_vstar_tools_eval_a100_5gpu.sbatch`
  - Migrated repo/data/model/Ray/log paths and CodeVision env.
  - Added `/opt/conda` conda fallback.

- `scripts/slurm_tools_smoke_a100.sbatch`
  - Migrated repo/run/log paths and all env defaults.
  - Added `/opt/conda` conda fallback.

- `scripts/slurm_tools_eval_all_base_cur_a100_3gpu.sbatch`
  - Migrated base model path, Slurm logs, and delegated script path.

- `recipe/codevision/eval_vstar_tools_a100_4gpu.sh`
  - Migrated default SFT model and VStar parquet paths to `WORKSPACE_ROOT`/`BENCHMARK_ROOT`.

- `recipe/codevision/eval_vstar_base.sh`
  - Migrated default base model and VStar parquet paths.

- `recipe/codevision/prepare_benchmarks.py`
  - Added `WORKSPACE_ROOT`/`BENCHMARK_ROOT` defaults instead of a fixed old benchmark root.

- `recipe/codevision/prepare_vstar_bench.py`
  - Added `WORKSPACE_ROOT`/`VSTAR_BENCH_ROOT` defaults for VStar input/output paths.

- `recipe/codevision/tools/convert_vstar_to_parquet.py`
  - Updated usage examples to the new benchmark path.

- `scripts/run_tools_eval_all_wait_5gpu_nohup.sh`
  - Added a nohup-friendly eval launcher for the new SFT checkpoint.
  - Waits for the SFT model output to be complete, then waits for 5 free GPUs.
  - Uses 4 GPUs for the eval model/vLLM tensor parallel and 1 dedicated GPU for OCR/GroundedSAM2/Depth/CountGD services.
  - Defaults to `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/saves/qwen3vl-8b/sft-drop-simple-notool` and the migrated benchmark root.

- `/mnt/cpfs/delinmao/CodeVision/scripts/run_qwen3vl.sbatch`
  - Migrated training repo/model/output/log/cache/dataset paths.
  - Added robust conda sourcing and default `CODEVISION_ENV=toolvision`.

- `/mnt/cpfs/delinmao/CodeVision/LLaMA-Factory/examples/train_full/qwen3vl.yaml`
  - Migrated `model_name_or_path` to `/mnt/cpfs/delinmao/models/Qwen3-VL-8B-Thinking`.

- `/mnt/cpfs/delinmao/CodeVision_orig/scripts/slurm_tools_eval_all_base_orig_a100_2gpu.sbatch`
  - Migrated repo/model/benchmark/Ray/log paths and CodeVision env.
  - Added `/opt/conda` conda fallback.

- `/mnt/cpfs/delinmao/CodeVision_orig/scripts/slurm_tools_eval_all_sft_orig_a100_2gpu.sbatch`
  - Migrated SFT model and delegated script path.

- `/mnt/cpfs/delinmao/CodeVision_orig/scripts/slurm_vstar_base_orig_a100_2gpu.sbatch`
  - Migrated repo/model/parquet/Ray/log paths and CodeVision env.
  - Added robust conda sourcing.

- `/mnt/cpfs/delinmao/CodeVision_orig/recipe/codevision/eval_vstar_base_orig_a100_2gpu.sh`
  - Migrated default base model and VStar parquet paths.

## Verification

After the edits, this scoped search returned no old user-machine path defaults:

```bash
grep -RInE '/mnt/users/maodelin-20251119|conda/envs/codevision_clean|/data/home/suchenghao' \
  ToolVision/CodeVision/scripts ToolVision/CodeVision/recipe \
  CodeVision/scripts CodeVision/LLaMA-Factory/examples/train_full \
  CodeVision_orig/scripts CodeVision_orig/recipe \
  --include='*.sh' --include='*.sbatch' --include='*.py' --include='*.yaml'
```

The remaining absolute `/mnt/cpfs/delinmao` values are intentional new defaults and can be overridden with `WORKSPACE_ROOT` or the more specific variables documented above.
