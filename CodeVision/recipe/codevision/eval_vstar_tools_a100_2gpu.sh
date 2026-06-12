#!/usr/bin/env bash
# =============================================================================
# V*Bench tool-enabled eval on 2 model GPUs.
#
# This is a thin wrapper around eval_vstar_tools_a100_4gpu.sh. It keeps the
# result-facing eval parameters in that script unchanged, while overriding only
# the model GPU topology and vLLM admission/concurrency defaults for 2xA100.
# =============================================================================

set -euo pipefail

export EXP_NAME="${EXP_NAME:-vstar_tools_a100_2gpu}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export INFER_TP_SIZE="${INFER_TP_SIZE:-2}"

# Keep n_resp_per_prompt/max_turns/length limits unchanged. This only lowers
# concurrent vLLM admission pressure for the smaller 2-GPU tensor-parallel group.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# Do not enable Ray's NOSET_* path here. In this codebase, that makes verl
# derive LOCAL_RANK from Ray accelerator ids, which has been producing invalid
# device ordinal for multi-GPU eval jobs. We only need to clear AMD-specific
# visibility vars.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

exec bash recipe/codevision/eval_vstar_tools_a100_4gpu.sh
