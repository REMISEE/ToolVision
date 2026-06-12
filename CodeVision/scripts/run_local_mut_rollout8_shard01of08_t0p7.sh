#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cpfs/delinmao/ToolVision/CodeVision

source /opt/conda/etc/profile.d/conda.sh
conda activate codevision

export OCR_BASE_URL=http://172.17.8.225:18080
export GROUNDEDSAM2_BASE_URL=http://172.17.8.225:18081
export DEPTH_BASE_URL=http://172.17.8.225:18082
export COUNTGD_BASE_URL=http://172.17.8.225:18083

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS_PER_NODE=4
export INFER_TP_SIZE=4
export RAY_INIT_NUM_CPUS=32
export RAY_INIT_INCLUDE_DASHBOARD=false
export VAL_BSZ=16
export MAX_NUM_SEQS=16
export ROLLOUT_AGENT_NUM_WORKERS=8

export EVAL_PARQUET=/mnt/cpfs/delinmao/data/toolvision_pass16_lmms_rerun_all/final_v3/toolvision_eval/shards_8way_safe/mut_candidates_0_8_toolvision_eval_shard01of08.parquet
export EXP_NAME=mut_rollout8_local_shard01of08_t0p7
export STREAM_VALIDATION_DUMP=True

bash recipe/codevision/run_toolvision_mut_rollout8.sh
