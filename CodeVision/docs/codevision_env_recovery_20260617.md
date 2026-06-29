# CodeVision Environment Recovery 2026-06-17

## Decision

Use this environment for CodeVision RL/eval jobs:

```text
/mnt/cpfs/delinmao/envs/codevision_new
```

Do not use the old shared environment for new jobs:

```text
/mnt/cpfs/delinmao/envs/codevision
```

The old environment was modified in place and no longer imports the Qwen3-VL
training stack reliably.  It reproduces:

```text
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

## Verified Stack

`codevision_new` matches the original CodeVision-style pins for the critical
runtime:

```text
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
triton==3.4.0
transformers==4.57.1
vllm==0.11.0
flash-attn==2.8.3
xformers==0.0.32.post1
ray==2.54.1
numpy==1.26.4
huggingface-hub==0.36.2
fastapi==0.119.0
starlette==0.48.0
pydantic==2.10.6
```

Verified imports:

```text
from transformers import AutoModelForVision2Seq
import verl.trainer.main_ppo
import recipe.codevision.reward
```

Verified Qwen3-VL checkpoint load metadata:

```text
model_type=qwen3_vl
architectures=['Qwen3VLForConditionalGeneration']
processor=Qwen3VLProcessor
```

## Known Non-Blocking `pip check` Conflicts

These conflicts come from the frozen environment pins in
`scripts/codevision_env_20260424.yaml`, so they are not treated as blockers:

```text
vllm==0.11.0 with pydantic==2.10.6
opencv-python==4.13.0.92 with numpy==1.26.4
opencv-python-headless==4.13.0.92 with numpy==1.26.4
cupy-cuda12x==14.0.1 with numpy==1.26.4
```

Do not "fix" these by upgrading shared packages unless a separate isolated
environment is created and tested end to end.

## Bound Entrypoints

The RL/eval launchers now default to `codevision_new`, and DLC submit commands
explicitly pass `CODEVISION_ENV` into the remote command.

Important files:

```text
scripts/dlc_ray_direct_entrypoint.sh
scripts/dlc_ray_entrypoint.sh
scripts/submit_dlc_gspo_direct_full.sh
scripts/run_tools_eval_all_wait_5gpu_nohup.sh
recipe/codevision/run_toolvision_40k_pass16_eval_direct.sh
```

If a new launcher is added, set:

```bash
CODEVISION_ENV=/mnt/cpfs/delinmao/envs/codevision_new
```

before invoking any Ray/DLC CodeVision entrypoint.
