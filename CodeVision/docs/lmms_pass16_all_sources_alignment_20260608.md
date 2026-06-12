# lmms-eval All-Source Pass16 Alignment

## Current conclusion

The old vLLM/CodeVision rollout16 outputs are suspect for Qwen3-VL-8B. GQA/TextVQA/FSC147 controls showed that lmms-eval real images strongly outperform blank/shuffled images, while the old outputs often matched blank-image behavior or collapsed to repeated wrong answers.

Use lmms-eval only for rollout/inference. Save raw 16 generations, convert to pass16-like parquet, then score with ToolVision/RL scorer v2.

## GPU confirmation

The previous lmms smoke/control used GPU through:

```bash
accelerate launch --num_processes=8 -m lmms_eval --model qwen3_vl ...
```

During control, eight workers were launched and each GPU used about 17GB. The 9-task control completed generation in about 97 seconds.

## Files added for all-source rerun

- `recipe/codevision/tools/prepare_lmms_pass16_all_sources.py`
- `recipe/codevision/run_lmms_pass16_rerun_all_sources.sh`
- `lmms_tasks/toolvision_pass16_all/*.yaml`
- `lmms_tasks/toolvision_pass16_all/utils.py`

The converter was also fixed so source names with underscores, e.g. `pixmo_count`, are parsed correctly.

## Current source coverage

Old pass16 total: 127,814 samples.

Currently image-resolved sources:

| source | samples |
|---|---:|
| gqa | 20,000 |
| textvqa | 10,000 |
| fsc147 | 1,286 |
| arxivqa | 9,993 |
| countqa | 1,521 |
| ocrbench | 9,993 |
| refl4 | 11,993 |
| sat2 | 5,993 |
| virgorlsa | 9,993 |

Resolved subtotal: 80,772 samples.

Pending image resolver/source cache:

| source | samples | status |
|---|---:|---|
| chartqa | 19,993 | local ChartQA cache is partial; only some source_index rows resolve |
| ai2d | 2,427 | source_index image cache not located yet |
| docvqa | 5,342 | source_index image cache not located yet |
| infographicvqa | 2,794 | source_index image cache not located yet |
| mmstar | 1,493 | source_index image cache not located yet |
| pixmo_count | 14,993 | source_index image cache not located yet |

## Scoring standard

lmms-eval metrics are only sanity checks. Final score must be:

```bash
convert_lmms_samples_to_pass16.py -> rescore_pass16_v2.py
```

Primary scorer families:

| source | scorer |
|---|---|
| gqa | normalized exact; synonym dict still optional/future |
| textvqa | EvalAI/VQA soft score |
| fsc147 | relative count score, primary success >= 0.9 |
| chartqa | relaxed numeric/text score |
| docvqa / infographicvqa | ANLS / Levenshtein floor |
| ocrbench | currently OCR inclusion; OCRBench_v2 official scorer metadata should be added before final reporting |
| countqa / pixmo_count | numeric exact for primary; relative diagnostics recommended |
| refl4 | IoU, primary success >= 0.5 |
| ai2d / arxivqa / mmstar / virgorlsa | multiple choice |
| sat2 | routed by answer type/source |

## Size recommendation

If only the currently resolved 80,772 samples are rerun:

| usable rate | usable rows |
|---:|---:|
| 10% | 8,077 |
| 15% | 12,115 |
| 20% | 16,154 |
| 25% | 20,193 |
| 30% | 24,231 |

For the full 127,814 samples:

| usable rate | usable rows |
|---:|---:|
| 10% | 12,781 |
| 15% | 19,172 |
| 20% | 25,562 |
| 25% | 31,953 |
| 30% | 38,344 |

Recommendation: rerun all 127,814 once image resolvers are complete, then select about 26k by new pass16 buckets. If time is tight, first run the 80,772 resolved subset, but expect it may be short of 26k unless the usable rate is near 30%.

## Commands

Prepare 20-row smoke inputs:

```bash
cd /mnt/cpfs/delinmao/ToolVision/CodeVision
bash recipe/codevision/run_lmms_pass16_rerun_all_sources.sh prepare_smoke
```

Smoke on currently resolved sources:

```bash
SOURCES=gqa,textvqa,fsc147,arxivqa,countqa,ocrbench,refl4,sat2,virgorlsa \
bash recipe/codevision/run_lmms_pass16_rerun_all_sources.sh smoke
```

Prepare full inputs for currently resolved sources:

```bash
SOURCES=gqa,textvqa,fsc147,arxivqa,countqa,ocrbench,refl4,sat2,virgorlsa \
bash recipe/codevision/run_lmms_pass16_rerun_all_sources.sh prepare_full
```

Full rollout16 for currently resolved sources:

```bash
SOURCES=gqa,textvqa,fsc147,arxivqa,countqa,ocrbench,refl4,sat2,virgorlsa \
bash recipe/codevision/run_lmms_pass16_rerun_all_sources.sh full
```

Convert and rescore:

```bash
SOURCES=gqa,textvqa,fsc147,arxivqa,countqa,ocrbench,refl4,sat2,virgorlsa \
bash recipe/codevision/run_lmms_pass16_rerun_all_sources.sh convert
```

