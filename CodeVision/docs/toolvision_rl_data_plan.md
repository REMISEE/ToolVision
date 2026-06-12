# ToolVision RL Data Plan for CodeVision

Date: 2026-05-20

This document records the current ToolVision data generation status, target mix, CodeVision RL parquet alignment, deduplication policy, and follow-up work for GRPO training in this repository.

Concrete deduplication and conversion rules are specified in `docs/toolvision_rl_dedup_conversion.md`.

## Scope

The target training stack is:

- Project: `/mnt/cpfs/delinmao/ToolVision/CodeVision`
- RL framework: local `verl`
- Training config: `recipe/codevision/config/grpo_trainer.yaml`
- Dataset class: `recipe/codevision/uvtr.py::CustomRLHFDataset`
- Reward entry: `recipe/codevision/reward.py::compute_score`
- Multi-turn tool loop: `verl/experimental/agent_loop/tool_agent_loop.py`

The generated data currently lives under:

```text
/mnt/cpfs/delinmao/data/toolvision_innovator_scaled
```

Those files were produced with the Innovator-style data creation pipeline, but the final training data must be converted to CodeVision/verl RL parquet format.

## Naming Correction

For RL data, the dataset identity field should be `source_dataset`, not `source_benchmark`.

Rationale:

- `source_dataset` identifies where a training sample came from.
- `source_benchmark` is an evaluation concept and was used in CodeVision eval scripts to bucket metrics.
- RL data may include generated, filtered, mixed, or repurposed training samples that are not benchmark splits.

For compatibility with existing CodeVision reward code, a temporary mapping layer can map `source_dataset` to a reward family. The stable training metadata should still use `source_dataset`.

Recommended metadata fields:

```python
extra_info = {
    "source_dataset": "chartqa",
    "source_original_id": "...",
    "source_split": "train",
    "question": "...",
    "answer": "...",
    "answer_type": "number",
    "task_type": "chart_qa",
    "ability_bucket": "ocr_chart_document",
    "reward_family": "chartqa_relaxed",
    "image_path": "...",
    "image_hash": "...",
    "question_hash": "...",
    "qa_hash": "...",
    "dedup_group": "...",
}
```

## Current Generated Data Inventory

Current generated sample counts:

| source_dataset | generated |
|---|---:|
| chartqa | 9,222 |
| refl4 | 8,114 |
| virgorlsa | 4,699 |
| pixmo_count | 4,609 |
| sat2 | 4,000 |
| arxivqa | 3,000 |
| ocrbench | 1,269 |
| docvqa | 1,244 |
| infographicvqa | 1,001 |
| mmstar | 326 |
| ai2d | 200 |
| countqa | 170 |
| **Total** | **37,854** |

Notes:

- `pixmo_count` image fetching was rate-limited by Flickr, so current data is a partial successful subset.
- `sat2` was stopped after around 4k generated samples because it was slow.
- `arxivqa` currently has 3k generated samples, enough for the 2.5k target.
- Some generated samples may need to be regenerated after reward thresholds are finalized.

## Target 40k Mix

The working target is a balanced source-level mix rather than uniform cap per source.

### New ToolVision Sources

Exact 40k target:

| source_dataset | target | current available | action |
|---|---:|---:|---|
| chartqa | 4,790 | 9,222 | sample |
| refl4 | 3,000 | 8,114 | sample |
| virgorlsa | 3,000 | 4,699 | sample |
| pixmo_count | 4,000 | 4,609 | sample |
| sat2 | 2,500 | 4,000 | sample |
| arxivqa | 2,500 | 3,000 | sample |
| ocrbench | 1,269 | 1,269 | all |
| docvqa | 1,244 | 1,244 | all |
| infographicvqa | 1,001 | 1,001 | all |
| ai2d | 200 | 200 | all |
| countqa | 170 | 170 | all |
| mmstar | 326 | 326 | all |

New ToolVision subtotal: 24,000.

### Original/Control Sources

Target control mix:

| source_dataset | target | notes |
|---|---:|---|
| virl39k | 5,000 | stratify by stable `problem_type` / `answer_type` if available |
| WaltonFuture | 3,000 | source cap unless stable sublabels are available |
| thinklite | 2,000 | source cap |
| tqa | 2,000 | source cap or weak split into science/diagram/general |
| mmk12 | 1,500 | source cap or weak split into STEM/diagram |
| wemath_standard | 1,500 | math/STEM control |
| puzzlevqa | 1,000 | visual reasoning control |

Control subtotal: 16,000.

Total target: 40,000.

The 290-sample top-up to reach exactly 40k comes from `chartqa`, because it is abundant and uses deterministic rule scoring.

## Ability Buckets

The mix should be explained by capability coverage, not only source counts.

| bucket | sources |
|---|---|
| OCR/chart/document | chartqa, ocrbench, docvqa, infographicvqa |
| Counting | pixmo_count, countqa |
| Grounding/visual search | refl4, virgorlsa |
| Spatial/diagram | sat2, ai2d, mmk12 subset if available |
| Scientific/STEM figure reasoning | arxivqa, wemath_standard, tqa/mmk12 subsets |
| General visual reasoning/control | virl39k, WaltonFuture, thinklite, puzzlevqa, mmstar |

If a source cannot be stably split into subtypes, use source-level cap and record weak labels only as auxiliary metadata.

## CodeVision RL Parquet Format

The converted parquet should follow CodeVision/verl format:

```python
{
    "data_source": "chartqa",
    "agent_name": "tool_agent",
    "ability": "mm_qa",
    "prompt": [
        {
            "role": "user",
            "content": "<image>Image size = 1024x768 pixels.\n\nQuestion text..."
        }
    ],
    "images": [
        {"image": "file:///absolute/path/to/image.jpg"}
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"
    },
    "extra_info": {
        "index": 0,
        "uid": "chartqa::<source-id>",
        "source_dataset": "chartqa",
        "source_original_id": "...",
        "source_split": "train",
        "question": "Question text...",
        "answer": "42",
        "answer_type": "number",
        "task_type": "chart_qa",
        "ability_bucket": "ocr_chart_document",
        "reward_family": "chartqa_relaxed",
        "image_path": "/absolute/path/to/image.jpg",
        "image_width": 1024,
        "image_height": 768,
        "image_hash": "...",
        "question_hash": "...",
        "qa_hash": "...",
        "dedup_group": "..."
    }
}
```

Important constraints:

- Use `file:///absolute/path` image references first. Avoid storing image bytes in the RL parquet unless needed.
- Keep one `<image>` placeholder per image.
- First round should only support single-image samples.
- Preserve extra metadata even if the current reward does not consume it.
- Avoid introducing `source_benchmark` in newly generated RL data unless a temporary compatibility bridge is needed.

## Single-Image Policy

First-round RL data should be single-image only.

Reasons:

- Existing CodeVision benchmark conversion scripts are mostly single-image.
- The prompt convention includes one image-size line:

```text
<image>Image size = WxH pixels.

question
```

- Multi-image training would require stricter `image_index` behavior in tool calls.
- Multi-image reward/debugging is harder because the tool may return additional images during rollout.

Policy:

- Keep single-image rows.
- Filter multi-image rows for the first run.
- If a multi-image dataset has independent per-image questions, convert only when a stable one-image mapping exists.
- Revisit multi-image after the first GRPO smoke and main run are stable.

## Deduplication Plan

Do not deduplicate by image alone. Same-image multi-question samples are valid.

Hard dedup keys:

```text
source_dataset + source_original_id
image_hash + normalized_question
```

Soft grouping:

```text
image_hash
```

Recommended policy:

- Drop exact duplicate source ids.
- Drop exact same image + normalized question.
- Keep same image with different questions.
- Apply per-image soft cap only if a source is dominated by repeated images.
- Store group metadata for later audits.

Recommended stored fields:

```python
{
    "image_hash": "...",
    "question_hash": "...",
    "qa_hash": "...",
    "dedup_group": "image_hash",
    "image_group_rank": 0,
    "image_group_size": 3,
}
```

## Reward Migration Plan

CodeVision's current GRPO reward path is:

```text
UVTRRewardManager
  -> recipe/codevision/reward.py::compute_score
  -> optional tool reward shaping in verl/experimental/agent_loop/agent_loop.py
```

Current usable reward components:

- answer accuracy
- format reward
- tool usage metadata
- tool overuse penalty through `C_usage`
- tool execution success/error counters

Important current gap:

- Tool execution error is recorded as `tool_exec_error_count`, but it should be verified whether the current total reward formula actually subtracts it. If not, add a minimal error penalty.

### Source Reward Mapping

| source_dataset / answer_type | reward_family | current status |
|---|---|---|
| chartqa | chartqa_relaxed | deterministic rule |
| ocrbench | ocr_inclusion | deterministic rule |
| pixmo_count/countqa number | numeric_exact | deterministic rule |
| mmstar/arxivqa/ai2d/virgorlsa multiple-choice | multiple_choice | deterministic rule |
| docvqa/infographicvqa ocrtext | ocr_levenshtein | deterministic rule, optional judge fallback |
| refl4 bbox | bbox_iou | deterministic rule |
| sat2 multiple-choice | multiple_choice | deterministic rule |
| sat2 boolean | boolean | deterministic rule |
| sat2 number | numeric_exact | deterministic rule |
| sat2 any | exact | exact rule, optional judge fallback |
| original/control math-expressions | math_verify | deterministic rule |
| original/control judge | judge | exact fallback + LLM judge |
| original/control html-code/svg-code/general-code | html_code/svg_code/general_code | exact fallback + LLM judge; not included in current 40k target |

Migration principle:

- Do not import the full Innovator reward system.
- Reuse CodeVision reward plumbing.
- Port only missing answer-type scorers from Innovator as small functions.
- Keep reward routing based on `extra_info["reward_family"]` or `extra_info["answer_type"]`.

Implementation layout:

```text
recipe/codevision/reward.py
  Existing verl entry point. Keep this thin for new ToolVision RL data.

recipe/codevision/rewards/
  common.py        shared answer extraction, normalization, tool metadata
  rule_scorers.py  number, MCQ, boolean, OCR text, bbox IoU, relaxed chart scoring
  judge.py         lazy LLM judge wrapper
  router.py        source_dataset / answer_type / reward_family dispatch
```

New ToolVision RL data should be routed by:

```text
extra_info.source_dataset
extra_info.answer_type
extra_info.reward_family
```

The old benchmark/eval-specific reward logic can remain in `reward.py` for compatibility, but new dataset reward support should be added under `recipe/codevision/rewards/`.

LLM judge should use CodeVision's existing `LLMJudgeClient` interface. For DashScope-compatible API:

```bash
export LLM_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_JUDGE_MODEL_NAME=qwen3.6-plus
export LLM_JUDGE_API_KEY=...
```

Use LLM judge only as fallback for short-answer samples that cannot be scored reliably with rules.

## Validation Plan

Before full 40k conversion, run a small smoke set.

Smoke set:

- 10 samples per source where available
- single-image only
- no bytes images
- preserve all metadata

Checks:

1. Parquet schema loads through `CustomRLHFDataset`.
2. Image paths resolve through `qwen_vl_utils.fetch_image`.
3. Prompt contains exactly one `<image>` for single-image samples.
4. `reward_model.ground_truth` is non-empty.
5. `extra_info.source_dataset`, `answer_type`, `reward_family`, and hashes exist.
6. Reward returns a scalar/dict for each source.
7. Tool rollout can access the original image through `code_image_tool`.
8. Validation logs include `accuracy`, `format_reward`, `tool_count`, `tool_exec_error_count`.

## Proposed Work Order

1. Inspect and document the existing generated parquet schemas.
2. Build a source inventory with counts, answer types, image count distribution, and missing fields.
3. Define final CodeVision RL schema and metadata contract.
4. Implement dedup report only; do not drop data until counts are reviewed.
5. Implement 100-row smoke conversion.
6. Run dataloader/reward smoke.
7. Decide whether to regenerate ArxivQA to 2.5k or top up from control sources.
8. Convert full 40k candidate mix.
9. Produce final distribution report after dedup and caps.
10. Only then run GRPO.

## Open Questions

- Should ArxivQA be regenerated from 1k to 2.5k, or should the current 1k be accepted?
- What is the exact reward target for `refl4`, `virgorlsa`, and `sat2` in CodeVision: answer-only, bbox, or tool-process-aware?
- Should `docvqa` and `infographicvqa` use LLM judge fallback in the first GRPO run, or should they be excluded until rule scoring is stable?
- What per-image soft cap should be used for same-image multi-question sources?
- Should the first run enable tool reward shaping, or start with answer accuracy + format + simple tool penalties only?
