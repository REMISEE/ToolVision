# ToolVision RL Deduplication and Conversion Spec

Date: 2026-05-21

This document defines the concrete data-side workflow for building the 40k ToolVision/Innovator RL training set for CodeVision GRPO.

The goal is to convert mixed Innovator-style RL data into CodeVision/verl RL parquet while preserving enough metadata for reward routing, dedup audits, and later debugging.

## Inputs

### New ToolVision Generated Data

```text
/mnt/cpfs/delinmao/data/toolvision_innovator_scaled
```

Expected source folders:

```text
ai2d
arxivqa
chartqa
countqa
docvqa
infographicvqa
mmstar
ocrbench
pixmo_count
refl4
sat2
virgorlsa
```

Current schema:

```text
id
images
problem
answer
problem_type
answer_type
source
prompt_type
reward        optional
acc           optional
pred_rewards  optional
pred_accs     optional
```

### Original / Control Data

```text
/mnt/cpfs/delinmao/data/Innovator-VL-RL-172K/RL_part*.parquet
```

Expected schema:

```text
id
images
problem
answer
problem_type
answer_type
source
prompt_type
```

The control data must not be routed by `source` alone. It must be routed by per-row `answer_type`, with `problem_type` and `prompt_type` preserved as metadata.

## Target Mix

Final total: 40,000 rows.

### New ToolVision Subset

Subtotal: 24,000 rows.

| source_dataset | target |
|---|---:|
| chartqa | 4,790 |
| refl4 | 3,000 |
| virgorlsa | 3,000 |
| pixmo_count | 4,000 |
| sat2 | 2,500 |
| arxivqa | 2,500 |
| ocrbench | 1,269 |
| docvqa | 1,244 |
| infographicvqa | 1,001 |
| ai2d | 200 |
| countqa | 170 |
| mmstar | 326 |

`chartqa` carries the 290-row top-up because it is abundant and deterministic to score.

### Original / Control Subset

Subtotal: 16,000 rows.

| source_dataset | target | sampling note |
|---|---:|---|
| virl39k | 5,000 | stratify by `problem_type` + `answer_type` |
| WaltonFuture | 3,000 | normalize source from path to `WaltonFuture` |
| thinklite_vl_hard | 2,000 | source cap |
| tqa | 2,000 | source cap, keep answer_type stats |
| mmk12 | 1,500 | source cap |
| wemath_standard | 1,500 | source cap |
| puzzlevqa | 1,000 | source cap |

If a control source contains mixed `answer_type` values, sampling should preserve a report of the final distribution. For `virl39k`, do not sample all 5k from a single `answer_type` unless the candidate pool after filtering forces it.

## Processing Order

The conversion pipeline must run in this order:

```text
1. inventory raw candidates
2. normalize source names
3. parse questions and answers
4. validate images and filter unsupported rows
5. compute hashes and dedup keys
6. hard dedup
7. optional per-image soft cap
8. stratified target sampling
9. convert to CodeVision RL schema
10. write parquet + reports
11. run dataloader/reward smoke
```

Do not convert first and deduplicate later. Deduplication must happen on raw normalized fields so source ids, raw questions, and raw image identity remain auditable.

## Source Normalization

Write normalized source identity to:

```text
extra_info.source_dataset
data_source
```

Rules:

```text
source == "chartqa"                 -> chartqa
source == "refl4" or "ref_l4"       -> refl4
source == "virgorlsa"               -> virgorlsa
source == "pixmo_count"             -> pixmo_count
source == "sat2"                    -> sat2
source == "arxivqa"                 -> arxivqa
source == "ocrbench"                -> ocrbench
source == "docvqa"                  -> docvqa
source == "infographicvqa"          -> infographicvqa
source == "ai2d"                    -> ai2d
source == "countqa"                 -> countqa
source == "mmstar"                  -> mmstar
source == "virl39k"                 -> virl39k
source contains "WaltonFuture"      -> WaltonFuture
source contains "Multimodal-RL-Data" -> WaltonFuture
source == "thinklite_vl_hard"       -> thinklite_vl_hard
source == "tqa"                     -> tqa
source == "mmk12"                   -> mmk12
source == "wemath_standard"         -> wemath_standard
source == "puzzlevqa"               -> puzzlevqa
```

Keep the raw source in:

```text
extra_info.source_raw
```

## Question Normalization

Raw Innovator-style `problem` usually starts with `<image>`.

Conversion should:

1. Remove leading `<image>` placeholders from the text body.
2. Strip leading/trailing whitespace.
3. Preserve original text in `extra_info.raw_problem`.
4. Write cleaned text to `extra_info.question`.
5. Create CodeVision user prompt as:

```text
<image>Image size = {width}x{height} pixels.

{question}
```

Only one `<image>` placeholder is allowed in the first-round RL data.

## Image Policy

First run supports single-image rows only.

Filter out:

```text
image_count != 1
image missing
image path does not exist
image bytes cannot be decoded
image has zero width or height
```

Image source forms in raw parquet:

```python
{"bytes": b"..."}
{"bytes": None, "path": "/abs/path/to/image.jpg"}
```

Conversion policy:

- Prefer existing absolute path if present and readable.
- If only bytes are present, write the image to a deterministic local image cache.
- Final CodeVision parquet should use file references:

```python
"images": [
    {"image": "file:///absolute/path/to/image.jpg"}
]
```

Recommended image cache:

```text
/mnt/cpfs/delinmao/data/toolvision_codevision_40k/images/{source_dataset}/{image_hash}.{ext}
```

Use PNG for decoded byte images unless the original format can be safely inferred.

## Answer Normalization

Raw `answer` may be:

```text
numpy array
list
tuple
single string
stringified list, e.g. "['C']"
```

Convert it into:

```python
canonical_answer: str
acceptable_answers: list[str]
```

Rules:

### Multiple Choice

For `answer_type=multiple-choice`:

```text
canonical_answer = option letter, e.g. "C"
acceptable_answers = ["C"]
```

If the raw answer is `["C"]` or `"['C']"`, canonical answer must be `"C"`, not `"['C']"`.

If answer content is available and the option list can be parsed from the question, keep it in:

```text
extra_info.choices
```

### Number

For `answer_type=number`:

```text
canonical_answer = first valid answer as string
acceptable_answers = all normalized aliases
```

Do not strip units unless the source contract is known to be numeric-only.

### Boolean

For `answer_type=boolean`:

```text
canonical_answer = "true" or "false"
acceptable_answers = original aliases
```

Map yes/true to true and no/false to false.

### OCR Text / Short Text

For `answer_type=ocrtext` or short text:

```text
canonical_answer = first alias
acceptable_answers = all aliases
```

Examples:

```python
raw answer = ["$975.00", "975.00"]
canonical_answer = "$975.00"
acceptable_answers = ["$975.00", "975.00"]
```

### BBox

For `answer_type=bbox`:

```text
canonical_answer = "[x_min, y_min, x_max, y_max]"
acceptable_answers = all bbox aliases if any
```

Coordinates should remain in the dataset's stated range. For `refl4`, prompts state the range is 0 to 1, so do not convert to pixels.

### Math Expressions

For `answer_type=math-expressions`:

```text
canonical_answer = first valid math/string answer
acceptable_answers = all aliases
```

The reward uses `math_verify`.

### Judge

For `answer_type=judge`:

```text
canonical_answer = first answer
acceptable_answers = all aliases
reward_family = "judge"
```

Judge samples should still keep aliases because rule exact matching is attempted before calling LLM judge.

## Reward Family Mapping

Write `extra_info.reward_family` during conversion. Do not rely on source fallback when conversion can determine the answer type.

### Source Overrides

These sources override generic `answer_type=ocrtext` behavior:

| source_dataset | reward_family | reason |
|---|---|---|
| chartqa | chartqa_relaxed | ChartQA uses relaxed numeric accuracy |
| ocrbench | ocr_inclusion | OCRBench-style answers can be contained in longer predictions |

### Answer Type Mapping

| answer_type | reward_family |
|---|---|
| multiple-choice | multiple_choice |
| number | numeric_exact |
| boolean | boolean |
| ocrtext | ocr_levenshtein |
| bbox | bbox_iou |
| any | exact |
| math-expressions | math_verify |
| judge | judge |
| html-code | html_code |
| svg-code | svg_code |
| general-code | general_code |

`html-code`, `svg-code`, and `general-code` are not part of the current 40k target. If they appear unexpectedly, either exclude them or keep them only with judge enabled.

## Dedup Keys

Compute these fields before sampling:

```text
image_hash
question_hash
qa_hash
dedup_group
image_group_size
image_group_rank
```

### Normalized Question

```text
normalized_question = lowercase(
    collapse_whitespace(
        question without leading <image>
    )
)
```

Do not remove semantically meaningful punctuation from questions.

### Image Hash

For image bytes:

```text
image_hash = sha256(decoded image bytes or raw bytes)
```

For image paths:

```text
image_hash = sha256(file bytes)
```

If reading file bytes is too expensive, path-based hashes may be used for an initial report, but final conversion should use content hash.

### Question Hash

```text
question_hash = sha256(normalized_question)
```

### QA Hash

```text
qa_hash = sha256(normalized_question + "\n" + canonical_answer)
```

## Hard Deduplication

Drop duplicates by:

```text
source_dataset + source_original_id
image_hash + normalized_question
```

If two rows have the same `image_hash + normalized_question` but different answers, flag them as conflicts and exclude until reviewed.

Do not deduplicate by image alone. Same image with different questions is valid.

## Soft Image Cap

After hard deduplication, compute groups by:

```text
dedup_group = image_hash
```

Default policy:

```text
keep same-image different-question rows
do not apply soft cap unless a source is dominated by repeated images
```

If soft cap is needed, use per-source caps:

```text
docvqa / infographicvqa: max 8 questions per image
refl4: max 8 questions per image
chartqa: max 5 questions per image
other sources: no cap unless report shows concentration
```

Store group metadata even when no cap is applied.

## Sampling

Sampling happens after filtering and hard deduplication.

Use deterministic random seed:

```text
seed = 20260521
```

Sampling order:

1. Sample New ToolVision sources to fixed targets.
2. Sample original/control sources to fixed targets.
3. For mixed sources, stratify by `problem_type + answer_type` when feasible.
4. Write pre-sampling and post-sampling distribution reports.

For `virl39k`, use stratified sampling by:

```text
problem_type
answer_type
```

For source-cap control datasets such as `tqa`, `mmk12`, and `puzzlevqa`, still report final `answer_type` distribution.

## Output Schema

Each converted row must follow this CodeVision/verl schema:

```python
{
    "data_source": "chartqa",
    "agent_name": "tool_agent",
    "ability": "mm_qa",
    "prompt": [
        {
            "role": "user",
            "content": "<image>Image size = 1024x768 pixels.\n\nQuestion text"
        }
    ],
    "images": [
        {"image": "file:///absolute/path/to/image.png"}
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"
    },
    "extra_info": {
        "index": 0,
        "uid": "chartqa::chartqa_0000",
        "source_dataset": "chartqa",
        "source_raw": "chartqa",
        "source_original_id": "chartqa_0000",
        "source_split": "train",
        "question": "Question text",
        "raw_problem": "<image>\nQuestion text",
        "answer": "42",
        "acceptable_answers": ["42"],
        "answer_type": "number",
        "problem_type": "ocr",
        "prompt_type": "normal",
        "reward_family": "numeric_exact",
        "ability_bucket": "ocr_chart_document",
        "image_path": "/absolute/path/to/image.png",
        "image_width": 1024,
        "image_height": 768,
        "image_hash": "...",
        "question_hash": "...",
        "qa_hash": "...",
        "dedup_group": "...",
        "image_group_rank": 0,
        "image_group_size": 1,
        "conversion_version": "toolvision_rl_40k_v1"
    }
}
```

## Ability Buckets

Write `extra_info.ability_bucket` using this mapping:

| ability_bucket | sources / conditions |
|---|---|
| ocr_chart_document | chartqa, ocrbench, docvqa, infographicvqa |
| counting | pixmo_count, countqa |
| grounding_visual_search | refl4, virgorlsa |
| spatial_diagram | sat2, ai2d, mmk12 if selected |
| stem_math_science | arxivqa, wemath_standard, math-expressions rows |
| general_visual_reasoning | virl39k, WaltonFuture, thinklite_vl_hard, tqa, puzzlevqa, mmstar |

If a source has a more precise `problem_type`, prefer the more specific bucket.

## Output Files

Recommended output root:

```text
/mnt/cpfs/delinmao/data/toolvision_codevision_rl_40k
```

Expected files:

```text
train.parquet
smoke_120.parquet
reports/raw_inventory_by_source.csv
reports/raw_inventory_by_type.csv
reports/filter_report.csv
reports/dedup_report.csv
reports/final_mix_by_source.csv
reports/final_mix_by_reward_family.csv
reports/final_mix_by_answer_type.csv
reports/image_group_report.csv
reports/schema_report.json
```

Smoke set should contain up to 10 rows per source and cover every reward family present in the final set.

## Validation Checklist

Before training, verify:

1. Row count is exactly 40,000 for `train.parquet`.
2. All rows are single-image.
3. Every image path resolves and can be opened.
4. Every prompt has exactly one `<image>`.
5. Every prompt includes `Image size = WxH pixels`.
6. `reward_model.ground_truth` is non-empty.
7. `extra_info.answer_type` is non-empty.
8. `extra_info.reward_family` is non-empty.
9. No unsupported reward family appears unless explicitly approved.
10. `compute_score` returns a dict for every smoke row.
11. Dataloader loads with `CustomRLHFDataset`.
12. Tool kwargs are auto-generated for `code_image_tool`.

## Reward Smoke Cases

The smoke set must include at least one row for each present family:

```text
chartqa_relaxed
ocr_inclusion
ocr_levenshtein
numeric_exact
multiple_choice
boolean
exact
bbox_iou
math_verify
judge
```

For `judge`, run two modes:

```text
without LLM_JUDGE_BASE_URL: exact fallback only
with LLM_JUDGE_BASE_URL: actual judge request
```

## Failure Policy

Rows should be excluded, not silently repaired, when:

```text
image cannot be decoded
question is empty after normalization
answer is empty
answer_type is missing
reward_family cannot be assigned
multiple-choice answer cannot be parsed
bbox answer cannot be parsed
same image + question has conflicting answers
```

Write excluded rows to a report with:

```text
source_dataset
source_original_id
reason
raw_problem
raw_answer
image_reference
```

## Next Implementation Steps

1. Build an inventory script that reads both input roots and emits source/type reports.
2. Build a dry-run dedup script that writes reports but no final parquet.
3. Review final candidate counts after hard dedup.
4. Build the converter for `smoke_120.parquet`.
5. Run dataloader and reward smoke.
6. Convert full `train.parquet`.
7. Run final validation checklist.
