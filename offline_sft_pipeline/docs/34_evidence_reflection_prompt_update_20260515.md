# Evidence Reflection Prompt Update

Date: 2026-05-15

## Summary

This update keeps the existing plan-and-execute pipeline shape, but makes the planner and executor reasoning less free-form.

The goal is not to optimize for a single dataset or sample. The goal is to make every tool-use step carry an explicit evidence update:

1. what is visible now
2. what evidence has already been obtained
3. what the latest tool result added or failed to add
4. what uncertainty remains
5. why the next action is answer or another tool branch

The old prompt files are preserved:

- `offline_sft_pipeline/prompts/planner_system_v05.txt`
- `offline_sft_pipeline/prompts/executor_system_v03.txt`

The new prompt files are:

- `offline_sft_pipeline/prompts/planner_system_v06.txt`
- `offline_sft_pipeline/prompts/executor_system_v04.txt`

The follow-up prompt files are:

- `offline_sft_pipeline/prompts/planner_system_v07.txt`
- `offline_sft_pipeline/prompts/executor_system_v05.txt`

## Why This Was Changed

The previous planner prompt asked for evidence and uncertainty, but did not require a stable evidence-delta structure. As a result, planner CoT could become generic, such as "crop to improve readability" or "run OCR to reduce uncertainty", without proving that the previous tool result changed the evidence state.

The executor prompt had a similar issue. The executor can see:

- full message history
- visible image timeline
- planner global CoT
- suggestion CoT
- current step goal
- capability plan

However, the executor's CoT is the part that is written into trajectory `messages.json`. If it does not absorb the planner's evidence state, planner reflection is mostly hidden and does not become training data.

The new prompts therefore keep the same schema, but make the reasoning sections more explicit.

## What Changed In Planner Prompt

The planner output schema is unchanged:

```json
{
  "mode": "answer",
  "think": "...",
  "answer": "..."
}
```

or:

```json
{
  "mode": "suggestions",
  "think": "...",
  "suggestions": []
}
```

The `think` field is now expected to contain a compact reflection:

1. `Current visual state`
2. `Evidence so far`
3. `Latest evidence update`
4. `Remaining uncertainty`
5. `Decision`

This is intentionally general. It does not say to distrust OCR, detection, counting, depth, crop, or enhancement. Instead, it says helper outputs are evidence observations and should be combined with visible image context when the task depends on spatial layout, visual identity, color, ordering, or relations.

The `suggestion_cot` guidance now asks each branch to state:

- what uncertainty it tests
- what new evidence it expects to obtain or verify
- why it is not merely a repeat of an already attempted tool result

## What Changed In Executor Prompt

The executor output schema is unchanged:

```json
{
  "think": "...",
  "tool_call": {
    "name": "code_image_tool",
    "arguments": {
      "code": "...",
      "description": "..."
    }
  }
}
```

The executor `think` is now expected to include:

1. `Evidence already available`
2. `Step uncertainty`
3. `Input image choice`
4. `Expected evidence`
5. `Operation rationale`

This is important because executor CoT is persisted into `messages.json`. The executor should absorb planner context and restate it as its own step-level reasoning, without copying hidden fields verbatim.

## What Changed In Thresholds

The planner policy thresholds were changed from:

- `must_suggest_score_threshold = 0.7`
- `must_answer_score_threshold = 0.9`

to:

- `must_suggest_score_threshold = 0.25`
- `must_answer_score_threshold = 0.75`

The first planner round still remains `MUST_SUGGEST`.

This keeps the intended behavior that every generated trajectory uses at least one tool step, while allowing difficult examples where only a few strong judges solve the sample to enter `MAY_ANSWER_OR_SUGGEST` earlier. The prompt update is paired with this threshold change so the middle band does not become "always answer immediately"; the planner still has to reason about evidence update and remaining uncertainty.

## Low-Score Termination Fix

The stop policy already had no-improvement logic, but it was too weak in the branching setup because each executor step forks a new child trajectory. The fix makes stop-policy scoring look across the actual parent trajectory lineage rather than only the current child trajectory's local judge records.

This means a branch can still continue after one weak tool result, but repeated low scores with no improvement along the same lineage can stop earlier instead of using the full budget.

## Tool List Alignment

The old tool list was:

- `offline_sft_pipeline/example/tool_capabilities.json`

The new default tool list is:

- `offline_sft_pipeline/example/tool_capabilities_code_image_tool_v04.json`

It is aligned with the current `CodeImageTool` helper surface:

- `_call_ocr_assist`
- `_call_manual_box`
- `_call_manual_crop`
- `_call_ground_box`
- `_call_sam_mask`
- `_call_dino_crop`
- `_call_blur_bg`
- `_call_manual_depth`
- `_call_ground_depth`
- `_call_count_assist`

It also documents direct PIL operations that are available in the safe execution environment, such as rotate, flip, resize, brightness, contrast, and sharpness.

The runtime still creates `CodeImageTool` via `build_default_code_image_tool_config(...)`. This update does not make the JSON tool list the runtime source of truth; it aligns the prompt-visible capability list with the current runtime helper names and signatures.

## Expected Behavioral Difference

This is a prompt discipline change, not a pipeline rewrite.

Unchanged:

- planner/executor JSON schemas
- branch count
- runtime tool interface
- first-round `MUST_SUGGEST`
- judge backend

Changed:

- planner and executor CoT should be less generic
- tool calls should more often be justified by expected new evidence
- repeated low-information branches should stop earlier
- middle-score trajectories can answer earlier when evidence is sufficient

The main expected data shift is better evidence-aware CoT, not a rigid labeled style. Executor CoT is intentionally kept as natural prose because it is the part that enters the retained training trajectory. The change is deliberately dataset-neutral and avoids adding ChartQA-specific rules.

## Follow-Up: Grounded Evidence Association

The first v06.1 smoke exposed a general failure mode: a tool can extract useful text or numbers while losing the visual association that the question depends on. In `chartqa__train__1`, OCR extracted the chart numbers, but flattened their relationship to the colored series. The planner then treated the OCR ordering as the series assignment and answered from the wrong visual target.

The v07 / v05 prompts address this without specializing to ChartQA:

- Tool outputs remain evidence observations, not authority by themselves.
- OCR text may flatten or reorder spatial and visual elements.
- When a question depends on label, color, position, object identity, or relation, planner reasoning must align tool outputs back to the visible target before answering.
- Answer-mode planner `think` is treated as a final training trace: it should combine image evidence and tool observations, resolve association issues, and conclude naturally without citing policy, judge, or meta stopping language.
- Executor `think` stays natural, but when it uses OCR, detection, or crop for a visually defined target, it should state the association the tool result needs to preserve or verify.
