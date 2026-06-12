"""OpenAI-compatible multimodal chat for DashScope (Qwen) planner/executor calls.

POST {base_url}/chat/completions with Authorization: Bearer.

Environment (do not hardcode secrets):
- OFFLINE_SFT_QWEN_API_KEY
- OFFLINE_SFT_QWEN_BASE_URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
- OFFLINE_SFT_QWEN_MODEL
- OFFLINE_SFT_QWEN_TIMEOUT_S (seconds, default 120)
- OFFLINE_SFT_API_DRY_RUN (1/true/yes: skip HTTP, return local fake planner text from caller)
- OFFLINE_SFT_PLANNER_DEBUG (1/true/yes: print to stderr the JSON messages sent to the API with base64
  image URLs shortened; choices[0].message with content replaced by length (to spot reasoning/extra fields);
  then the full assistant text; dry_run path does not call HTTP so no debug dump)
- OFFLINE_SFT_PLANNER_USE_TOOL_ROLE (1/true/yes: send internal tool-result messages as native role="tool"
  instead of wrapping them as user-visible environment observations)
- OFFLINE_SFT_EXECUTOR_DEBUG (1/true/yes: same as planner debug, but for executor calls)
- OFFLINE_SFT_EXECUTOR_USE_TOOL_ROLE (1/true/yes: same as planner tool-role behavior, but for executor calls)

Reference: https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from offline_sft_pipeline.core.models import ConversationMessage, ImageArtifactRef
from .request_models import ExecutorClientRequest, JudgeClientRequest, PlannerClientRequest, ToolCapability

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.6-plus"
DEFAULT_QWEN_TIMEOUT_S = 200.0

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def env_qwen_config() -> dict[str, Any]:
    return {
        "api_key": os.environ.get("OFFLINE_SFT_QWEN_API_KEY"),
        "base_url": os.environ.get("OFFLINE_SFT_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/"),
        "model": os.environ.get("OFFLINE_SFT_QWEN_MODEL", DEFAULT_QWEN_MODEL),
        "timeout_s": float(os.environ.get("OFFLINE_SFT_QWEN_TIMEOUT_S", str(DEFAULT_QWEN_TIMEOUT_S))),
        "dry_run": os.environ.get("OFFLINE_SFT_API_DRY_RUN", "").strip().lower() in {"1", "true", "yes"},
    }


def env_planner_debug_enabled() -> bool:
    return os.environ.get("OFFLINE_SFT_PLANNER_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def env_planner_use_tool_role() -> bool:
    return os.environ.get("OFFLINE_SFT_PLANNER_USE_TOOL_ROLE", "").strip().lower() in {"1", "true", "yes"}


def env_executor_debug_enabled() -> bool:
    return os.environ.get("OFFLINE_SFT_EXECUTOR_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def env_executor_use_tool_role() -> bool:
    return os.environ.get("OFFLINE_SFT_EXECUTOR_USE_TOOL_ROLE", "").strip().lower() in {"1", "true", "yes"}


def sanitize_messages_for_debug(payload: Any) -> Any:
    """Recursively shorten data:...;base64,... strings so debug logs stay readable."""

    def _shorten_data_url(s: str) -> str:
        if "data:" in s and "base64," in s and len(s) > 160:
            return s[:80] + f"...[base64 omitted, total_len={len(s)}]"
        return s

    if isinstance(payload, str):
        return _shorten_data_url(payload)
    if isinstance(payload, dict):
        return {k: sanitize_messages_for_debug(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_messages_for_debug(v) for v in payload]
    return payload


def is_placeholder_api_key(api_key: str | None) -> bool:
    if api_key is None:
        return True
    k = api_key.strip().lower()
    return k in {"", "dummykey", "dummy", "placeholder"}


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")


def file_to_data_url(path: Path, *, media_type: str | None = None) -> str:
    raw = path.read_bytes()
    mt = media_type or _guess_media_type(path)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mt};base64,{b64}"


def _resolve_runtime_image_path(*, runtime_result_path: Path, raw_path: str) -> Path | None:
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = (runtime_result_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_file():
        return candidate

    fallback = runtime_result_path.parent / Path(str(raw_path)).name
    if fallback.is_file():
        return fallback.resolve()
    return None


def _index_runtime_result_images(
    *,
    index: dict[str, Path],
    runtime_result_path: Path,
    seen_runtime_results: set[Path],
) -> None:
    resolved_runtime_result = runtime_result_path.resolve()
    if resolved_runtime_result in seen_runtime_results or not resolved_runtime_result.is_file():
        return
    seen_runtime_results.add(resolved_runtime_result)

    try:
        data = json.loads(resolved_runtime_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    for img in data.get("images") or []:
        if not isinstance(img, dict):
            continue
        aid = img.get("artifact_id")
        raw_path = img.get("path")
        if not aid or not raw_path:
            continue
        resolved_image_path = _resolve_runtime_image_path(
            runtime_result_path=resolved_runtime_result,
            raw_path=str(raw_path),
        )
        if resolved_image_path is not None:
            index[str(aid)] = resolved_image_path


def _iter_runtime_result_paths_from_trajectory(trajectory_dir: Path) -> list[Path]:
    trajectory_json = trajectory_dir / "trajectory.json"
    if not trajectory_json.is_file():
        return []

    try:
        payload = json.loads(trajectory_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []

    def _step_sort_key(item: Any) -> tuple[int, str]:
        if not isinstance(item, dict):
            return (10**9, "")
        raw_step_idx = item.get("step_idx")
        try:
            step_idx = int(raw_step_idx)
        except (TypeError, ValueError):
            step_idx = 10**9
        raw_runtime_result_path = item.get("runtime_result_path")
        return (step_idx, str(raw_runtime_result_path or ""))

    runtime_result_paths: list[Path] = []
    for step in sorted(raw_steps, key=_step_sort_key):
        if not isinstance(step, dict):
            continue
        raw_runtime_result_path = step.get("runtime_result_path")
        if not raw_runtime_result_path:
            continue
        runtime_result_path = Path(str(raw_runtime_result_path))
        if not runtime_result_path.is_absolute():
            runtime_result_path = (trajectory_dir / runtime_result_path).resolve()
        runtime_result_paths.append(runtime_result_path)
    return runtime_result_paths


def build_artifact_path_index(
    *,
    sample_dir: str | None,
    trajectory_dir: str | None,
    visible_images: list[ImageArtifactRef],
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ref in visible_images:
        if not ref.artifact_id:
            continue
        p = Path(ref.path)
        if p.is_file():
            index[ref.artifact_id] = p.resolve()

    if sample_dir:
        art = Path(sample_dir) / "artifacts"
        if art.is_dir():
            for f in art.iterdir():
                if not f.is_file() or f.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                index.setdefault(f.stem, f.resolve())

    if trajectory_dir:
        trajectory_root = Path(trajectory_dir)
        seen_runtime_results: set[Path] = set()

        for runtime_result_path in _iter_runtime_result_paths_from_trajectory(trajectory_root):
            _index_runtime_result_images(
                index=index,
                runtime_result_path=runtime_result_path,
                seen_runtime_results=seen_runtime_results,
            )

        steps_root = trajectory_root / "steps"
        if steps_root.is_dir():
            for step_dir in sorted(steps_root.iterdir()):
                if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
                    continue
                rr_path = step_dir / "runtime_result.json"
                _index_runtime_result_images(
                    index=index,
                    runtime_result_path=rr_path,
                    seen_runtime_results=seen_runtime_results,
                )
    return index


def _finalize_openai_content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0].get("text", ""))
    return parts


def _openai_content_parts_for_message(
    message: ConversationMessage,
    index: dict[str, Path],
    missing: list[str],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for aid in message.image_artifact_ids:
        p = index.get(aid)
        if p is None or not p.is_file():
            missing.append(aid)
            parts.append({"type": "text", "text": f"[missing image artifact_id={aid!r}]"})
            continue
        parts.append({"type": "image_url", "image_url": {"url": file_to_data_url(p)}})
        try:
            with Image.open(p) as image:
                width, height = image.size
            parts.append(
                {
                    "type": "text",
                    "text": (
                        f"Image geometry: width={int(width)}, height={int(height)}. "
                        "Coordinate origin is top-left; x increases to the right, y increases downward."
                    ),
                }
            )
        except OSError:
            pass
    text = message.content or ""
    if text.strip() or not parts:
        parts.append({"type": "text", "text": text})
    return parts


def _format_capabilities(caps: list[ToolCapability]) -> str:
    if not caps:
        return (
            "(none configured — use only capability names that your runtime exposes; "
            "load example/tool_capabilities_code_image_tool_v04.json for production runs.)"
        )
    blocks: list[str] = []
    for c in caps:
        desc = str(c.description or "").strip().replace("\n", "\n  ")
        block = f"- capability `{c.name}`\n  {desc}"
        if c.usage_notes:
            usage_notes = str(c.usage_notes).strip().replace("\n", "\n  ")
            block += f"\n  Usage notes: {usage_notes}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_image_geometry(image: ImageArtifactRef) -> str:
    width = image.width
    height = image.height
    if width is None or height is None:
        try:
            with Image.open(Path(image.path)) as pil_image:
                width, height = pil_image.size
        except OSError:
            width, height = None, None
    if width is None or height is None:
        return "size unknown"
    return f"size=({int(width)}, {int(height)})"


def _format_visible_image_timeline(visible_images: list[ImageArtifactRef]) -> str:
    lines = [
        "- Visible image indices are append-only over the executed trajectory.",
        "- Initial/root images come first; each previous tool-result image is appended in execution order.",
        "- Use these exact indices when choosing `input_image_index` or helper `image_index=`.",
    ]
    for idx, image in enumerate(visible_images):
        lines.append(
            f"- image index {idx}: artifact_id=`{image.artifact_id}`; {_format_image_geometry(image)}"
        )
    return "\n".join(lines)


def _planner_policy_mode(req: PlannerClientRequest) -> str:
    if req.must_answer_now:
        return "must_answer"
    metadata_policy = str(req.metadata.get("planning_policy") or "").strip().lower()
    if metadata_policy in {"must_suggest", "may_answer_or_suggest", "must_answer"}:
        return metadata_policy
    if bool(req.metadata.get("must_suggest_now")):
        return "must_suggest"
    return "may_answer_or_suggest"


def _build_planner_round_policy_block(
    req: PlannerClientRequest,
    *,
    policy_mode: str,
    suggestion_count: int,
    forced_final_answer: dict[str, Any] | None,
    judge_consensus_answer_hint: dict[str, Any] | None,
) -> str:
    lines = ["Round policy:"]
    if policy_mode == "must_answer":
        lines.extend(
            [
                "- This round is `MUST_ANSWER`.",
                "- Return `mode=\"answer\"`.",
                "- Do not return any `suggestions` field.",
                "- Finalize the best answer now from the visible evidence and executed trajectory above.",
            ]
        )
        if isinstance(forced_final_answer, dict):
            lines.append(
                "- The executed trajectory appears sufficient; answer from the visible evidence without citing any hidden policy or evaluation signal."
            )
        if isinstance(judge_consensus_answer_hint, dict):
            candidate_answer = str(judge_consensus_answer_hint.get("candidate_answer") or "").strip()
            if candidate_answer:
                lines.append(
                    "- Independent judge consensus from the executed trajectory proposes this final answer: "
                    f"`{candidate_answer}`. Treat it as a high-confidence candidate from the visible/tool evidence; "
                    "verify it against the visible image before finalizing."
                )
        return "\n".join(lines)

    if policy_mode == "must_suggest":
        lines.extend(
            [
                "- This round is `MUST_SUGGEST`.",
                "- Return `mode=\"suggestions\"`.",
                "- Do not return any `answer` field.",
                f"- The top-level `suggestions` array must contain exactly {suggestion_count} branch objects.",
                "- Each suggestion must be an executable alternative strategy branch grounded in the evidence above.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
            [
                "- This round is `MAY_ANSWER_OR_SUGGEST`.",
                "- You must choose exactly one mode: `answer` or `suggestions`.",
                "- Prefer `answer` mode when the evidence so far resolves the main uncertainty and important tool observations are grounded back to the visible image.",
                "- Use `suggestions` mode when a specific uncertainty remains and another tool step is likely to provide new evidence or useful verification.",
                "- Avoid suggesting a tool only to make the image look cleaner unless that cleanup is expected to reveal task-relevant information.",
                f"- If you choose `suggestions`, the top-level `suggestions` array must contain exactly {suggestion_count} branch objects.",
        ]
    )
    return "\n".join(lines)


def _build_planner_answer_format_block(req: PlannerClientRequest, *, policy_mode: str) -> str:
    if policy_mode == "must_suggest" or not req.answer_instruction:
        return ""
    lines = [
        "Answer format constraint:",
        f"- If you return `mode=\"answer\"`, the `answer` field must follow this instruction: {req.answer_instruction}",
    ]
    if policy_mode == "may_answer_or_suggest":
        lines.append("- If you return `mode=\"suggestions\"`, do not emit any `answer` field.")
    return "\n".join(lines)


def _build_planner_budget_block(req: PlannerClientRequest) -> str:
    remaining = int(req.budget.remaining_exec_steps)
    lines = [
        "Budget constraints:",
        f"- `remaining_exec_steps = {remaining}` is the total number of executor steps still available on this trajectory before the final answer.",
        "- Every suggested branch must fit within this remaining budget.",
    ]
    if remaining <= 1:
        lines.append("- If you choose `suggestions`, each branch should contain exactly one immediate next step.")
    elif remaining == 2:
        lines.append("- If you choose `suggestions`, keep each branch very short, typically one or two steps at most.")
    else:
        lines.append("- Prefer short, high-value branches. Do not spend steps on low-information detours.")
    lines.append("- When the executable budget is exhausted, the caller will switch to a final-answer round.")
    return "\n".join(lines)


def _build_planner_guidance_block() -> str:
    return (
        "Planning guidance:\n"
        "- The top-level `think` is the round-level Global CoT: in natural prose, summarize the current visual state, evidence so far, latest evidence update, remaining uncertainty, and decision.\n"
        "- Do not make the `think` a rigid labeled checklist unless the task is unusually complex.\n"
        "- Treat OCR, detection, counting, crop, depth, and image enhancement outputs as evidence observations to combine with the visible image context when relevant.\n"
        "- OCR text can flatten or reorder visual elements; when labels, colors, positions, or object identities matter, align extracted text back to the visible target before answering.\n"
        "- In answer mode, write the `think` like final assistant reasoning for training: combine image evidence and tool observations, resolve any association issue, then conclude without citing hidden policy or judge signals.\n"
        "- Each `suggestion_cot` is branch-level reasoning: explain why that branch is promising, what uncertainty it tests, and what new evidence it expects.\n"
        "- Each `step_goal` is the exact local objective of one step in that branch.\n"
        "- Each `executor_instruction` is short executor-facing implementation guidance for that step.\n"
        "- Each `input_image_index` selects the default starting image for that step from the visible image timeline in the user prompt.\n"
        "- When a question depends on two distinct targets or regions, it is often better to revisit the original/full-scene image index for separate high-value calls on each target, then combine the evidence, instead of overcommitting to one narrow crop.\n"
        "- A suggestion is a full strategy branch, not just a single tool name.\n"
        "- Different suggestions should differ in branch logic, not just wording.\n"
        "- If a branch intentionally revisits an earlier image index instead of the latest visible one, make that source switch explicit in the reasoning.\n"
        "- Do not invent unsupported capabilities or fabricate tool effects.\n"
    )


def _build_planner_dataset_specific_guidance(req: PlannerClientRequest) -> str:
    sample_id = str(req.sample_id or "").strip().lower()
    if sample_id.startswith("cavqa_multichoice__"):
        return (
            "Dataset-specific guidance:\n"
            "- For this question, the coordinates written in the question are only hints and are not always accurate.\n"
            "- When the question ties an object name to a provided box, prefer proposing both a coordinate-based branch and a grounding/detection-based verification branch for the same object.\n"
            "- If the boxed region does not visually match the named object, explicitly rethink the localization from the image content and correct the interpretation before answering.\n"
        )
    return ""


def _format_planner_capability_plan(req: ExecutorClientRequest) -> str:
    lines = [
        "- These are the capability operations intended for this step.",
        "- Use them as the primary execution direction unless there is a strong local reason to compose them within the same step more explicitly.",
    ]
    for item in req.step_spec.capability_plan:
        lines.append(
            f"- order {item.order}: capability=`{item.capability}`; instruction={item.instruction}"
        )
    return "\n".join(lines)


def _conversation_to_openai_messages(
    *,
    messages: list[ConversationMessage],
    index: dict[str, Path],
    missing: list[str],
    use_tool_role: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "tool":
            parts = _openai_content_parts_for_message(msg, index, missing)
            if use_tool_role:
                out.append({"role": "tool", "content": _finalize_openai_content(parts)})
                continue
            wrapped: list[dict[str, Any]] = [
                {"type": "text", "text": "[Tool output — environment observation, not user speech]\n"},
            ]
            wrapped.extend(parts)
            out.append({"role": "user", "content": wrapped})
            continue
        parts = _openai_content_parts_for_message(msg, index, missing)
        out.append({"role": msg.role, "content": _finalize_openai_content(parts)})
    return out


def build_planner_control_user_text(req: PlannerClientRequest) -> str:
    suggestion_count = int(req.requested_suggestion_count or 1)
    policy_mode = _planner_policy_mode(req)
    forced_final_answer = req.metadata.get("forced_final_answer")
    judge_consensus_answer_hint = req.metadata.get("judge_consensus_answer_hint")
    round_policy_block = _build_planner_round_policy_block(
        req,
        policy_mode=policy_mode,
        suggestion_count=suggestion_count,
        forced_final_answer=forced_final_answer if isinstance(forced_final_answer, dict) else None,
        judge_consensus_answer_hint=(
            judge_consensus_answer_hint if isinstance(judge_consensus_answer_hint, dict) else None
        ),
    )
    answer_format_block = _build_planner_answer_format_block(req, policy_mode=policy_mode)
    budget_block = _build_planner_budget_block(req)
    guidance_block = _build_planner_guidance_block()
    dataset_guidance_block = _build_planner_dataset_specific_guidance(req)
    answer_format_section = f"{answer_format_block}\n\n" if answer_format_block else ""
    dataset_guidance_section = f"{dataset_guidance_block}\n" if dataset_guidance_block else ""
    return (
        "The conversation above already contains the original user question, "
        "visible images, prior assistant actions, and tool outputs.\n\n"
        "Current visible image timeline:\n"
        f"{_format_visible_image_timeline(req.visible_images)}\n\n"
        f"{round_policy_block}\n\n"
        f"{answer_format_section}"
        f"{budget_block}\n"
        f"{guidance_block}\n"
        f"{dataset_guidance_section}"
        "Capability reference (use only these names in `capability_plan`):\n"
        f"{_format_capabilities(req.tool_capabilities)}"
    )


def planner_to_openai_messages(
    *,
    system_prompt: str,
    req: PlannerClientRequest,
) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    index = build_artifact_path_index(
        sample_dir=req.sample_dir,
        trajectory_dir=req.trajectory_dir,
        visible_images=req.visible_images,
    )
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    out.extend(
        _conversation_to_openai_messages(
            messages=req.messages,
            index=index,
            missing=missing,
            use_tool_role=env_planner_use_tool_role(),
        )
    )
    out.append({"role": "user", "content": build_planner_control_user_text(req)})
    deduped_missing = list(dict.fromkeys(missing))
    return out, deduped_missing


def build_judge_system_prompt(system_prompt: str, req: JudgeClientRequest) -> str:
    prompt = str(system_prompt or "").strip()
    if req.answer_instruction:
        prompt += (
            "\n\n"
            "Follow the answer instruction:\n"
            f"{req.answer_instruction}\n"
            "- The answer instruction is a format constraint. Do not repeat or quote the instruction itself.\n\n"
        )
    return prompt


def build_judge_control_user_text(req: JudgeClientRequest) -> str:
    return (
        "Judge task:\n"
        "- Solve the original user question using the trajectory and visible evidence above.\n"
        "- Do not explain your reasoning.\n"
        "- Return only the final answer text.\n\n"
    )


def judge_to_openai_messages(
    *,
    system_prompt: str,
    req: JudgeClientRequest,
) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    index = build_artifact_path_index(
        sample_dir=req.sample_dir,
        trajectory_dir=req.trajectory_dir,
        visible_images=req.visible_images,
    )
    out: list[dict[str, Any]] = [{"role": "system", "content": build_judge_system_prompt(system_prompt, req)}]
    out.extend(
        _conversation_to_openai_messages(
            messages=req.messages,
            index=index,
            missing=missing,
            use_tool_role=False,
        )
    )
    out.append({"role": "user", "content": build_judge_control_user_text(req)})
    deduped_missing = list(dict.fromkeys(missing))
    return out, deduped_missing


def _normalize_instruction(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().split()).lower()


def _format_executor_visible_images(req: ExecutorClientRequest) -> str:
    role_lines: list[str] = []
    for idx, image in enumerate(req.visible_images):
        size_text = _format_image_geometry(image)
        role_lines.append(
            f"  - local image index {idx}: artifact_id=`{image.artifact_id}`; {size_text}"
        )

    default_index = req.step_spec.input_image_index
    intro = [
        "- These are the images you may rely on in this step.",
        "- Local image indices are append-only across the executed trajectory: initial/root images first, then prior tool-result images.",
    ]
    if len(req.visible_images) <= 1:
        intro.append(
            f"- Only one visible image is available in this step, and it is already bound as the default `image` / `img` (index {default_index})."
        )
    else:
        intro.append(
            f"- The default starting image already bound to `image` / `img` is local image index {default_index}."
        )
        intro.append("- If you explicitly need a non-default visible image, use helper `image_index=` with the local mapping below.")
    return "\n".join(intro + role_lines)


def _format_executor_image_usage(req: ExecutorClientRequest) -> str:
    default_index = req.step_spec.input_image_index
    lines = [
        "- Start from the already bound default image unless another visible image is explicitly needed.",
        f"- In this step, the default bound image is local image index {default_index}.",
        "- If you switch to another visible image, use helper `image_index=` with the local mapping above.",
        "- If you intentionally revisit an earlier image index instead of the latest visible image, state that reason explicitly in `think`.",
        '- If you continue processing an intermediate helper output within this step, prefer `image_obj=prev["image"]`.',
        "- Do not load images from files or URLs.",
        "- Assign the final chosen image to `result`.",
    ]
    return "\n".join(lines)


def _format_executor_tool_helpers(req: ExecutorClientRequest) -> str:
    if not req.tool_capabilities:
        return (
            "- No tool helper definitions were provided.\n"
            "- In that case, use only helper names already made explicit elsewhere in the prompt."
        )

    blocks: list[str] = [
        "- Use only the real helper functions described below.",
        "- Call helpers exactly by their provided names such as `_call_manual_crop`, `_call_ground_box`, `_call_ocr_assist`.",
        "- Do not invent shortened helper names such as `manual_crop(...)`.",
    ]
    for cap in req.tool_capabilities:
        blocks.append(f"\nHelper `{cap.name}`")
        blocks.append(f"{cap.description}")
        if cap.usage_notes:
            blocks.append(f"Notes:\n{cap.usage_notes}")
    return "\n".join(blocks)


def build_executor_control_user_text(req: ExecutorClientRequest) -> str:
    normalized_goal = _normalize_instruction(req.step_spec.step_goal)
    normalized_executor_instruction = _normalize_instruction(req.step_spec.executor_instruction)
    extra_instruction = ""
    if normalized_executor_instruction and normalized_executor_instruction != normalized_goal:
        extra_instruction = f"- Additional local instruction: {req.step_spec.executor_instruction.strip()}\n"

    return (
        "The conversation above is the executed trajectory so far "
        "(user question, prior assistant actions, and tool outputs).\n\n"
        "Your task now is to produce the next single-step execution trace.\n\n"
        "Execution requirements for this step:\n"
        "- Return exactly one JSON object with `think` and `tool_call`.\n"
        "- Do not answer the question directly.\n"
        "- Produce a self-contained step-level rationale for this step.\n\n"
        "How to use the hidden context:\n"
        "- Your `think` must read as your own direct reasoning for the current step.\n"
        "- Do not mention or quote any upstream planning process.\n"
        "- Do not use words such as `planner`, `suggestion`, `branch`, `guidance`, `Global CoT`, `Suggestion CoT`, `Step Goal`, or `executor instruction`.\n"
        "- Write `think` as natural reasoning for training a model, not as a numbered checklist or labeled form.\n"
        "- Absorb the hidden context silently and restate it only as your own evidence summary, uncertainty, expected evidence, and next-step rationale.\n"
        "- If this step uses OCR, detection, or a crop for a target defined by a label, color, position, object, or relation, state the association the tool result needs to preserve or verify.\n"
        "- If you use a manual coordinate tool, describe the box or coordinates only as your current localization judgment from the question and image, never as something supplied by an upstream process.\n"
        "- Do not simply repeat the fields below. Convert them into your own reasoning about what is visible now, what remains uncertain, and why the next tool action is justified.\n\n"
        "Hidden context for the current step:\n"
        f"- Current reasoning context: {req.planner_global_chain_cot or '(none provided)'}\n"
        f"- Current action rationale: {req.suggestion_cot or '(none provided)'}\n"
        f"- Current objective: {req.step_spec.step_goal}\n"
        f"{extra_instruction}"
        "\n"
        "Capability plan for this step:\n"
        f"{_format_planner_capability_plan(req)}\n\n"
        "Current visual inputs available for this step:\n"
        f"{_format_executor_visible_images(req)}\n\n"
        "How image usage works in this step:\n"
        f"{_format_executor_image_usage(req)}\n\n"
        "Tool helpers you may use in code:\n"
        f"{_format_executor_tool_helpers(req)}"
    )


def executor_to_openai_messages(
    *,
    system_prompt: str,
    req: ExecutorClientRequest,
) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    index = build_artifact_path_index(
        sample_dir=req.sample_dir,
        trajectory_dir=req.trajectory_dir,
        visible_images=req.visible_images,
    )
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    out.extend(
        _conversation_to_openai_messages(
            messages=req.messages,
            index=index,
            missing=missing,
            use_tool_role=env_executor_use_tool_role(),
        )
    )
    out.append({"role": "user", "content": build_executor_control_user_text(req)})
    deduped_missing = list(dict.fromkeys(missing))
    return out, deduped_missing


def assistant_text_from_chat_response(raw: dict[str, Any]) -> str:
    choice0 = (raw.get("choices") or [{}])[0]
    msg = choice0.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
        return "".join(chunks)
    return str(content or "")


def summarize_openai_message_for_debug(msg: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly view of `choices[].message`: omit full `content`, show length or part count only."""

    out: dict[str, Any] = {}
    for k, v in msg.items():
        if k == "content":
            if isinstance(v, str):
                out["content"] = f"<string, len={len(v)}>"
            elif isinstance(v, list):
                out["content"] = f"<list of {len(v)} parts>"
            else:
                out["content"] = repr(v)[:400]
        elif isinstance(v, str) and len(v) > 800:
            out[k] = v[:800] + f"... (truncated, len={len(v)})"
        else:
            out[k] = v
    return out


def chat_completions_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    request_body: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_QWEN_TIMEOUT_S,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if request_body:
        for key, value in request_body.items():
            if key in {"model", "messages"}:
                raise ValueError(f"request_body must not override reserved field {key!r}.")
            payload[key] = value
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            raw: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat API HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat API connection error: {exc}") from exc

    return assistant_text_from_chat_response(raw), raw


def coerce_planner_request(obj: Any) -> PlannerClientRequest | None:
    if isinstance(obj, PlannerClientRequest):
        return obj
    if isinstance(obj, dict):
        try:
            return PlannerClientRequest.model_validate(obj)
        except Exception:
            return None
    return None


def coerce_executor_request(obj: Any) -> ExecutorClientRequest | None:
    if isinstance(obj, ExecutorClientRequest):
        return obj
    if isinstance(obj, dict):
        try:
            return ExecutorClientRequest.model_validate(obj)
        except Exception:
            return None
    return None


__all__ = [
    "DEFAULT_QWEN_BASE_URL",
    "DEFAULT_QWEN_MODEL",
    "assistant_text_from_chat_response",
    "build_artifact_path_index",
    "build_executor_control_user_text",
    "build_judge_control_user_text",
    "build_planner_control_user_text",
    "chat_completions_text",
    "coerce_executor_request",
    "coerce_planner_request",
    "env_executor_debug_enabled",
    "env_executor_use_tool_role",
    "env_planner_debug_enabled",
    "env_planner_use_tool_role",
    "env_qwen_config",
    "executor_to_openai_messages",
    "file_to_data_url",
    "is_placeholder_api_key",
    "judge_to_openai_messages",
    "planner_to_openai_messages",
    "sanitize_messages_for_debug",
    "summarize_openai_message_for_debug",
]
