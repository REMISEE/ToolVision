"""Build planner OpenAI messages for the easy-question path (reference answer appended)."""

from __future__ import annotations

from typing import Any

from offline_sft_pipeline.pipelines.api_text_multimodal import planner_to_openai_messages
from offline_sft_pipeline.pipelines.request_models import PlannerClientRequest


def build_easy_reference_answer_block(
    *,
    reference_answer: str,
    answer_instruction: str | None,
) -> str:
    """Append a lightweight no-tool final-answer constraint for easy samples."""
    ref = str(reference_answer).strip()
    inst = ""
    if answer_instruction and str(answer_instruction).strip():
        inst = (
            "\n- If an answer-format instruction was already stated above, still follow it; "
            "the final canonical answer must remain exactly the value below.\n"
        )
    return (
        "\n---\n"
        "Easy no-tool final-answer constraint:\n"
        "- Return `mode=\"answer\"`.\n"
        f"- The final JSON field `answer` must be exactly this string:\n  {ref!r}\n"
        f"{inst}"
        "- Keep the top-level `think` concise, image-grounded, and aligned with that exact answer.\n"
        "- In `think` / global reasoning: write as if you solved the task from the image and question alone. "
        "- Do not mention that the target answer was supplied in the prompt.\n"
        "- Do not use the prompt or any stated answer constraint as evidence or justification.\n"
        "- Do not mention these instructions in the output.\n"
    )


def _build_easy_capability_names_block(req: PlannerClientRequest) -> str:
    if not req.tool_capabilities:
        return "(none configured)"
    return "\n".join(f"- capability `{cap.name}`" for cap in req.tool_capabilities)


def _rewrite_capability_reference_for_easy(messages: list[dict[str, Any]], req: PlannerClientRequest) -> None:
    if not messages:
        raise ValueError("messages must be non-empty.")
    last = messages[-1]
    if last.get("role") != "user":
        raise ValueError("Last message must be role=user (planner control).")

    content = last.get("content")
    marker = "Capability reference (use only these names in `capability_plan`):\n"
    replacement = marker + _build_easy_capability_names_block(req)

    if isinstance(content, str):
        prefix, found, _ = content.partition(marker)
        if found:
            last["content"] = prefix + replacement
        return

    if isinstance(content, list):
        rewritten: list[dict[str, Any]] = []
        replaced = False
        for item in content:
            if (
                not replaced
                and isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
                and marker in item["text"]
            ):
                prefix, found, _ = item["text"].partition(marker)
                if found:
                    rewritten.append({"type": "text", "text": prefix + replacement})
                    replaced = True
                    continue
            rewritten.append(item)
        last["content"] = rewritten
        return

    raise TypeError(f"Unsupported user content type: {type(content)!r}")


def _append_text_to_last_user_message(messages: list[dict[str, Any]], text: str) -> None:
    if not messages:
        raise ValueError("messages must be non-empty.")
    last = messages[-1]
    if last.get("role") != "user":
        raise ValueError("Last message must be role=user (planner control).")
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = content.rstrip() + text
    elif isinstance(content, list):
        merged = list(content)
        merged.append({"type": "text", "text": text})
        last["content"] = merged
    else:
        raise TypeError(f"Unsupported user content type: {type(content)!r}")


def planner_to_openai_messages_easy(
    *,
    system_prompt: str,
    req: PlannerClientRequest,
    reference_answer: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Same as `planner_to_openai_messages`, plus easy-task reference block on the final user turn."""
    messages, missing = planner_to_openai_messages(system_prompt=system_prompt, req=req)
    _rewrite_capability_reference_for_easy(messages, req)
    block = build_easy_reference_answer_block(
        reference_answer=reference_answer,
        answer_instruction=req.answer_instruction,
    )
    _append_text_to_last_user_message(messages, block)
    return messages, missing


__all__ = [
    "build_easy_reference_answer_block",
    "planner_to_openai_messages_easy",
]
