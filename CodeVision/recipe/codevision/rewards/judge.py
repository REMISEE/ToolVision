import os
from typing import Any


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        from verl.utils.reward_score.llmjudge import LLMJudgeClient
    except Exception:
        return None

    _CLIENT = LLMJudgeClient(
        base_url=os.getenv("LLM_JUDGE_BASE_URL", ""),
        model_name=os.getenv("LLM_JUDGE_MODEL_NAME", None),
        api_key=os.getenv("LLM_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        timeout=int(os.getenv("LLM_JUDGE_TIMEOUT", "100")),
        max_retries=int(os.getenv("LLM_JUDGE_MAX_RETRIES", "3")),
        temperature=0.0,
    )
    return _CLIENT


def judge_score(answer: str, ground_truth: str, *, question: str = "", task: str = "general") -> tuple[float, int]:
    client = _client()
    if client is None or client.client is None:
        return 0.0, 0
    score, _reason = client.verify(
        answer=answer,
        ground_truth=ground_truth,
        question=question,
        task=task,
    )
    return float(score), 1


def should_use_judge(extra_info: dict[str, Any] | None) -> bool:
    extra_info = extra_info or {}
    value = extra_info.get("use_llm_judge", os.getenv("TOOLVISION_RL_USE_LLM_JUDGE", "0"))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
