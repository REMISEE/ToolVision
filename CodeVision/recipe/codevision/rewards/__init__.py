"""Reward helpers for CodeVision RL data.

The public verl entry point remains ``recipe/codevision/reward.py``.  This
package keeps ToolVision/CodeVision RL dataset reward routing separate from
the legacy benchmark/eval reward code.
"""

from .router import compute_toolvision_score
from .step_answerability import (
    STEP_REWARD_VERSION,
    StepAnswerabilityConfig,
    StepAnswerabilityJudgeClient,
    compute_step_answerability_delta,
)

__all__ = [
    "STEP_REWARD_VERSION",
    "StepAnswerabilityConfig",
    "StepAnswerabilityJudgeClient",
    "compute_step_answerability_delta",
    "compute_toolvision_score",
]
