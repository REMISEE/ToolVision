"""Reward helpers for CodeVision RL data.

The public verl entry point remains ``recipe/codevision/reward.py``.  This
package keeps ToolVision/CodeVision RL dataset reward routing separate from
the legacy benchmark/eval reward code.
"""

from .router import compute_toolvision_score

__all__ = ["compute_toolvision_score"]
