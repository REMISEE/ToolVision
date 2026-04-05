"""Load `ToolCapability` list from JSON (aligned with docs/21)."""

from __future__ import annotations

import json
from pathlib import Path

from .request_models import ToolCapability

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "example" / "tool_capabilities.json"


def load_tool_capabilities_from_file(path: str | Path | None = None) -> list[ToolCapability]:
    """Load capabilities from JSON array of {name, description, usage_notes?}."""
    p = Path(path) if path is not None else _DEFAULT_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"tool capabilities file must be a JSON array: {p}")
    return [ToolCapability.model_validate(item) for item in raw]


__all__ = ["load_tool_capabilities_from_file"]
