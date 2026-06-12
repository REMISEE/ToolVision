#!/usr/bin/env python3
"""Standalone Depth Pro service launcher.

This avoids importing ``verl.__init__`` so the HTTP service can run inside the
dedicated depth environment with only service dependencies installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"
EXTERNAL_ROOT = VERL_ROOT / "external_services"
DEPTH_ROOT = EXTERNAL_ROOT / "depth"


def _ensure_namespace(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return

    paths = list(getattr(module, "__path__", []))
    path_str = str(path)
    if path_str not in paths:
        paths.append(path_str)
        module.__path__ = paths


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot build import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_service_app():
    _ensure_namespace("verl", VERL_ROOT)
    _ensure_namespace("verl.external_services", EXTERNAL_ROOT)
    _ensure_namespace("verl.external_services.depth", DEPTH_ROOT)

    _load_module("verl.external_services.depth.codec", DEPTH_ROOT / "codec.py")
    _load_module("verl.external_services.depth.runner", DEPTH_ROOT / "runner.py")
    return _load_module("verl.external_services.depth.service_app", DEPTH_ROOT / "service_app.py")


def main() -> None:
    try:
        service_app = _load_service_app()
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            "Missing dependency while launching Depth service: "
            f"{missing_name}. Install the Depth Pro service environment first."
        ) from exc
    service_app.main()


if __name__ == "__main__":
    main()
