from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class DepthProRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self._depth_initialized = False
        self.torch = None
        self.device = None
        self.precision = None
        self.model = None
        self.transform = None
        self.cm = None
        self.depth_pro_module = None
        self.grounding_client = None

        self._depth_cache: dict[str, np.ndarray] = {}
        self._depth_cache_order: list[str] = []

    def _codevision_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def _toolvision_root(self) -> str:
        return os.path.dirname(self._codevision_root())

    def _default_depth_root(self) -> str:
        return os.path.join(self._toolvision_root(), "ml-depth-pro-main")

    def _resolve_path(self, path_value: Any, *, root: Optional[str] = None) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return raw
        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            return expanded
        candidates = [os.path.abspath(expanded)]
        if root:
            candidates.append(os.path.abspath(os.path.join(root, expanded)))
        candidates.append(os.path.abspath(os.path.join(self._toolvision_root(), expanded)))
        candidates.append(os.path.abspath(os.path.join(self._codevision_root(), expanded)))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[-1]

    def _normalize_device(self, device_value: Any, *, fallback: str) -> str:
        raw = str(device_value or "").strip().lower()
        if not raw:
            return fallback
        if raw.startswith("gpu"):
            return "cuda" + raw[3:]
        return raw

    def _default_depth_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_path_prepend(self, path_value: str) -> None:
        path_abs = os.path.abspath(path_value)
        if os.path.isdir(path_abs) and path_abs not in sys.path:
            sys.path.insert(0, path_abs)

    def _image_cache_key(self, image: Image.Image) -> str:
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _remember_depth_cache(self, key: str, depth: np.ndarray) -> None:
        max_items = max(0, int(self.config.get("cache_size", 8)))
        if max_items <= 0:
            return
        self._depth_cache[key] = depth
        if key in self._depth_cache_order:
            self._depth_cache_order.remove(key)
        self._depth_cache_order.append(key)
        while len(self._depth_cache_order) > max_items:
            oldest = self._depth_cache_order.pop(0)
            self._depth_cache.pop(oldest, None)

    def _init_depth_model(self) -> None:
        if self._depth_initialized:
            return

        depth_root = self._resolve_path(self.config.get("depth_pro_root") or self._default_depth_root())
        if not os.path.isdir(depth_root):
            raise RuntimeError(f"Depth Pro root not found: {depth_root}")
        self._ensure_path_prepend(os.path.join(depth_root, "src"))

        try:
            import torch
            from matplotlib import cm
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        except Exception as exc:
            raise RuntimeError(f"Cannot import Depth Pro dependencies: {exc!r}") from exc

        checkpoint_path = self._resolve_path(
            self.config.get("checkpoint_path") or os.path.join("checkpoints", "depth_pro.pt"),
            root=depth_root,
        )
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Depth Pro checkpoint not found: {checkpoint_path}")

        depth_device_text = self._normalize_device(
            self.config.get("device"),
            fallback=self._default_depth_device(),
        )
        if depth_device_text.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Depth service requested CUDA but no CUDA GPU is available.")

        self.torch = torch
        self.device = torch.device(depth_device_text)
        self.precision = torch.half if depth_device_text.startswith("cuda") else torch.float32
        depth_config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=checkpoint_path)
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=depth_config,
            device=self.device,
            precision=self.precision,
        )
        self.model.eval()
        self.cm = cm.get_cmap("turbo")
        self.depth_pro_module = depth_pro
        self._depth_initialized = True

    def _init_grounding_client(self) -> None:
        if self.grounding_client is not None:
            return
        try:
            from ..groundedsam2.client import GroundedSAM2HTTPClient
        except Exception as exc:
            raise RuntimeError(f"Cannot import GroundedSAM2 HTTP client for ground_depth: {exc!r}") from exc

        self.grounding_client = GroundedSAM2HTTPClient(
            base_url=str(self.config.get("groundedsam2_base_url") or "http://127.0.0.1:8081"),
            request_timeout=int(self.config.get("request_timeout", 180)),
        )

    def _normalize_prompt(self, kwargs: dict[str, Any]) -> str:
        text_prompt = str(kwargs.get("text_prompt") or self.config.get("default_text_prompt") or "").strip()
        if not text_prompt:
            raise RuntimeError("text_prompt is required for ground_depth.")
        if not text_prompt.endswith("."):
            text_prompt += "."
        return text_prompt.lower()

    def _run_grounding(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._init_grounding_client()
        text_prompt = self._normalize_prompt(kwargs)
        result = self.grounding_client.infer(
            image.convert("RGB"),
            {
                "_operation": "box",
                "text_prompt": text_prompt,
                "box_threshold": float(kwargs.get("box_threshold", self.config.get("box_threshold", 0.35))),
                "text_threshold": float(kwargs.get("text_threshold", self.config.get("text_threshold", 0.25))),
            },
        )
        meta = result.get("meta") or {}
        annotations = meta.get("annotations") or []
        return {
            "text_prompt": text_prompt,
            "annotations": annotations,
        }

    def _estimate_depth(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        self._init_depth_model()
        rgb = image.convert("RGB")
        cache_key = self._image_cache_key(rgb)
        cached = self._depth_cache.get(cache_key)
        if cached is not None:
            return cached, True

        prediction = self.model.infer(self.transform(rgb), f_px=None)
        depth = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)
        self._remember_depth_cache(cache_key, depth)
        return depth, False

    def _normalize_inverse_depth(self, depth: np.ndarray) -> np.ndarray:
        safe = np.clip(depth.astype(np.float32), 1e-4, 1e4)
        inv = 1.0 / safe
        max_inv = min(float(np.nanmax(inv)), 1.0 / 0.1)
        min_inv = max(1.0 / 250.0, float(np.nanmin(inv)))
        if not np.isfinite(max_inv) or not np.isfinite(min_inv) or max_inv <= min_inv:
            return np.zeros_like(inv, dtype=np.float32)
        norm = (inv - min_inv) / (max_inv - min_inv)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    def _colorize_depth(self, depth: np.ndarray) -> Image.Image:
        norm = self._normalize_inverse_depth(depth)
        rgb = (self.cm(norm)[..., :3] * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _build_visualization(self, image: Image.Image, depth: np.ndarray, *, vis_mode: str) -> Image.Image:
        heatmap = self._colorize_depth(depth)
        if vis_mode == "heatmap":
            return heatmap
        return Image.blend(image.convert("RGB"), heatmap.resize(image.size), 0.45)

    def _resolve_region_box(self, image: Image.Image, kwargs: dict[str, Any]) -> list[int]:
        width, height = image.size
        region_box = kwargs.get("region_box")
        if isinstance(region_box, (list, tuple)) and len(region_box) == 4:
            values = [float(v) for v in region_box]
        elif all(key in kwargs for key in ("x1", "y1", "x2", "y2")):
            values = [float(kwargs["x1"]), float(kwargs["y1"]), float(kwargs["x2"]), float(kwargs["y2"])]
        else:
            raise RuntimeError("manual_depth requires region_box or x1,y1,x2,y2.")

        padding = max(0, int(kwargs.get("padding", 0)))
        x1 = max(0, int(round(min(values[0], values[2]) - padding)))
        y1 = max(0, int(round(min(values[1], values[3]) - padding)))
        x2 = min(width, int(round(max(values[0], values[2]) + padding)))
        y2 = min(height, int(round(max(values[1], values[3]) + padding)))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"Invalid region box after clipping: {[x1, y1, x2, y2]}")
        return [x1, y1, x2, y2]

    def _region_stats(self, depth: np.ndarray, box: list[int]) -> dict[str, Any]:
        x1, y1, x2, y2 = [int(v) for v in box]
        roi = depth[y1:y2, x1:x2]
        finite = roi[np.isfinite(roi)]
        valid_ratio = 0.0 if roi.size == 0 else float(finite.size / roi.size)
        if finite.size == 0:
            return {
                "region_box": [x1, y1, x2, y2],
                "valid_ratio": valid_ratio,
                "mean_depth_m": None,
                "median_depth_m": None,
                "min_depth_m": None,
                "max_depth_m": None,
                "p25_depth_m": None,
                "p75_depth_m": None,
            }
        return {
            "region_box": [x1, y1, x2, y2],
            "valid_ratio": valid_ratio,
            "mean_depth_m": float(np.mean(finite)),
            "median_depth_m": float(np.median(finite)),
            "min_depth_m": float(np.min(finite)),
            "max_depth_m": float(np.max(finite)),
            "p25_depth_m": float(np.percentile(finite, 25)),
            "p75_depth_m": float(np.percentile(finite, 75)),
        }

    def _annotate_region(
        self,
        image: Image.Image,
        *,
        box: list[int],
        label: str,
        color: str = "lime",
    ) -> Image.Image:
        out = image.copy()
        draw = ImageDraw.Draw(out)
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=color)
        return out

    def _pick_stat_value(self, stats: dict[str, Any], stat: str) -> tuple[str, Optional[float]]:
        normalized = str(stat or "median").strip().lower()
        if normalized == "mean":
            return "mean_depth_m", stats.get("mean_depth_m")
        return "median_depth_m", stats.get("median_depth_m")

    def infer_estimate(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        depth, cache_hit = self._estimate_depth(image)
        vis_mode = str(kwargs.get("vis_mode", "overlay")).strip().lower()
        out = self._build_visualization(image, depth, vis_mode=vis_mode)
        meta = {
            "model": "depth_pro",
            "operation": "estimate",
            "depth_unit": "meter",
            "vis_mode": vis_mode,
            "cache_hit": cache_hit,
            "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
            "min_depth_m": float(np.nanmin(depth)),
            "max_depth_m": float(np.nanmax(depth)),
            "mean_depth_m": float(np.nanmean(depth)),
            "median_depth_m": float(np.nanmedian(depth)),
        }
        text = f"Estimated metric depth map with range {meta['min_depth_m']:.2f}m to {meta['max_depth_m']:.2f}m."
        return {
            "images": [out],
            "text": text,
            "meta": meta,
        }

    def infer_manual_depth(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        depth, cache_hit = self._estimate_depth(image)
        vis_mode = str(kwargs.get("vis_mode", "overlay")).strip().lower()
        region_box = self._resolve_region_box(image, kwargs)
        stats = self._region_stats(depth, region_box)
        stat_key, stat_value = self._pick_stat_value(stats, str(kwargs.get("stat", "median")))
        label_text = str(kwargs.get("label") or "region").strip()
        metric_name = "mean" if stat_key.startswith("mean") else "median"
        display_value = "n/a" if stat_value is None else f"{stat_value:.2f}m"
        vis = self._build_visualization(image, depth, vis_mode=vis_mode)
        vis = self._annotate_region(vis, box=region_box, label=f"{label_text}: {display_value}")
        text = f"{label_text} {metric_name} depth is {display_value}."
        return {
            "images": [vis],
            "text": text,
            "meta": {
                "model": "depth_pro",
                "operation": "manual_depth",
                "depth_unit": "meter",
                "vis_mode": vis_mode,
                "cache_hit": cache_hit,
                "stat": metric_name,
                **stats,
            },
        }

    def infer_ground_depth(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        grounded = self._run_grounding(image, kwargs)
        annotations = grounded.get("annotations") or []
        if not annotations:
            depth, cache_hit = self._estimate_depth(image)
            vis_mode = str(kwargs.get("vis_mode", "overlay")).strip().lower()
            out = self._build_visualization(image, depth, vis_mode=vis_mode)
            text_prompt = grounded.get("text_prompt") or str(kwargs.get("text_prompt") or "").strip()
            return {
                "images": [out],
                "text": f"No objects found for {text_prompt!r}.",
                "meta": {
                    "model": "depth_pro",
                    "operation": "ground_depth",
                    "depth_unit": "meter",
                    "vis_mode": vis_mode,
                    "cache_hit": cache_hit,
                    "text_prompt": text_prompt,
                    "annotations": [],
                },
            }

        detection_index = int(kwargs.get("detection_index", 0))
        detection_index = max(0, min(detection_index, len(annotations) - 1))
        selected = annotations[detection_index]
        region_kwargs = dict(kwargs)
        region_kwargs["region_box"] = selected["bbox"]
        region_kwargs["label"] = str(kwargs.get("label") or selected.get("class_name") or f"det_{detection_index}")

        result = self.infer_manual_depth(image, region_kwargs)
        result_meta = dict(result.get("meta") or {})
        result_meta.update(
            {
                "operation": "ground_depth",
                "text_prompt": grounded["text_prompt"],
                "detection_index": detection_index,
                "grounding_annotation": selected,
                "annotations": annotations,
            }
        )
        stat_name = result_meta.get("stat", "median")
        stat_value = result_meta.get(f"{stat_name}_depth_m")
        stat_text = "n/a" if stat_value is None else f"{float(stat_value):.2f}m"
        result["text"] = (
            f"Detected {len(annotations)} objects for {grounded['text_prompt']!r}; "
            f"using detection {detection_index}. {region_kwargs['label']} {stat_name} depth is {stat_text}."
        )
        result["meta"] = result_meta
        return result

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise RuntimeError("DepthProRunner expects PIL image input.")

        operation = str(kwargs.get("operation", kwargs.get("_operation", "estimate"))).strip().lower()
        if operation == "estimate":
            return self.infer_estimate(image, kwargs)
        if operation == "manual_depth":
            return self.infer_manual_depth(image, kwargs)
        if operation == "ground_depth":
            return self.infer_ground_depth(image, kwargs)
        raise RuntimeError(f"Unsupported depth operation: {operation}")
