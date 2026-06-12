from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class CountGDRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config or {})
        self._initialized = False
        self.device = None
        self.torch = None
        self.ndimage = None
        self.colormap = None
        self.model = None
        self.transform = None

    def _codevision_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def _toolvision_root(self) -> str:
        return os.path.dirname(self._codevision_root())

    def _default_countgd_root(self) -> str:
        return os.path.join(self._toolvision_root(), "CountGD")

    def _resolve_countgd_root(self) -> str:
        raw = str(self.config.get("countgd_root") or self._default_countgd_root()).strip()
        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            return expanded
        return os.path.abspath(os.path.join(self._toolvision_root(), expanded))

    def _resolve_path(self, path_value: Any, *, root: str) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return raw
        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            return expanded
        return os.path.abspath(os.path.join(root, expanded))

    def _normalize_device(self, device_value: Any) -> str:
        raw = str(device_value or "").strip().lower()
        if not raw:
            return "cuda"
        if raw.startswith("gpu"):
            return "cuda" + raw[3:]
        return raw

    def _ensure_countgd_on_path(self, countgd_root: str) -> None:
        if countgd_root not in sys.path:
            sys.path.insert(0, countgd_root)

    def _build_model_args(self, *, config_path: str, checkpoint_path: str, text_encoder_path: str, device: str):
        from util.slconfig import SLConfig

        cfg = SLConfig.fromfile(config_path)
        cfg.merge_from_dict({"text_encoder_type": text_encoder_path})
        cfg_dict = cfg._cfg_dict.to_dict()

        args = argparse.Namespace(
            options=None,
            remove_difficult=False,
            fix_size=False,
            note="",
            device=device,
            resume="",
            pretrain_model_path=checkpoint_path,
            config=config_path,
            image_path="",
            output_image_name="",
            text="",
            confidence_thresh=float(self.config.get("default_confidence_thresh", 0.23)),
            finetune_ignore=None,
            start_epoch=0,
            eval=True,
            num_workers=8,
            test=False,
            debug=False,
            find_unused_params=False,
            save_results=False,
            save_log=False,
            world_size=1,
            dist_url="env://",
            rank=0,
            local_rank=None,
            amp=False,
        )
        for key, value in cfg_dict.items():
            setattr(args, key, value)
        args.device = device
        args.pretrain_model_path = checkpoint_path
        args.config = config_path
        return args

    def _build_model_and_transform(self, args):
        import datasets_inference.transforms as T
        from models.registry import MODULE_BUILD_FUNCS

        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        data_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

        assert args.modelname in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build_func(args)

        checkpoint = self.torch.load(args.pretrain_model_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model, data_transform

    def initialize(self) -> None:
        if self._initialized:
            return

        countgd_root = self._resolve_countgd_root()
        if not os.path.isdir(countgd_root):
            raise RuntimeError(f"CountGD root not found: {countgd_root}")
        self._ensure_countgd_on_path(countgd_root)

        try:
            import torch
            import scipy.ndimage as ndimage
            from matplotlib import cm
        except Exception as exc:
            raise RuntimeError(f"Cannot import CountGD dependencies: {exc!r}") from exc

        device_text = self._normalize_device(self.config.get("device", "cuda"))
        if device_text.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CountGD service requested CUDA but no CUDA GPU is available.")

        self.torch = torch
        self.ndimage = ndimage
        self.colormap = cm.get_cmap("jet")
        self.device = torch.device(device_text)

        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        config_path = self._resolve_path(
            self.config.get("config_path") or os.path.join("config", "cfg_fsc147_vit_b.py"),
            root=countgd_root,
        )
        checkpoint_path = self._resolve_path(
            self.config.get("pretrain_model_path") or os.path.join("checkpoints", "checkpoint_fsc147_best.pth"),
            root=countgd_root,
        )
        text_encoder_path = self._resolve_path(
            self.config.get("text_encoder_type") or os.path.join("checkpoints", "bert-base-uncased"),
            root=countgd_root,
        )

        missing = [
            path for path in (config_path, checkpoint_path, text_encoder_path)
            if not os.path.exists(path)
        ]
        if missing:
            raise RuntimeError(f"CountGD required files not found: {missing}")

        model_args = self._build_model_args(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            text_encoder_path=text_encoder_path,
            device=str(self.device),
        )
        self.model, self.transform = self._build_model_and_transform(model_args)
        self._initialized = True

    def _normalize_prompt(self, kwargs: dict[str, Any]) -> str:
        text_prompt = str(kwargs.get("text_prompt") or "").strip()
        if not text_prompt:
            raise RuntimeError("text_prompt is required for CountGD inference.")
        return text_prompt

    def _predict(self, image: Image.Image, *, text_prompt: str, confidence_thresh: float):
        source = image.convert("RGB")
        transformed, target = self.transform(source, {"exemplars": self.torch.tensor([])})
        input_image = transformed.to(self.device)
        input_exemplar = target["exemplars"].to(self.device)

        with self.torch.no_grad():
            model_output = self.model(
                input_image.unsqueeze(0),
                [input_exemplar],
                [self.torch.tensor([0], device=self.device)],
                captions=[text_prompt + " ."],
            )
        logits = model_output["pred_logits"][0].sigmoid()
        boxes = model_output["pred_boxes"][0]

        scores = logits.max(dim=-1).values
        keep_mask = scores > confidence_thresh
        return boxes[keep_mask], scores[keep_mask]

    def _boxes_to_xyxy(self, boxes, width: int, height: int) -> list[list[int]]:
        if boxes.numel() == 0:
            return []
        scale = self.torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
        xyxy = boxes.clone() * scale
        xyxy[:, 0] -= xyxy[:, 2] / 2.0
        xyxy[:, 1] -= xyxy[:, 3] / 2.0
        xyxy[:, 2] += xyxy[:, 0]
        xyxy[:, 3] += xyxy[:, 1]
        xyxy = xyxy.round().to(dtype=self.torch.int64).cpu().numpy()

        out: list[list[int]] = []
        for x1, y1, x2, y2 in xyxy.tolist():
            out.append(
                [
                    max(0, min(int(width), int(x1))),
                    max(0, min(int(height), int(y1))),
                    max(0, min(int(width), int(x2))),
                    max(0, min(int(height), int(y2))),
                ]
            )
        return out

    def _render_box_overlay(self, image: Image.Image, boxes_xyxy: list[list[int]]) -> Image.Image:
        out = image.convert("RGB").copy()
        draw = ImageDraw.Draw(out)
        for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy, start=1):
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), str(idx), fill="yellow")
        return out

    def _render_heatmap_overlay(self, image: Image.Image, boxes, *, sigma: float) -> Image.Image:
        width, height = image.size
        det_map = np.zeros((height, width), dtype=np.float32)
        if boxes.numel() > 0:
            xs = np.clip((width * boxes[:, 0]).detach().cpu().numpy().astype(int), 0, width - 1)
            ys = np.clip((height * boxes[:, 1]).detach().cpu().numpy().astype(int), 0, height - 1)
            det_map[ys, xs] = 1.0

        det_map = self.ndimage.gaussian_filter(det_map, sigma=(sigma, sigma), order=0)
        if float(det_map.max()) > 0.0:
            det_map = det_map / float(det_map.max())

        heat_rgba = self.colormap(det_map)
        heat_rgb = (heat_rgba[..., :3] * 255.0).astype(np.uint8)
        alpha = (det_map * 180.0).clip(0, 180).astype(np.uint8)

        base = image.convert("RGBA")
        overlay = Image.fromarray(heat_rgb, mode="RGB").convert("RGBA")
        overlay.putalpha(Image.fromarray(alpha, mode="L"))
        return Image.alpha_composite(base, overlay).convert("RGB")

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        if not isinstance(image, Image.Image):
            raise RuntimeError("CountGD runner expects PIL image input.")

        text_prompt = self._normalize_prompt(kwargs or {})
        confidence_thresh = float(kwargs.get("confidence_thresh", self.config.get("default_confidence_thresh", 0.23)))
        visualize = str(kwargs.get("visualize", self.config.get("default_visualize", "heatmap"))).strip().lower()
        heatmap_sigma = float(kwargs.get("heatmap_sigma", self.config.get("heatmap_sigma", 5.0)))

        boxes, scores = self._predict(
            image,
            text_prompt=text_prompt,
            confidence_thresh=confidence_thresh,
        )
        width, height = image.size
        boxes_xyxy = self._boxes_to_xyxy(boxes, width, height)

        if visualize == "boxes":
            vis_image = self._render_box_overlay(image, boxes_xyxy)
        else:
            vis_image = self._render_heatmap_overlay(image, boxes, sigma=heatmap_sigma)
            visualize = "heatmap"

        count = len(boxes_xyxy)
        return {
            "images": [vis_image],
            "text": f"CountGD predicted count={count} for prompt='{text_prompt}'.",
            "meta": {
                "model": "countgd",
                "count": count,
                "text_prompt": text_prompt,
                "confidence_thresh": confidence_thresh,
                "visualize": visualize,
                "boxes_xyxy": boxes_xyxy,
                "scores": [float(score) for score in scores.detach().cpu().tolist()],
            },
        }
