import os
import sys
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFilter


class GroundedSAM2Runner:
    def __init__(self, config: dict[str, Any]):
        self.config = config or {}
        self._initialized = False
        self.torch = None
        self.device = None
        self.predict_fn = None
        self.box_convert = None
        self.grounding_transform = None
        self.sam2_predictor = None
        self.grounding_model = None

    def _normalize_device(self, device_value: Any) -> str:
        raw = str(device_value or "").strip().lower()
        if not raw:
            return raw
        if raw.startswith("gpu"):
            return "cuda" + raw[3:]
        return raw

    def _codevision_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def _repo_root(self) -> str:
        return os.path.dirname(self._codevision_root())

    def _resolve_local_path(self, path_value: Any) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return raw
        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            return expanded
        candidates = [
            os.path.abspath(expanded),
            os.path.abspath(os.path.join(self._codevision_root(), expanded)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[-1]

    def _find_groundedsam2_root(self, paths: list[str]) -> Optional[str]:
        for path in paths:
            if not path:
                continue
            current = os.path.abspath(path)
            if os.path.isfile(current):
                current = os.path.dirname(current)
            while True:
                has_sam2 = os.path.isdir(os.path.join(current, "sam2"))
                has_gdino = os.path.isdir(os.path.join(current, "grounding_dino"))
                if has_sam2 and has_gdino:
                    return current
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
        return None

    def _ensure_path_prepend(self, path_value: str):
        path_abs = os.path.abspath(path_value)
        if os.path.isdir(path_abs) and path_abs not in sys.path:
            sys.path.insert(0, path_abs)

    def _resolve_grounding_dino_text_encoder(
        self,
        text_encoder_type: Any,
        model_config_path: str,
        grounded_sam2_root: Optional[str],
    ) -> str:
        raw = str(text_encoder_type or "").strip()
        if not raw:
            return raw

        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            return expanded

        is_path_like = any(sep in expanded for sep in (os.sep, "/", "\\"))
        if not is_path_like:
            return raw

        candidates = [os.path.abspath(expanded)]
        config_dir = os.path.dirname(os.path.abspath(model_config_path))
        base_dirs = [
            config_dir,
            grounded_sam2_root,
            os.path.join(grounded_sam2_root, "grounding_dino") if grounded_sam2_root else None,
            self._codevision_root(),
            self._repo_root(),
        ]
        for base_dir in base_dirs:
            if not base_dir:
                continue
            candidate = os.path.abspath(os.path.join(base_dir, expanded))
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return raw

    def _load_grounding_dino_model(
        self,
        *,
        build_model_fn,
        clean_state_dict_fn,
        slconfig_cls,
        torch_module,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str,
        grounded_sam2_root: Optional[str],
    ):
        args = slconfig_cls.fromfile(model_config_path)
        args.device = device
        args.text_encoder_type = self._resolve_grounding_dino_text_encoder(
            getattr(args, "text_encoder_type", ""),
            model_config_path=model_config_path,
            grounded_sam2_root=grounded_sam2_root,
        )

        model = build_model_fn(args)
        checkpoint = torch_module.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict_fn(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def _register_grounding_dino_alias(self):
        if "grounding_dino.groundingdino" in sys.modules:
            return
        try:
            import groundingdino as gd_pkg
        except Exception:
            return
        import types

        ns_pkg = sys.modules.get("grounding_dino")
        if ns_pkg is None:
            ns_pkg = types.ModuleType("grounding_dino")
            ns_pkg.__path__ = []
            sys.modules["grounding_dino"] = ns_pkg
        setattr(ns_pkg, "groundingdino", gd_pkg)
        sys.modules["grounding_dino.groundingdino"] = gd_pkg

    def initialize(self):
        if self._initialized:
            return

        required = {
            "sam2_checkpoint": self._resolve_local_path(self.config.get("sam2_checkpoint")),
            "sam2_model_config": self._resolve_local_path(self.config.get("sam2_model_config")),
            "grounding_dino_config": self._resolve_local_path(self.config.get("grounding_dino_config")),
            "grounding_dino_checkpoint": self._resolve_local_path(self.config.get("grounding_dino_checkpoint")),
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(f"Missing Grounded SAM2 config fields: {missing}")

        grounded_sam2_root = self._find_groundedsam2_root(list(required.values()))
        if grounded_sam2_root:
            self._ensure_path_prepend(grounded_sam2_root)
            self._ensure_path_prepend(os.path.join(grounded_sam2_root, "grounding_dino"))

        try:
            import torch
            import transformers
            from transformers import BertModel

            if not hasattr(BertModel, "get_head_mask"):
                raise RuntimeError(
                    "Incompatible transformers version "
                    f"{getattr(transformers, '__version__', 'unknown')}. "
                    "GroundingDINO in Grounded-SAM-2 expects BertModel.get_head_mask; "
                    "use transformers==4.33.2."
                )
            try:
                from grounding_dino.groundingdino.datasets import transforms as gdino_transforms
                from grounding_dino.groundingdino.models import build_model as build_grounding_dino_model
                from grounding_dino.groundingdino.util.inference import predict
                from grounding_dino.groundingdino.util.misc import clean_state_dict
                from grounding_dino.groundingdino.util.slconfig import SLConfig
            except Exception:
                self._register_grounding_dino_alias()
                try:
                    from grounding_dino.groundingdino.datasets import transforms as gdino_transforms
                    from grounding_dino.groundingdino.models import build_model as build_grounding_dino_model
                    from grounding_dino.groundingdino.util.inference import predict
                    from grounding_dino.groundingdino.util.misc import clean_state_dict
                    from grounding_dino.groundingdino.util.slconfig import SLConfig
                except Exception:
                    from groundingdino.datasets import transforms as gdino_transforms
                    from groundingdino.models import build_model as build_grounding_dino_model
                    from groundingdino.util.inference import predict
                    from groundingdino.util.misc import clean_state_dict
                    from groundingdino.util.slconfig import SLConfig
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from torchvision.ops import box_convert
        except Exception as exc:
            raise RuntimeError(f"Cannot import Grounded SAM2 dependencies: {exc!r}") from exc

        self.torch = torch
        self.device = self._normalize_device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.predict_fn = predict
        self.box_convert = box_convert
        self.grounding_transform = gdino_transforms.Compose(
            [
                gdino_transforms.RandomResize([800], max_size=1333),
                gdino_transforms.ToTensor(),
                gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        missing_files = [k for k, v in required.items() if not os.path.exists(v)]
        if missing_files:
            missing_detail = {k: required[k] for k in missing_files}
            raise RuntimeError(f"Grounded SAM2 files not found: {missing_detail}")

        sam2_config_name = required["sam2_model_config"].replace("\\", "/")
        marker = "/sam2/"
        if marker in sam2_config_name:
            sam2_config_name = sam2_config_name.split(marker, 1)[1]

        sam2_model = build_sam2(
            sam2_config_name,
            required["sam2_checkpoint"],
            device=self.device,
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.grounding_model = self._load_grounding_dino_model(
            build_model_fn=build_grounding_dino_model,
            clean_state_dict_fn=clean_state_dict,
            slconfig_cls=SLConfig,
            torch_module=torch,
            model_config_path=required["grounding_dino_config"],
            model_checkpoint_path=required["grounding_dino_checkpoint"],
            device=self.device,
            grounded_sam2_root=grounded_sam2_root,
        )
        self._initialized = True

    def _normalize_prompt(self, kwargs: dict[str, Any]) -> str:
        text_prompt = str(kwargs.get("text_prompt") or self.config.get("default_text_prompt") or "").strip()
        if not text_prompt:
            raise RuntimeError("text_prompt is required for grounded_sam2 operations.")
        if not text_prompt.endswith("."):
            text_prompt += "."
        return text_prompt.lower()

    def _empty_grounding(self, text_prompt: str) -> dict[str, Any]:
        return {"text_prompt": text_prompt, "input_boxes": [], "labels": [], "confidences": []}

    def _is_recoverable_grounding_empty(self, exc: Exception) -> bool:
        message = str(exc)
        return "selected index k out of range" in message

    def _run_grounding(self, image_rgb, kwargs: dict[str, Any]) -> dict[str, Any]:
        text_prompt = self._normalize_prompt(kwargs)
        if self.grounding_transform is None:
            raise RuntimeError("GroundingDINO transform is not initialized.")
        image_tensor, _ = self.grounding_transform(Image.fromarray(image_rgb).convert("RGB"), None)
        try:
            boxes, confidences, labels = self.predict_fn(
                model=self.grounding_model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=float(kwargs.get("box_threshold", self.config.get("box_threshold", 0.35))),
                text_threshold=float(kwargs.get("text_threshold", self.config.get("text_threshold", 0.25))),
                device=self.device,
            )
        except RuntimeError as exc:
            if self._is_recoverable_grounding_empty(exc):
                return self._empty_grounding(text_prompt)
            raise

        if boxes is None or len(boxes) == 0:
            return self._empty_grounding(text_prompt)

        torch = self.torch
        if not hasattr(boxes, "detach"):
            boxes = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        h, w, _ = image_rgb.shape
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
        boxes_xyxy = self.box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
        input_boxes = boxes_xyxy.detach().cpu().numpy()
        conf_list = confidences.detach().cpu().numpy().tolist() if hasattr(confidences, "detach") else list(confidences)
        labels_list = [str(x) for x in labels]
        return {
            "text_prompt": text_prompt,
            "input_boxes": input_boxes,
            "labels": labels_list,
            "confidences": conf_list,
        }

    def _predict_masks(self, image_rgb, input_boxes, kwargs: dict[str, Any]):
        import numpy as np

        if input_boxes is None or len(input_boxes) == 0:
            return np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool), []

        self.sam2_predictor.set_image(image_rgb)
        multimask_output = bool(kwargs.get("multimask_output", False))
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        if multimask_output and masks is not None and masks.ndim == 4:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
            if hasattr(scores, "ndim") and scores.ndim > 1:
                scores = scores[np.arange(scores.shape[0]), best]
        if masks is not None and masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool) if masks is not None else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
        score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return masks, score_list

    def _draw_boxes(self, image: Image.Image, boxes, labels, confidences):
        out = image.copy()
        draw = ImageDraw.Draw(out)
        ann = []
        for label, conf, box in zip(labels, confidences, boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
            draw.text((x1 + 2, y1 + 2), f"{label} {float(conf):.2f}", fill="lime")
            ann.append({"class_name": label, "bbox": [x1, y1, x2, y2], "confidence": float(conf)})
        return out, ann

    def _overlay_masks(self, image: Image.Image, masks, alpha: float = 0.45):
        import numpy as np

        base = np.array(image.convert("RGB")).astype(np.float32)
        if masks is None or len(masks) == 0:
            return Image.fromarray(base.astype(np.uint8))
        color = np.array([30.0, 144.0, 255.0], dtype=np.float32)
        out = base.copy()
        for mask in masks:
            m = mask.astype(bool)
            out[m] = out[m] * (1.0 - alpha) + color * alpha
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def _mask_to_bbox(self, mask):
        import numpy as np

        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def infer_box(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        image_rgb = np.array(image.convert("RGB"))
        grounded = self._run_grounding(image_rgb, kwargs)
        boxes = grounded["input_boxes"]
        if len(boxes) == 0:
            return {
                "images": [image.copy()],
                "text": "No objects found.",
                "meta": {"model": "grounded_sam2", "operation": "box", "annotations": []},
            }
        vis, ann = self._draw_boxes(image, boxes, grounded["labels"], grounded["confidences"])
        return {
            "images": [vis],
            "text": f"GroundedSAM2(box) detected {len(ann)} objects.",
            "meta": {
                "model": "grounded_sam2",
                "operation": "box",
                "text_prompt": grounded["text_prompt"],
                "annotations": ann,
            },
        }

    def infer_mask(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        image_rgb = np.array(image.convert("RGB"))
        grounded = self._run_grounding(image_rgb, kwargs)
        boxes = grounded["input_boxes"]
        if len(boxes) == 0:
            return {
                "images": [image.copy()],
                "text": "No objects found.",
                "meta": {"model": "grounded_sam2", "operation": "mask", "annotations": []},
            }
        masks, scores = self._predict_masks(image_rgb, boxes, kwargs)
        out = self._overlay_masks(image, masks, alpha=float(kwargs.get("mask_alpha", 0.45)))
        if bool(kwargs.get("draw_box_on_mask", True)):
            out, ann = self._draw_boxes(out, boxes, grounded["labels"], grounded["confidences"])
        else:
            ann = [
                {"class_name": label, "bbox": [float(v) for v in box], "confidence": float(conf)}
                for label, conf, box in zip(grounded["labels"], grounded["confidences"], boxes)
            ]
        return {
            "images": [out],
            "text": f"GroundedSAM2(mask) generated {len(masks)} masks.",
            "meta": {
                "model": "grounded_sam2",
                "operation": "mask",
                "text_prompt": grounded["text_prompt"],
                "mask_scores": scores,
                "annotations": ann,
            },
        }

    def infer_dino_crop(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        image_rgb = np.array(image.convert("RGB"))
        grounded = self._run_grounding(image_rgb, kwargs)
        boxes = grounded["input_boxes"]
        if len(boxes) == 0:
            return {
                "images": [image.copy()],
                "text": "No objects found.",
                "meta": {"model": "grounded_sam2", "operation": "dino_crop", "crop_boxes": []},
            }

        based_on = str(kwargs.get("based_on", "box")).lower()
        candidate_boxes = [[float(v) for v in box] for box in boxes]
        if based_on == "mask":
            masks, _ = self._predict_masks(image_rgb, boxes, kwargs)
            mask_boxes = [self._mask_to_bbox(mask) for mask in masks]
            candidate_boxes = [x for x in mask_boxes if x is not None] or candidate_boxes

        padding = int(kwargs.get("padding", 0))
        max_crops = max(1, int(kwargs.get("max_crops", 1)))
        detection_index = int(kwargs.get("detection_index", 0))
        ordered = candidate_boxes
        if detection_index >= 0:
            if detection_index >= len(candidate_boxes):
                detection_index = len(candidate_boxes) - 1
            ordered = [candidate_boxes[detection_index]]

        w, h = image.size
        crop_boxes = []
        crops = []
        for box in ordered[:max_crops]:
            x1, y1, x2, y2 = box
            x1 = max(0, int(round(x1 - padding)))
            y1 = max(0, int(round(y1 - padding)))
            x2 = min(w, int(round(x2 + padding)))
            y2 = min(h, int(round(y2 + padding)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_boxes.append([x1, y1, x2, y2])
            crops.append(image.crop((x1, y1, x2, y2)))

        if not crops:
            crops = [image.copy()]
        return {
            "images": crops,
            "text": f"GroundedSAM2(dino_crop) returned {len(crops)} crop images.",
            "meta": {
                "model": "grounded_sam2",
                "operation": "dino_crop",
                "text_prompt": grounded["text_prompt"],
                "based_on": based_on,
                "crop_boxes": crop_boxes,
            },
        }

    def infer_blur_bg(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        image_rgb = np.array(image.convert("RGB"))
        grounded = self._run_grounding(image_rgb, kwargs)
        boxes = grounded["input_boxes"]
        if len(boxes) == 0:
            return {
                "images": [image.copy()],
                "text": "No objects found.",
                "meta": {"model": "grounded_sam2", "operation": "blur_bg", "annotations": []},
            }

        masks, _ = self._predict_masks(image_rgb, boxes, kwargs)
        if len(masks) == 0:
            return self.infer_box(image, kwargs)

        blur_radius = float(kwargs.get("blur_radius", 8.0))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        fg = np.array(image.convert("RGB"))
        bg = np.array(blurred.convert("RGB"))
        union_mask = np.any(masks.astype(bool), axis=0)
        mixed = bg.copy()
        mixed[union_mask] = fg[union_mask]
        out = Image.fromarray(mixed.astype(np.uint8))
        out, ann = self._draw_boxes(out, boxes, grounded["labels"], grounded["confidences"])
        return {
            "images": [out],
            "text": "GroundedSAM2(blur_bg) generated background blur image.",
            "meta": {
                "model": "grounded_sam2",
                "operation": "blur_bg",
                "text_prompt": grounded["text_prompt"],
                "blur_radius": blur_radius,
                "annotations": ann,
            },
        }

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        if not isinstance(image, Image.Image):
            raise RuntimeError("GroundedSAM2Runner expects PIL image input.")

        operation = str(kwargs.get("_operation", kwargs.get("operation", "box"))).lower()
        if operation == "box":
            return self.infer_box(image, kwargs)
        if operation == "mask":
            return self.infer_mask(image, kwargs)
        if operation == "dino_crop":
            return self.infer_dino_crop(image, kwargs)
        if operation == "blur_bg":
            return self.infer_blur_bg(image, kwargs)
        raise RuntimeError(f"Unsupported grounded_sam2 operation: {operation}")
