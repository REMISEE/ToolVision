from __future__ import annotations

from typing import Any

import requests
from PIL import Image

from .codec import base64_to_image, image_to_base64


class DepthHTTPClient:
    def __init__(self, *, base_url: str, request_timeout: int = 180):
        self.base_url = str(base_url).rstrip("/")
        self.infer_url = f"{self.base_url}/infer"
        self.request_timeout = int(request_timeout)
        self.session = requests.Session()
        self.session.trust_env = False

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(self.infer_url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Depth HTTP {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        if data.get("errorCode", 0) != 0:
            raise RuntimeError(f"Depth service error: code={data.get('errorCode')} msg={data.get('errorMsg')}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"Depth response missing result object: {str(data)[:800]}")
        return result

    def _build_payload(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": image_to_base64(image.convert("RGB")),
            "file_type": int(kwargs.get("file_type", kwargs.get("fileType", 1))),
            "operation": str(kwargs.get("_operation", kwargs.get("operation", "estimate"))).strip().lower(),
            "vis_mode": str(kwargs.get("vis_mode", "overlay")).strip().lower(),
            "stat": str(kwargs.get("stat", "median")).strip().lower(),
            "padding": int(kwargs.get("padding", 0)),
        }
        passthrough_fields = [
            "text_prompt",
            "detection_index",
            "box_threshold",
            "text_threshold",
            "label",
            "x1",
            "y1",
            "x2",
            "y2",
            "region_box",
        ]
        for field in passthrough_fields:
            if field in kwargs:
                payload[field] = kwargs[field]
        return payload

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise RuntimeError("DepthHTTPClient expects PIL image input.")
        payload = self._build_payload(image, kwargs or {})
        result = self._request_json(payload)
        raw_images = result.get("images") or []
        images = [base64_to_image(item) for item in raw_images if isinstance(item, str) and item]
        if not images:
            images = [image.copy()]
        return {
            "images": images,
            "text": str(result.get("text", "")),
            "meta": result.get("meta", {}),
        }
