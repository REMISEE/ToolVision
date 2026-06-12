from typing import Any

import requests
from PIL import Image

from .codec import base64_to_image, image_to_base64


class CountGDHTTPClient:
    def __init__(self, *, base_url: str, request_timeout: int = 180):
        self.base_url = str(base_url).rstrip("/")
        self.infer_url = f"{self.base_url}/infer"
        self.request_timeout = int(request_timeout)
        self.session = requests.Session()
        self.session.trust_env = False

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(self.infer_url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"CountGD HTTP {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        if data.get("errorCode", 0) != 0:
            raise RuntimeError(f"CountGD service error: code={data.get('errorCode')} msg={data.get('errorMsg')}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"CountGD response missing result object: {str(data)[:800]}")
        return result

    def _build_payload(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": image_to_base64(image.convert("RGB")),
            "file_type": int(kwargs.get("file_type", kwargs.get("fileType", 1))),
            "text_prompt": str(kwargs.get("text_prompt") or "").strip(),
            "confidence_thresh": float(kwargs.get("confidence_thresh", 0.23)),
            "visualize": str(kwargs.get("visualize", "heatmap")).strip().lower(),
        }
        if "heatmap_sigma" in kwargs:
            payload["heatmap_sigma"] = float(kwargs["heatmap_sigma"])
        return payload

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise RuntimeError("CountGDHTTPClient expects PIL image input.")
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
