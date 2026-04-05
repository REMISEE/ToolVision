import base64
import io
from typing import Any, Optional

import requests
from PIL import Image

from .parser import parse_ocr_result


class PaddleOCRHTTPClient:
    def __init__(
        self,
        *,
        base_url: str,
        request_timeout: int = 180,
        default_file_type: int = 1,
        default_visualize: Optional[bool] = None,
        line_y_threshold: float = 0.6,
        default_request_options: Optional[dict[str, Any]] = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.ocr_url = f"{self.base_url}/ocr"
        self.request_timeout = int(request_timeout)
        self.default_file_type = int(default_file_type)
        self.default_visualize = default_visualize
        self.line_y_threshold = float(line_y_threshold)
        self.default_request_options = dict(default_request_options or {})

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _decode_base64_image(self, data: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(self.ocr_url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OCR HTTP {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        if data.get("errorCode", 0) != 0:
            raise RuntimeError(f"OCR service error: code={data.get('errorCode')} msg={data.get('errorMsg')}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"OCR response missing result object: {str(data)[:800]}")
        return result

    def _build_payload(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self._image_to_base64(image),
            "fileType": int(kwargs.get("fileType", kwargs.get("file_type", self.default_file_type))),
        }
        field_map = {
            "use_doc_orientation_classify": "useDocOrientationClassify",
            "use_doc_unwarping": "useDocUnwarping",
            "use_textline_orientation": "useTextlineOrientation",
            "text_det_limit_side_len": "textDetLimitSideLen",
            "text_det_limit_type": "textDetLimitType",
            "text_det_thresh": "textDetThresh",
            "text_det_box_thresh": "textDetBoxThresh",
            "text_det_unclip_ratio": "textDetUnclipRatio",
            "text_rec_score_thresh": "textRecScoreThresh",
            "return_word_box": "returnWordBox",
            "visualize": "visualize",
        }
        for src, dst in field_map.items():
            if src in kwargs:
                payload[dst] = kwargs[src]
            elif dst in kwargs:
                payload[dst] = kwargs[dst]
            elif src in self.default_request_options:
                payload[dst] = self.default_request_options[src]
            elif dst in self.default_request_options:
                payload[dst] = self.default_request_options[dst]
        if "visualize" not in payload and self.default_visualize is not None:
            payload["visualize"] = self.default_visualize
        return payload

    def _extract_output_images(self, raw_pages: list[dict[str, Any]], fallback: Image.Image) -> list[Image.Image]:
        out_images: list[Image.Image] = []
        for raw_page in raw_pages:
            if not isinstance(raw_page, dict):
                continue
            for image_key in ("ocrImage", "docPreprocessingImage", "inputImage"):
                image_b64 = raw_page.get(image_key)
                if not isinstance(image_b64, str) or not image_b64:
                    continue
                try:
                    out_images.append(self._decode_base64_image(image_b64))
                    break
                except Exception:
                    continue
        return out_images or [fallback.copy()]

    def infer(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise RuntimeError("PaddleOCRHTTPClient expects PIL image input.")
        payload = self._build_payload(image.convert("RGB"), kwargs or {})
        ocr_result = self._request_json(payload)
        parsed = parse_ocr_result(ocr_result, self.line_y_threshold)
        return {
            "images": self._extract_output_images(parsed["raw_pages"], image),
            "text": parsed["text"],
            "meta": {
                "model": "paddleocr_http",
                "ocr_result": ocr_result,
                "ocr_pages": parsed["ocr_pages"],
                "num_ocr_items": parsed["num_ocr_items"],
            },
        }
