# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import base64
import io
import json
import logging
import os
import threading
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests
from PIL import Image, ImageDraw, ImageFilter
from qwen_vl_utils import fetch_image

from ..external_services.groundedsam2.client import GroundedSAM2HTTPClient
from ..external_services.paddleocr.client import PaddleOCRHTTPClient
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

SUCCESS_FOLLOWUP_TEXT = (
    "Here is the processed image. Now, analyze the returned results. "
    "Please keep thinking step-by-step inside the <think></think> tags to determine the next action. "
    "If additional tools are required, call them inside the <tool_calls></tool_calls> tags. "
    "Otherwise, provide your final answer within the <answer></answer> tags."
)


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class CodeExecutionWorker:
    """Worker for executing code-based image processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="code-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing code-based image processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


class BaseExternalModelAdapter:
    """Common adapter interface for external model backends."""

    # 初始化外部模型适配器基础配置。
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._initialized = False

    # 预留初始化钩子，子类可按需加载模型/客户端。
    def initialize(self):
        self._initialized = True

    # 统一推理接口：输入图像 + 参数，输出标准字典结构。
    def infer(self, image: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class PaddleOCRServiceAdapter(BaseExternalModelAdapter):
    """HTTP adapter for PP-OCRv5/general OCR serving endpoints."""

    _DEFAULT_REQUEST_OPTION_KEYS = (
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_textline_orientation",
        "text_det_limit_side_len",
        "text_det_limit_type",
        "text_det_thresh",
        "text_det_box_thresh",
        "text_det_unclip_ratio",
        "text_rec_score_thresh",
        "return_word_box",
        "visualize",
    )

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.client: Optional[PaddleOCRHTTPClient] = None

    def initialize(self):
        if self._initialized:
            return
        base_url = str(self.config.get("base_url") or self.config.get("serving_base_url") or "http://127.0.0.1:8080")
        default_request_options = {
            key: self.config[key]
            for key in self._DEFAULT_REQUEST_OPTION_KEYS
            if key in self.config
        }
        self.client = PaddleOCRHTTPClient(
            base_url=base_url,
            request_timeout=int(self.config.get("request_timeout", 180)),
            default_file_type=int(self.config.get("default_file_type", 1)),
            default_visualize=self.config.get("visualize"),
            line_y_threshold=float(self.config.get("line_y_threshold", 0.6)),
            default_request_options=default_request_options,
        )
        self._initialized = True

    def infer(self, image: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        if self.client is None:
            raise RuntimeError("PaddleOCR service client is not initialized.")
        if not isinstance(image, Image.Image):
            raise RuntimeError("PaddleOCR service adapter expects PIL image input.")
        return self.client.infer(image, kwargs or {})


class PaddleOCRVLAdapter(BaseExternalModelAdapter):
    """HTTP adapter for PaddleOCR-VL service endpoints."""

    # HTTP-only override: call PaddleOCR serving API instead of local PaddleOCR Python pipeline.
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.base_url = ""
        self.layout_url = ""
        self.restructure_url = ""
        self.request_timeout = 180
        self.default_file_type = 1
        self.default_visualize: Optional[bool] = None
        self.default_restructure_pages = False

    def initialize(self):
        if self._initialized:
            return
        base = str(self.config.get("serving_base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.base_url = base
        self.layout_url = f"{base}/layout-parsing"
        self.restructure_url = f"{base}/restructure-pages"
        self.request_timeout = int(self.config.get("request_timeout", 180))
        self.default_file_type = int(self.config.get("default_file_type", 1))
        self.default_visualize = self.config.get("visualize")
        self.default_restructure_pages = bool(self.config.get("restructure_pages", False))
        self._initialized = True

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _decode_base64_image(self, data: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def _extract_text_from_pages(self, pages: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for page in pages:
            markdown = page.get("markdown") or {}
            md_text = markdown.get("text")
            if isinstance(md_text, str) and md_text.strip():
                chunks.append(md_text.strip())
                continue
            pruned = page.get("prunedResult")
            if pruned is not None:
                chunks.append(json.dumps(pruned, ensure_ascii=False))
        return "\n\n".join(chunks)[:8000]

    def _request_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OCR HTTP {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        if data.get("errorCode", 0) != 0:
            raise RuntimeError(f"OCR service error: code={data.get('errorCode')} msg={data.get('errorMsg')}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"OCR response missing result object: {str(data)[:800]}")
        return result

    def _build_layout_payload(self, image: Image.Image, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self._image_to_base64(image),
            "fileType": int(kwargs.get("fileType", kwargs.get("file_type", self.default_file_type))),
        }
        field_map = {
            "use_doc_orientation_classify": "useDocOrientationClassify",
            "use_doc_unwarping": "useDocUnwarping",
            "use_layout_detection": "useLayoutDetection",
            "use_chart_recognition": "useChartRecognition",
            "use_seal_recognition": "useSealRecognition",
            "use_ocr_for_image_block": "useOcrForImageBlock",
            "layout_threshold": "layoutThreshold",
            "layout_nms": "layoutNms",
            "layout_unclip_ratio": "layoutUnclipRatio",
            "layout_merge_bboxes_mode": "layoutMergeBboxesMode",
            "layout_shape_mode": "layoutShapeMode",
            "prompt_label": "promptLabel",
            "format_block_content": "formatBlockContent",
            "repetition_penalty": "repetitionPenalty",
            "temperature": "temperature",
            "top_p": "topP",
            "min_pixels": "minPixels",
            "max_pixels": "maxPixels",
            "max_new_tokens": "maxNewTokens",
            "merge_layout_blocks": "mergeLayoutBlocks",
            "markdown_ignore_labels": "markdownIgnoreLabels",
            "vlm_extra_args": "vlmExtraArgs",
            "prettify_markdown": "prettifyMarkdown",
            "show_formula_number": "showFormulaNumber",
            "restructure_pages": "restructurePages",
            "merge_tables": "mergeTables",
            "relevel_titles": "relevelTitles",
            "visualize": "visualize",
        }
        for src, dst in field_map.items():
            if src in kwargs:
                payload[dst] = kwargs[src]
            elif dst in kwargs:
                payload[dst] = kwargs[dst]
        if "visualize" not in payload and self.default_visualize is not None:
            payload["visualize"] = self.default_visualize
        if "restructurePages" not in payload and self.default_restructure_pages:
            payload["restructurePages"] = True
        return payload

    def infer(self, image: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        if not isinstance(image, Image.Image):
            raise RuntimeError("PaddleOCR-VL adapter expects PIL image input.")

        layout_payload = self._build_layout_payload(image.convert("RGB"), kwargs)
        layout_result = self._request_json(self.layout_url, layout_payload)

        pages = layout_result.get("layoutParsingResults") or []
        if not isinstance(pages, list):
            pages = []
        final_pages = pages
        do_restructure = bool(layout_payload.get("restructurePages", False))

        if do_restructure and pages:
            pages_req = []
            for page in pages:
                markdown = page.get("markdown") or {}
                pages_req.append(
                    {
                        "prunedResult": page.get("prunedResult"),
                        "markdownImages": markdown.get("images"),
                    }
                )
            restructure_payload = {
                "pages": pages_req,
                "mergeTables": bool(kwargs.get("merge_tables", kwargs.get("mergeTables", False))),
                "relevelTitles": bool(kwargs.get("relevel_titles", kwargs.get("relevelTitles", False))),
                "concatenatePages": bool(kwargs.get("concatenate_pages", kwargs.get("concatenatePages", True))),
                "prettifyMarkdown": bool(kwargs.get("prettify_markdown", True)),
                "showFormulaNumber": bool(kwargs.get("show_formula_number", False)),
            }
            restructure_result = self._request_json(self.restructure_url, restructure_payload)
            final_pages = restructure_result.get("layoutParsingResults") or []

        text = self._extract_text_from_pages(final_pages) or "OCR completed but no markdown text extracted."

        out_images = [image.copy()]
        if final_pages:
            first_page = final_pages[0]
            output_images = first_page.get("outputImages") or {}
            if isinstance(output_images, dict):
                for _, b64_img in output_images.items():
                    try:
                        out_images = [self._decode_base64_image(b64_img)]
                        break
                    except Exception:
                        pass

        return {
            "images": out_images,
            "text": text,
            "meta": {
                "model": "paddleocr_vl_http",
                "layout_result": layout_result,
                "used_restructure_pages": do_restructure,
            },
        }


class GroundedSAM2ServiceAdapter(BaseExternalModelAdapter):
    """HTTP adapter for GroundedSAM2 service endpoints."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.client: Optional[GroundedSAM2HTTPClient] = None

    def initialize(self):
        if self._initialized:
            return
        base_url = str(self.config.get("base_url") or "http://127.0.0.1:8081")
        self.client = GroundedSAM2HTTPClient(
            base_url=base_url,
            request_timeout=int(self.config.get("request_timeout", 180)),
        )
        self._initialized = True

    def infer(self, image: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        if self.client is None:
            raise RuntimeError("GroundedSAM2 service client is not initialized.")
        if not isinstance(image, Image.Image):
            raise RuntimeError("GroundedSAM2 service adapter expects PIL image input.")
        return self.client.infer(image, kwargs or {})


def init_code_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize code execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(CodeExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class CodeImageTool(BaseTool):
    """A unified tool for image processing using executable Python code.

    This tool allows MLLM to write Python code to perform various image operations
    including zoom, flip, rotate, contrast, brightness adjustments, and more.
    The tool provides a safe execution environment with predefined image processing
    libraries and utilities.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the image processing code
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.max_code_length = config.get("max_code_length", 2000)
        self.enable_external_model_functions = config.get("enable_external_model_functions", True)
        self.external_call_mode = str(config.get("external_call_mode", "service")).strip().lower()
        if self.external_call_mode != "service":
            raise RuntimeError(
                "CodeImageTool external model path is now service-only. "
                "Set external_call_mode='service' and use HTTP backends."
            )
        self.external_services_config = config.get("external_services", {}) or {}
        self.ocr_model_name = self._resolve_ocr_model_name(config)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_code_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.service_model_clients: dict[str, BaseExternalModelAdapter] = {}
        # Service mode reuses HTTP-capable adapters and leaves room for future dedicated clients.
        self.service_adapter_registry: dict[str, type[BaseExternalModelAdapter]] = {
            "ocr": PaddleOCRServiceAdapter,
            "paddleocr": PaddleOCRServiceAdapter,
            "paddleocr_v5": PaddleOCRServiceAdapter,
            "paddleocr_vl": PaddleOCRVLAdapter,
            "grounded_sam2": GroundedSAM2ServiceAdapter,
        }
        logger.info(f"Initialized CodeImageTool with config: {config}")

    def _get_service_model_config(self, model_name: str) -> dict[str, Any]:
        normalized = str(model_name).strip().lower()
        candidates = [normalized]
        if normalized in {"ocr", "paddleocr", "paddleocr_v5"}:
            candidates.extend(["paddleocr", "paddleocr_v5", "ocr"])
        if normalized == "paddleocr_vl":
            candidates.append("paddleocr_vl")

        for candidate in candidates:
            model_config = self.external_services_config.get(candidate)
            if model_config is not None:
                return model_config
        return {}

    def _resolve_ocr_model_name(self, config: dict[str, Any]) -> str:
        explicit = str(config.get("ocr_model_name") or "").strip().lower()
        if explicit:
            return explicit
        for candidate in ("paddleocr", "paddleocr_v5", "ocr", "paddleocr_vl"):
            if candidate in self.external_services_config:
                return candidate
        return "paddleocr"

    def _get_service_model_client(self, model_name: str) -> BaseExternalModelAdapter:
        normalized = str(model_name).strip().lower()
        if normalized not in self.service_model_clients:
            adapter_cls = self.service_adapter_registry.get(normalized)
            if adapter_cls is None:
                available = sorted(self.service_adapter_registry.keys())
                raise RuntimeError(
                    f"Service mode has no client registered for model '{model_name}'. Available service clients: {available}"
                )
            model_config = self._get_service_model_config(normalized)
            self.service_model_clients[normalized] = adapter_cls(model_config or {})
        return self.service_model_clients[normalized]

    # 调用 HTTP service client 执行外部模型推理，并转换为统一返回格式。
    def _call_external_model(self, model_name: str, image: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.enable_external_model_functions:
            raise RuntimeError("External model functions are disabled.")
        client = self._get_service_model_client(model_name)
        result = client.infer(image, kwargs or {})
        images = result.get("images", [])
        return {
            "image": images[0] if images else image,
            "images": images,
            "text": result.get("text", ""),
            "meta": result.get("meta", {}),
        }

    def _validate_image_size(self, image: Any) -> tuple[bool, Optional[int], Optional[int], Optional[str]]:
        """Validate output image dimensions and aspect ratio, consistent with _sanitize_images_for_processor.

        Rules:
        - width > 0 and height > 0
        - extreme aspect ratio is invalid: max(w/h, h/w) >= 200.0

        Returns: (is_valid, width, height, error_message)
        """
        try:
            width_val: Optional[int] = None
            height_val: Optional[int] = None
            # PIL Image-like
            if hasattr(image, "size"):
                w, h = image.size
                width_val = int(w)
                height_val = int(h)
            else:
                # Array/tensor-like
                shape = getattr(image, "shape", None)
                if shape is not None and len(shape) >= 2:
                    height_val = int(shape[-2])
                    width_val = int(shape[-1])

            if width_val is None or height_val is None:
                return True, width_val, height_val, None  # Unknown type; treat as valid to avoid false negatives
            if width_val <= 0 or height_val <= 0:
                return False, width_val, height_val, f"The result has an invalid image size ({width_val}x{height_val}). Width and height must be positive. Please check your code."
            # Check extreme aspect ratio (align with _sanitize_images_for_processor threshold)
            try:
                aspect = max(float(width_val) / float(height_val), float(height_val) / float(width_val))
                if aspect >= 200.0:
                    return False, width_val, height_val, f"The result has an invalid image aspect ratio ({width_val}x{height_val}, aspect={aspect:.2f}) >= 200.0. Please check your code."
            except Exception:
                pass
            return True, width_val, height_val, None
        except Exception as e:
            return False, None, None, f"Failed to validate image dimensions: {str(e)}. Please check your code."

    def _normalize_helper_trace(self, helper_trace: Any) -> list[dict[str, Any]]:
        """Normalize helper trace into a stable list of {order, name, status} records."""
        normalized: list[dict[str, Any]] = []
        if not isinstance(helper_trace, list):
            return normalized

        for idx, item in enumerate(helper_trace, start=1):
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or "").strip()
            if not name:
                continue

            order = item.get("order")
            if not isinstance(order, int) or order < 1:
                order = idx

            status = item.get("status")
            if status not in {"ok", "error"}:
                status = None

            normalized.append({"order": order, "name": name, "status": status})

        return normalized

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Validate the Python code for safety and syntax."""
        try:
            # Check code length
            if len(code) > self.max_code_length:
                return False, f"Code too long. Maximum allowed length: {self.max_code_length}"

            # Parse the code to check syntax
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_imports = [
                'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
                'open', 'file', 'input', 'raw_input', '__import__'
            ]
            
            dangerous_functions = [
                'exit', 'quit', 'help', 'dir', 'vars', 'globals', 'locals'
            ]
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            return False, f"Dangerous import detected: {alias.name}"
                
                if isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        return False, f"Dangerous import detected: {node.module}"
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                        return False, f"Dangerous function call detected: {node.func.id}"
                
                # Check for attribute access to dangerous modules
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id in dangerous_imports:
                        return False, f"Dangerous attribute access detected: {node.value.id}.{node.attr}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Code validation error: {e}"

    def _create_safe_globals(self, image, images: list[Any], image_index: int) -> dict:
        """Create a safe global namespace for code execution."""
        import PIL
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
        import numpy as np
        import math
        
        # Try to import OpenCV, but don't fail if it's not available
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2 = None
            cv2_available = False
        
        allowed_imports = {"PIL", "PIL.Image", "PIL.ImageEnhance", "PIL.ImageFilter", "PIL.ImageOps",
                       "PIL.ImageDraw", "PIL.ImageFont", "numpy", "math", "cv2"}

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == mod or name.startswith(mod + ".") for mod in allowed_imports):
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of '{name}' is not allowed in safe mode")
        
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()
        safe_globals: dict[str, Any] = {}

        # 将 helper 返回图像设为当前“活跃图像”，便于后续链式处理。
        def _set_active_image(new_image):
            safe_globals["image"] = new_image
            safe_globals["img"] = new_image
            safe_globals["draw"] = ImageDraw.Draw(new_image)
            return new_image

        def _record_helper_result(result: dict[str, Any]):
            safe_globals["__last_helper_result__"] = result
            return result

        def _record_helper_trace(name: str, status: str):
            helper_trace = safe_globals.setdefault("__helper_trace__", [])
            helper_trace.append(
                {
                    "order": len(helper_trace) + 1,
                    "name": name,
                    "status": status,
                }
            )
            return helper_trace[-1]

        def _run_helper(name: str, fn: Callable[[], dict[str, Any]]):
            try:
                result = fn()
                _record_helper_result(result)
                _set_active_image(result["image"])
            except Exception:
                _record_helper_trace(name, "error")
                raise
            _record_helper_trace(name, "ok")
            return result

        # 解析 helper 输入图像：优先 image_obj，其次 image_index。
        def _select_image(target_index: Optional[int] = None, image_obj: Optional[Any] = None):
            if image_obj is not None:
                return image_obj
            idx = image_index if target_index is None else int(target_index)
            if idx < 0 or idx >= len(images):
                raise RuntimeError(f"image_index {idx} out of range. Available: 0..{len(images)-1}")
            return images[idx].copy()

        def _normalize_xyxy_box(
            *,
            source_image: Any,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            padding: int = 0,
        ) -> list[int]:
            if not hasattr(source_image, "size"):
                raise RuntimeError("Selected image does not provide size information.")
            width, height = source_image.size
            px1 = min(float(x1), float(x2))
            py1 = min(float(y1), float(y2))
            px2 = max(float(x1), float(x2))
            py2 = max(float(y1), float(y2))
            pad = max(0, int(padding))

            nx1 = max(0, int(round(px1 - pad)))
            ny1 = max(0, int(round(py1 - pad)))
            nx2 = min(int(width), int(round(px2 + pad)))
            ny2 = min(int(height), int(round(py2 + pad)))

            if nx2 <= nx1 or ny2 <= ny1:
                raise RuntimeError(
                    f"Invalid box after clipping: [{nx1}, {ny1}, {nx2}, {ny2}] for image size {width}x{height}."
                )
            return [nx1, ny1, nx2, ny2]

        def _build_local_helper_result(
            *,
            output_image: Any,
            text: str,
            meta: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "image": output_image,
                "images": [output_image],
                "text": text,
                "meta": meta,
            }

        # OCR helper：按配置选择 PaddleOCR 服务客户端。
        def _call_ocr_assist(image_index: Optional[int] = None, image_obj: Optional[Any] = None, **kwargs):
            return _run_helper(
                "_call_ocr_assist",
                lambda: self._call_external_model(
                    self.ocr_model_name,
                    _select_image(target_index=image_index, image_obj=image_obj),
                    kwargs,
                ),
            )

        # 本地坐标画框 helper：不依赖检测模型，直接按显式坐标画框。
        def _call_manual_box(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            outline: str = "lime",
            width: int = 2,
            label: Optional[str] = None,
            label_fill: Optional[str] = None,
        ):
            def _impl() -> dict[str, Any]:
                selected = _select_image(target_index=image_index, image_obj=image_obj)
                bbox = _normalize_xyxy_box(
                    source_image=selected,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                out = selected.copy()
                draw_obj = ImageDraw.Draw(out)
                draw_obj.rectangle(tuple(bbox), outline=outline, width=max(1, int(width)))
                if label is not None:
                    text_fill = label_fill or outline
                    text_x = bbox[0] + 2
                    text_y = max(0, bbox[1] - 14)
                    draw_obj.text((text_x, text_y), str(label), fill=text_fill)
                return _build_local_helper_result(
                    output_image=out,
                    text=f"Manual box drawn at {bbox}.",
                    meta={
                        "model": "local_geometry",
                        "operation": "manual_box",
                        "bbox": bbox,
                        "outline": outline,
                        "width": max(1, int(width)),
                        "label": None if label is None else str(label),
                    },
                )

            return _run_helper("_call_manual_box", _impl)

        # 本地坐标裁剪 helper：不依赖检测模型，直接按显式坐标裁剪。
        def _call_manual_crop(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            padding: int = 0,
        ):
            def _impl() -> dict[str, Any]:
                selected = _select_image(target_index=image_index, image_obj=image_obj)
                crop_box = _normalize_xyxy_box(
                    source_image=selected,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    padding=padding,
                )
                cropped = selected.crop(tuple(crop_box))
                return _build_local_helper_result(
                    output_image=cropped,
                    text=f"Manual crop returned 1 crop image from {crop_box}.",
                    meta={
                        "model": "local_geometry",
                        "operation": "manual_crop",
                        "crop_box": crop_box,
                        "padding": max(0, int(padding)),
                    },
                )

            return _run_helper("_call_manual_crop", _impl)

        # Grounding helper：仅绘制检测框。
        def _call_ground_box(
            text_prompt: str,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
            **kwargs,
        ):
            call_kwargs = {
                "_operation": "box",
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            }
            call_kwargs.update(kwargs)
            return _run_helper(
                "_call_ground_box",
                lambda: self._call_external_model(
                    "grounded_sam2",
                    _select_image(target_index=image_index, image_obj=image_obj),
                    call_kwargs,
                ),
            )

        # SAM2 helper：输出 mask 半透明高亮图。
        def _call_sam_mask(
            text_prompt: str,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
            multimask_output: bool = False,
            mask_alpha: float = 0.45,
            draw_box_on_mask: bool = True,
            **kwargs,
        ):
            call_kwargs = {
                "_operation": "mask",
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "multimask_output": multimask_output,
                "mask_alpha": mask_alpha,
                "draw_box_on_mask": draw_box_on_mask,
            }
            call_kwargs.update(kwargs)
            return _run_helper(
                "_call_sam_mask",
                lambda: self._call_external_model(
                    "grounded_sam2",
                    _select_image(target_index=image_index, image_obj=image_obj),
                    call_kwargs,
                ),
            )

        # 裁剪 helper：按 box/mask 返回裁剪图像。
        def _call_dino_crop(
            text_prompt: str,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            based_on: str = "box",
            detection_index: int = 0,
            max_crops: int = 1,
            padding: int = 0,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
            multimask_output: bool = False,
            **kwargs,
        ):
            call_kwargs = {
                "_operation": "dino_crop",
                "text_prompt": text_prompt,
                "based_on": based_on,
                "detection_index": detection_index,
                "max_crops": max_crops,
                "padding": padding,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "multimask_output": multimask_output,
            }
            call_kwargs.update(kwargs)
            return _run_helper(
                "_call_dino_crop",
                lambda: self._call_external_model(
                    "grounded_sam2",
                    _select_image(target_index=image_index, image_obj=image_obj),
                    call_kwargs,
                ),
            )

        # 背景模糊 helper：前景保持清晰，背景模糊。
        def _call_blur_bg(
            text_prompt: str,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            blur_radius: float = 8.0,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
            multimask_output: bool = False,
            **kwargs,
        ):
            call_kwargs = {
                "_operation": "blur_bg",
                "text_prompt": text_prompt,
                "blur_radius": blur_radius,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "multimask_output": multimask_output,
            }
            call_kwargs.update(kwargs)
            return _run_helper(
                "_call_blur_bg",
                lambda: self._call_external_model(
                    "grounded_sam2",
                    _select_image(target_index=image_index, image_obj=image_obj),
                    call_kwargs,
                ),
            )

        # 兼容别名：保留 _call_focus 旧名称，行为等同 _call_ground_box。
        def _call_focus(
            text_prompt: str,
            image_index: Optional[int] = None,
            image_obj: Optional[Any] = None,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
            multimask_output: bool = False,
            **kwargs,
        ):
            call_kwargs = {
                "multimask_output": multimask_output,
            }
            call_kwargs.update(kwargs)
            helper_kwargs = {
                "_operation": "box",
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            }
            helper_kwargs.update(call_kwargs)
            return _run_helper(
                "_call_focus",
                lambda: self._call_external_model(
                    "grounded_sam2",
                    _select_image(target_index=image_index, image_obj=image_obj),
                    helper_kwargs,
                ),
            )

        safe_globals.update({
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
                'print': print, 'isinstance': isinstance, 'type': type,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                'math': math, 'np': np,
                '__import__': safe_import,
            },
            # Image processing libraries
            'PIL': PIL,
            'Image': Image,
            'ImageEnhance': ImageEnhance,
            'ImageFilter': ImageFilter,
            'ImageOps': ImageOps,
            'ImageDraw': ImageDraw,
            'ImageFont': ImageFont,
            'numpy': np,
            'math': math,
            # The input image
            'image': image_copy,
            'img': image_copy,  # Alias for convenience
            # Common constants
            'PI': math.pi,
            'E': math.e,
            # Drawing utilities
            'draw': ImageDraw.Draw(image_copy),
            '__last_helper_result__': None,
            '__helper_trace__': [],
            '_call_ocr_assist': _call_ocr_assist,
            '_call_manual_box': _call_manual_box,
            '_call_manual_crop': _call_manual_crop,
            '_call_ground_box': _call_ground_box,
            '_call_sam_mask': _call_sam_mask,
            '_call_dino_crop': _call_dino_crop,
            '_call_blur_bg': _call_blur_bg,
            '_call_focus': _call_focus,
        })
        
        # Add OpenCV if available
        if cv2_available:
            safe_globals['cv2'] = cv2
            safe_globals['cv'] = cv2  # Alias for convenience
        
        return safe_globals

    def _execute_code(
        self,
        code: str,
        image,
        images: list[Any],
        image_index: int,
    ) -> tuple[Any, str, Optional[dict[str, Any]], list[dict[str, Any]], str, str]:
        """Execute the provided code safely and return the result."""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        safe_globals: Optional[dict[str, Any]] = None
        try:
            # Validate code first
            is_valid, validation_msg = self._validate_code(code)
            if not is_valid:
                return None, f"Code validation failed: {validation_msg}", None, [], "", ""
            
            # Create safe execution environment
            safe_globals = self._create_safe_globals(image, images=images, image_index=image_index)
            safe_locals = {}
            
            # Execute the code
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, safe_globals, safe_locals)
            
            # Try to get the result from common variable names
            result = None
            for var_name in ['result', 'output', 'processed_image', 'img', 'image']:
                if var_name in safe_locals:
                    result = safe_locals[var_name]
                    break
                elif var_name in safe_globals and safe_globals[var_name] is not image:
                    result = safe_globals[var_name]
                    break
            
            # If no result found, return the modified image
            if result is None:
                result = safe_globals.get('image', image)
            
            helper_result = safe_globals.get("__last_helper_result__")
            helper_trace = self._normalize_helper_trace(safe_globals.get("__helper_trace__", []))
            return (
                result,
                "Code executed successfully",
                helper_result,
                helper_trace,
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
            )
            
        except Exception as e:
            helper_result = safe_globals.get("__last_helper_result__") if isinstance(safe_globals, dict) else None
            helper_trace = self._normalize_helper_trace(
                safe_globals.get("__helper_trace__", []) if isinstance(safe_globals, dict) else []
            )
            return (
                None,
                f"Code execution error: {str(e)}",
                helper_result,
                helper_trace,
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
            )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for code-based image processing tool.

        Args:
            instance_id: An optional unique identifier for the instance.
            **kwargs: Should contain 'image' key with image data (single image or list of images).

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        # Handle both single image and list of images
        if isinstance(image, list):
            # Process list of images
            images = []
            for img_data in image:
                img = fetch_image({"image": img_data})
                images.append(img)
        else:
            # Single image
            img = fetch_image({"image": image})
            images = [img]

        self._instance_dict[instance_id] = {
            "images": images,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        Execute the image processing code.

        Args:
            instance_id: The instance id of the tool.
            parameters: Dictionary containing 'code', 'description', and 'image_index' parameters.

        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        code = parameters.get("code", "")
        image_index = parameters.get("image_index", 0)
        empty_execution_observation = {
            "observed_helper_call_count": 0,
            "observed_helper_calls": [],
            "stdout_text": "",
            "stderr_text": "",
        }
        
        if not code or not isinstance(code, str):
            return (
                ToolResponse(text="Error: 'code' parameter is missing or not a string. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "missing_code", **empty_execution_observation},
            )

        if not isinstance(image_index, int) or image_index < 0:
            return (
                ToolResponse(text="Error: 'image_index' must be a non-negative integer. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "invalid_image_index", **empty_execution_observation},
            )

        if instance_id not in self._instance_dict:
            return (
                ToolResponse(text="Error: Instance not found. Please create an instance first."),
                -0.05,
                {"success": False, "error": "instance_not_found", **empty_execution_observation},
            )

        instance_data = self._instance_dict[instance_id]
        images = instance_data["images"]
        
        # Validate image_index
        if image_index >= len(images):
            return (
                ToolResponse(text=f"Error: image_index {image_index} is out of range. Available images: 0 to {len(images)-1}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {"success": False, "error": "image_index_out_of_range", **empty_execution_observation},
            )
        
        # Select the image to process
        image = images[image_index]

        try:
            # Execute the code using the worker pool
            result, message, helper_result, helper_trace, stdout_text, stderr_text = ray.get(
                self.execution_pool.execute.remote(self._execute_code, code, image, images, image_index)
            )

            observed_helper_calls = self._normalize_helper_trace(helper_trace)
            execution_observation = {
                "observed_helper_call_count": len(observed_helper_calls),
                "observed_helper_calls": observed_helper_calls,
                "stdout_text": str(stdout_text or ""),
                "stderr_text": str(stderr_text or ""),
            }

            helper_text = ""
            helper_meta = None
            if isinstance(helper_result, dict):
                helper_text = str(helper_result.get("text", "") or "").strip()
                raw_meta = helper_result.get("meta")
                if isinstance(raw_meta, dict):
                    helper_meta = raw_meta
            
            if result is None:
                return (
                    ToolResponse(text=f"Error: {message}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                    -0.05,
                    {
                        "success": False,
                        "error": "execution_failed",
                        "message": message,
                        "helper_text": helper_text,
                        "helper_meta": helper_meta,
                        **execution_observation,
                    },
                )
            
            # Ensure result is a PIL Image
            if hasattr(result, 'save'):  # It's already a PIL Image
                processed_image = result
            else:
                return (
                    ToolResponse(text="Error: Code must return a PIL Image object."),
                    -0.05,
                    {
                        "success": False,
                        "error": "invalid_return_type",
                        "helper_text": helper_text,
                        "helper_meta": helper_meta,
                        **execution_observation,
                    },
                )

            # Validate processed image size (output image)
            is_valid, w, h, err = self._validate_image_size(processed_image)
            if not is_valid:
                return (
                    ToolResponse(text=f"Error: {err}" if err else "Error: Invalid output image size."),
                    -0.05,
                    {
                        "success": False,
                        "error": "invalid_output_image_size",
                        "width": w,
                        "height": h,
                        "processed_image_index": image_index,
                        "helper_text": helper_text,
                        "helper_meta": helper_meta,
                        **execution_observation,
                    },
                )
            response_text = helper_text or message
            if response_text:
                response_text = f"{response_text}\n\n{SUCCESS_FOLLOWUP_TEXT}"
            else:
                response_text = SUCCESS_FOLLOWUP_TEXT
            
            return (
                ToolResponse(
                    image=[processed_image],
                    text=response_text,
                    meta=helper_meta,
                ),
                0.0,
                {
                    "success": True,
                    "message": message,
                    "helper_text": helper_text,
                    "helper_meta": helper_meta,
                    "processed_image_index": image_index,
                    "total_images": len(images),
                    **execution_observation,
                },
            )
            
        except Exception as e:
            logger.error(f"Error in code-based image processing on image {image_index}: {e}")
            return (
                ToolResponse(text=f"Error processing image {image_index}: {str(e)}. Please keep thinking step-by-step inside the <think></think> tags to determine the next action. If additional tools are required, call them inside the <tool_calls></tool_calls> tags. Otherwise, provide your final answer within the <answer></answer> tags."),
                -0.05,
                {
                    "success": False,
                    "error": "unexpected_error",
                    "message": str(e),
                    "processed_image_index": image_index,
                    **empty_execution_observation,
                },
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
