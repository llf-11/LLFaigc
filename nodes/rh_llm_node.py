"""
LLFaigc - RH LLM node (RunningHub unified multimodal text node).

Merges all 17 RH text models (G-3-flash, G-3-pro, G-25-flash, G-25-pro,
Qwen-27b, CV variants) into a single unified node.

Features:
  - Model dropdown with all RH text models grouped by family
  - Single reference image (IMAGE input)
  - 1 video input (VIDEO)
  - Auto-detection: text-to-text when no media is connected
  - Dynamic endpoint switching based on connected media type
  - Uses RunningHub submit/poll API protocol (NOT OpenAI-compatible)
"""

import os
import json
from typing import Dict, List, Tuple, Optional

import torch

from ..core.base import BaseNode
from ..core.api_key import get_config
from ..core.upload import upload_file
from ..core.image import tensor_to_bytes
from ..core.task import submit, poll


# ---------------------------------------------------------------------------
# Model registry: (display_name, endpoint_base, supports_modes)
#
# endpoint_base is the model identifier WITHOUT the mode suffix.
# Mode suffixes (/text-to-text, /image-to-text, /video-to-text) are appended
# dynamically based on connected media.
#
# supports_modes: which modes this model supports.
#   "t2t" = text-to-text, "i2t" = image-to-text, "v2t" = video-to-text
#
# Special models with non-standard endpoints use "full_endpoints" instead.
# ---------------------------------------------------------------------------

_MODEL_GROUPS = [
    ("G-3-Pro-Preview", [
        ("G-3-Pro-Preview", "rhart-text-g-3-pro-preview", ("t2t", "i2t", "v2t")),
    ]),
    ("G-3-Flash-Preview", [
        ("G-3-Flash-Preview", "rhart-text-g-3-flash-preview", ("t2t", "i2t", "v2t")),
    ]),
    ("G-25-Pro", [
        ("G-25-Pro", "rhart-text-g-25-pro", ("t2t", "i2t", "v2t")),
    ]),
    ("G-25-Flash", [
        ("G-25-Flash", "rhart-text-g-25-flash", ("t2t", "i2t", "v2t")),
    ]),
    ("G-3-Pro-Preview-CV (Computer Vision)", [
        ("G-3-Pro-Preview-CV", "rhart-text-g-3-pro-preview-cv", ("i2t",)),
    ]),
    ("G-3-Flash-Preview-CV (Computer Vision)", [
        ("G-3-Flash-Preview-CV", "rhart-text-g-3-flash-preview-cv", ("i2t",)),
    ]),
    ("G-25-Flash-CV (Computer Vision)", [
        ("G-25-Flash-CV", "rhart-text-g-25-flash-cv", ("i2t",)),
    ]),
    ("G-25-Pro-CV (Computer Vision)", [
        ("G-25-Pro-CV", "rhart-text-g-25-pro-cv", ("i2t",)),
    ]),
    ("Qwen-27b (Unified Multimodal Chat)", [
        ("Qwen-27b", "rhart-text-qwen-27b/chat", ("t2t", "i2t", "v2t")),
    ]),
]

# Build flat lookup: model_key -> (endpoint_base, supports_modes)
_MODEL_LOOKUP: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
_MODEL_DISPLAY_LIST: List[str] = []
_GROUP_HEADER_MAP: Dict[str, str] = {}

for group_name, models in _MODEL_GROUPS:
    for display, endpoint_base, modes in models:
        key = f"{display} ({endpoint_base})"
        _MODEL_LOOKUP[key] = (endpoint_base, modes)
        _MODEL_DISPLAY_LIST.append(key)
        _GROUP_HEADER_MAP[key] = group_name

# Build flat model list (ComfyUI combo does not support non-selectable headers)
MODEL_OPTIONS = []
for group_name, models in _MODEL_GROUPS:
    for display, endpoint_base, modes in models:
        key = f"{display} ({endpoint_base})"
        MODEL_OPTIONS.append(key)

# Default to G-3-Pro-Preview
DEFAULT_MODEL_KEY = "G-3-Pro-Preview (rhart-text-g-3-pro-preview)"


def _resolve_model_key(model_selection: str) -> Optional[str]:
    """Resolve a model selection to a valid model key.

    Handles group headers (which start with '---') by returning None.
    """
    if not model_selection or model_selection.startswith("---"):
        return None
    return model_selection if model_selection in _MODEL_LOOKUP else None


class LLFaigcRHLLM(BaseNode):
    """Unified RunningHub multimodal LLM node.

    Supports text-to-text, image-to-text (single image), and video-to-text.
    Automatically selects the correct API endpoint mode based on connected media.

    Uses RunningHub OpenAPI submit/poll protocol.
    """

    CATEGORY = "LLFaigc/RH-LLM"
    FUNCTION = "execute"
    OUTPUT_TYPE = "string"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "url", "response")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {"default": DEFAULT_MODEL_KEY}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "api_config": ("RH_OPENAPI_CONFIG",),
                "skip_error": ("BOOLEAN", {"default": False}),
            }
        }

    @property
    def _log_prefix(self) -> str:
        return "RH_OpenAPI_LLFaigcRHLLM"

    def prepare_inputs(self, **kwargs) -> Dict:
        """Upload connected image and video to RunningHub."""
        uploaded = {}

        config = get_config(kwargs.get("api_config"))
        api_key = config["api_key"]
        base_url = config["base_url"]
        timeout = config.get("upload_timeout", 60)

        # Upload single image
        img = kwargs.get("image")
        if img is not None:
            img_bytes = tensor_to_bytes(img)
            filename = f"upload_rhllm_img_{hash(img_bytes) % 10**10}.png"
            url = upload_file(
                img_bytes, filename, "image/png",
                api_key, base_url,
                timeout=timeout,
                logger_prefix=self._log_prefix,
            )
            uploaded["__image_url"] = url
            print(f"[{self._log_prefix}] Uploaded image: {url[:100]}...")

        # Upload video
        video = kwargs.get("video")
        if video is not None:
            vbytes = self._read_video_bytes(video)
            if vbytes:
                filename = f"upload_rhllm_video_{hash(vbytes) % 10**10}.mp4"
                url = upload_file(
                    vbytes, filename, "video/mp4",
                    api_key, base_url,
                    timeout=config.get("upload_timeout", 120),
                    logger_prefix=self._log_prefix,
                )
                uploaded["__video_url"] = url
                print(f"[{self._log_prefix}] Uploaded video: {url[:100]}...")

        return uploaded

    def build_payload(self, **kwargs) -> Dict:
        """Build the API request payload.

        Payload structure varies by model and media type:
        - text-to-text: {"prompt": "..."}
        - image-to-text: {"prompt": "...", "imageUrl": "https://..."}
        - video-to-text: {"prompt": "...", "videoUrl": "https://..."}
        """
        payload = {}

        # Add prompt
        prompt_text = str(kwargs.get("prompt", "") or "").strip()
        if prompt_text:
            payload["prompt"] = prompt_text

        # Determine media mode
        image_url = kwargs.get("__image_url")
        video_url = kwargs.get("__video_url")

        # Build media fields
        if video_url:
            payload["videoUrl"] = video_url
            # Include image as well if both are connected (Qwen-27b supports both)
            model_key = _resolve_model_key(kwargs.get("model", ""))
            endpoint_base = _MODEL_LOOKUP.get(model_key, ("", ()))[0] if model_key else ""
            if endpoint_base == "rhart-text-qwen-27b/chat" and image_url:
                payload["imageUrl"] = image_url
        elif image_url:
            payload["imageUrl"] = image_url

        return payload

    def _resolve_endpoint(self, **kwargs) -> str:
        """Resolve the actual API endpoint based on model and connected media.

        Most models use: {endpoint_base}/{mode}
        Qwen-27b uses: rhart-text-qwen-27b/chat (single endpoint for all modes)
        """
        model_key = _resolve_model_key(kwargs.get("model", ""))
        if not model_key:
            raise ValueError(f"Invalid model selection: {kwargs.get('model', '')}")

        endpoint_base, supported_modes = _MODEL_LOOKUP[model_key]

        # Qwen-27b has a unified endpoint
        if endpoint_base == "rhart-text-qwen-27b/chat":
            return endpoint_base

        # Check connected media
        image_url = kwargs.get("__image_url")
        video_url = kwargs.get("__video_url")

        if video_url and "v2t" in supported_modes:
            return f"{endpoint_base}/video-to-text"
        elif image_url and "i2t" in supported_modes:
            return f"{endpoint_base}/image-to-text"
        elif "t2t" in supported_modes:
            return f"{endpoint_base}/text-to-text"
        else:
            # Model doesn't support the current media mode
            if image_url:
                raise ValueError(
                    f"Model {model_key} does not support image input. "
                    f"Supported modes: {', '.join(supported_modes)}"
                )
            elif video_url:
                raise ValueError(
                    f"Model {model_key} does not support video input. "
                    f"Supported modes: {', '.join(supported_modes)}"
                )
            # Fallback to first available mode
            if "t2t" in supported_modes:
                return f"{endpoint_base}/text-to-text"
            elif "i2t" in supported_modes:
                return f"{endpoint_base}/image-to-text"
            else:
                return endpoint_base

    def process_result(self, result_urls: List[str]) -> tuple:
        """Extract text from poll results.

        For text nodes, result_urls contains the text content directly
        (extracted by poll() from the "text"/"content"/"output" fields).
        """
        if not result_urls:
            raise RuntimeError("No text result returned from API")

        # Join multiple text results
        text = "\n".join(result_urls)
        return (text,)

    @staticmethod
    def _read_video_bytes(video) -> Optional[bytes]:
        """Extract raw bytes from a ComfyUI VIDEO object."""
        if video is None:
            return None

        # Try common VIDEO object patterns
        if hasattr(video, "get_stream_source"):
            source = video.get_stream_source()
            if isinstance(source, str) and os.path.isfile(source):
                with open(source, "rb") as f:
                    return f.read()
            elif hasattr(source, "read"):
                return source.read()

        for attr in ("path", "file_path", "file"):
            if hasattr(video, attr):
                p = getattr(video, attr, None)
                if isinstance(p, str) and os.path.isfile(p):
                    with open(p, "rb") as f:
                        return f.read()

        if hasattr(video, "_VideoFromFile__file"):
            p = getattr(video, "_VideoFromFile__file", None)
            if isinstance(p, str) and os.path.isfile(p):
                with open(p, "rb") as f:
                    return f.read()

        print("[RH_OpenAPI_LLFaigcRHLLM] WARNING: Could not read video data")
        return None
