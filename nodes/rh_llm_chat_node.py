"""
LLFaigc - RH LLM API Node (OpenAI-compatible).

Ported from the official ComfyUI_RH_LLM_API plugin (HM-RunningHub).
Extended to support up to 8 reference images (instead of 1 in the original).

Uses OpenAI-compatible chat.completions API (NOT RunningHub submit/poll protocol).
Supports text, multiple images, and video inputs.

Original: https://github.com/HM-RunningHub/ComfyUI_RH_LLM_API
"""

import base64
import os
import io
import time

import numpy as np
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core.api_key import get_config


def encode_image_b64(ref_image):
    """Encode ComfyUI IMAGE tensor to base64 JPEG without resizing.

    - Keeps original resolution (no resize), matching the original plugin behavior.
    - In-memory encoding (no temp files).
    """
    i = 255.0 * ref_image.cpu().numpy()
    if len(i.shape) == 4:
        i = i[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_video_file_path(video):
    """Try to extract a filesystem path from a ComfyUI VIDEO object."""
    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path

    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass

    for attr in ("path", "file_path", "file"):
        if hasattr(video, attr):
            p = getattr(video, attr, None)
            if isinstance(p, str) and os.path.exists(p):
                return p

    return None


def encode_video_b64(video):
    """Encode ComfyUI VIDEO object to base64 MP4 bytes.

    No ffmpeg, no compression, no resizing (matching original plugin behavior).
    """
    video_path = _get_video_file_path(video)
    if video_path:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if hasattr(video, "save_to"):
        temp_path = f"temp_video_{int(time.time())}.mp4"
        try:
            video.save_to(temp_path)
            with open(temp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    raise ValueError(f"Unable to read video data from object type: {type(video)}")


class LLFaigcRHLLMChat:
    """OpenAI-compatible LLM API node (ported from official ComfyUI_RH_LLM_API).

    Supports:
    - Text only (no media connected)
    - Up to 8 reference images
    - 1 video input

    Uses OpenAI chat.completions API (requires `openai` Python package).
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "rh_run_llmapi"
    CATEGORY = "LLFaigc/RH-LLM"

    # Default system message (can be overridden by including instructions in prompt)
    DEFAULT_SYSTEM_MSG = "You are a helpful assistant."

    # Model list for OpenAI-compatible Chat Completions API.
    # Users can still type a custom model name in ComfyUI's combo widget.
    MODEL_OPTIONS = [
        "RH-G-3-Pro-Preview",
        "RH-G-3-Flash-Preview",
        "RH-G-25-Pro",
        "RH-G-25-Flash",
        "Qwen-27b",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "gemini-pro",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_OPTIONS, {"default": "RH-G-3-Pro-Preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter your prompt here. You can include role/instruction at the beginning of your prompt."}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "api_config": ("RH_OPENAPI_CONFIG",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "video": ("VIDEO",),
                "skip_error": ("BOOLEAN", {"default": False}),
            }
        }

    def rh_run_llmapi(
        self,
        model,
        prompt,
        temperature,
        api_config=None,
        image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
        video=None,
        skip_error=False,
    ):
        log_prefix = "LLF-RH-LLM-Chat"

        if OpenAI is None:
            msg = "openai package not installed. Please install it with: pip install openai"
            print(f"[{log_prefix}] {msg}")
            if not skip_error:
                raise RuntimeError(msg)
            return (msg,)

        # Resolve api_key and base_url from api_config
        config = get_config(api_config)
        api_key = config["api_key"]
        base_url = config["base_url"]

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Collect connected images
        images = []
        for i, img in enumerate([image_1, image_2, image_3, image_4,
                                  image_5, image_6, image_7, image_8], 1):
            if img is not None:
                images.append((i, img))

        # Build messages (priority: video > images > text)
        # System message uses default; user can include role instructions in prompt text
        if video is not None:
            print(f"[{log_prefix}] Encoding video input...")
            base64_video = encode_video_b64(video)
            messages = [
                {"role": "system", "content": self.DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": [
                    {"type": "text", "text": str(prompt)},
                    {"type": "video_url", "video_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }},
                ]},
            ]
        elif images:
            print(f"[{log_prefix}] Encoding {len(images)} image(s)...")
            content_parts = [{"type": "text", "text": str(prompt)}]
            for idx, img in images:
                b64 = encode_image_b64(img)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            messages = [
                {"role": "system", "content": self.DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": content_parts},
            ]
        else:
            messages = [
                {"role": "system", "content": self.DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": str(prompt)},
            ]

        # Debug log (truncate base64)
        debug_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                debug_content = []
                for item in msg["content"]:
                    item_copy = dict(item)
                    if item_copy.get("type") == "image_url":
                        url = item_copy["image_url"]["url"]
                        item_copy["image_url"]["url"] = url[:60] + f"...[{len(url)} chars]"
                    elif item_copy.get("type") == "video_url":
                        url = item_copy["video_url"]["url"]
                        item_copy["video_url"]["url"] = url[:60] + f"...[{len(url)} chars]"
                    debug_content.append(item_copy)
                debug_messages.append({"role": msg["role"], "content": debug_content})
            else:
                debug_messages.append(msg)
        print(f"[{log_prefix}] Request messages: {debug_messages}")

        # Call API
        try:
            print(f"[{log_prefix}] Calling {model} via {base_url}")
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            if completion is not None and hasattr(completion, "choices"):
                result = completion.choices[0].message.content
                print(f"[{log_prefix}] Response received: {len(result)} chars")
            else:
                result = "Error: No response from API"
                print(f"[{log_prefix}] {result}")
        except Exception as e:
            result = f"Error calling API: {e}"
            print(f"[{log_prefix}] API call failed: {e}")
            import traceback
            traceback.print_exc()
            if not skip_error:
                raise

        return (result,)
