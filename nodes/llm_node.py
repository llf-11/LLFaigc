"""
LLFaigc - LLM API node (ported from Comfyui-zhenzhen).

Self-contained: no dependency on zhenzhen's Comfly.py or Comflyapi.json.
Supports text-only, image (up to 8), and video multimodal inputs via OpenAI-compatible API.
"""

import base64
import copy
import io
import json
import os
import subprocess
import time

import numpy as np
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Model list from gpt-best.apifox.cn
# ---------------------------------------------------------------------------

CHAT_MODELS = [
    # --- GPT Series ---
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",
    "chatgpt-4o-latest",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
    # --- Claude Series ---
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    # --- Gemini Series ---
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-3-pro-preview",
    # --- Grok Series ---
    "grok-3",
    "grok-3-mini",
    "grok-2",
    # --- Qwen Series ---
    "qwen-max",
    "qwen-plus",
    "qwen-turbo",
    "qwen-long",
    # --- DeepSeek Series ---
    "deepseek-chat",
    "deepseek-reasoner",
    # --- Llama Series ---
    "llama-3.1-405b",
    "llama-3.1-70b",
    "llama-3.1-8b",
    # --- Mistral Series ---
    "mistral-large",
    "mistral-medium",
    "mistral-small",
    # --- Yi / Baichuan / GLM ---
    "yi-large",
    "yi-medium",
    "yi-spark",
    "glm-4-plus",
    "glm-4",
    "glm-4-flash",
    # --- Other ---
    "command-r-plus",
    "command-r",
    "dbrx",
]


# ---------------------------------------------------------------------------
# Helpers: image/video encoding
# ---------------------------------------------------------------------------

def encode_image_b64(ref_image):
    """Encode ComfyUI IMAGE tensor to base64 JPEG with compression optimization."""
    try:
        # Handle batch: take first image
        if len(ref_image.shape) == 4:
            i = 255.0 * ref_image[0].cpu().numpy()
        else:
            i = 255.0 * ref_image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        original_size = img.size
        print(f"[LLF-LLM] Original image size: {original_size[0]}x{original_size[1]}")

        max_dimension = 1536
        if max(original_size) > max_dimension:
            ratio = max_dimension / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[LLF-LLM] Resized image to: {new_size[0]}x{new_size[1]}")

        formats_to_try = [
            ('JPEG', {'quality': 75, 'optimize': True}),
            ('JPEG', {'quality': 60, 'optimize': True}),
            ('JPEG', {'quality': 50, 'optimize': True}),
        ]

        best_result = None
        smallest_size = float('inf')

        for format_name, save_kwargs in formats_to_try:
            try:
                buf = io.BytesIO()
                img.save(buf, format=format_name, **save_kwargs)
                img_bytes = buf.getvalue()

                if len(img_bytes) < smallest_size:
                    smallest_size = len(img_bytes)
                    best_result = base64.b64encode(img_bytes).decode('utf-8')

                    base64_size_mb = len(best_result) / (1024 * 1024)
                    print(f"[LLF-LLM] Quality {save_kwargs['quality']}: {base64_size_mb:.2f}MB base64")

                    if base64_size_mb < 2.0:
                        break
            except Exception as e:
                print(f"[LLF-LLM] Failed encoding with quality {save_kwargs['quality']}: {e}")
                continue

        if best_result:
            final_size_mb = len(best_result) / (1024 * 1024)
            print(f"[LLF-LLM] Final image base64 size: {final_size_mb:.2f}MB")
            return best_result
        else:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"[LLF-LLM] Error encoding image: {str(e)}")
        i = 255.0 * ref_image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
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

    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path

    return None


def encode_video_b64(video):
    """Encode ComfyUI VIDEO object to base64 MP4 bytes with compression."""
    video_path = _get_video_file_path(video)
    temp_original = None

    if not video_path:
        if hasattr(video, "save_to"):
            temp_original = f"temp_video_original_{time.time()}.mp4"
            try:
                video.save_to(temp_original)
                video_path = temp_original
            except Exception as e:
                print(f"[LLF-LLM] Error saving video: {str(e)}")
                raise ValueError(f"Unable to save video: {str(e)}")
        else:
            raise ValueError(f"Unable to read video data from object type: {type(video)}")

    # Get original video info
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            if 'streams' in probe_data and len(probe_data['streams']) > 0:
                stream = probe_data['streams'][0]
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                duration = float(stream.get('duration', 0))
                print(f"[LLF-LLM] Original video: {width}x{height}, {duration:.1f}s")
    except Exception as e:
        print(f"[LLF-LLM] Could not probe video: {e}")

    try:
        original_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"[LLF-LLM] Original video file size: {original_size_mb:.2f}MB")
    except Exception:
        original_size_mb = 0

    # Compress video using ffmpeg
    compressed_path = f"temp_video_compressed_{time.time()}.mp4"

    try:
        compress_cmd = [
            'ffmpeg', '-i', video_path,
            '-t', '5',
            '-vf', 'scale=\'min(1280,iw)\':-2',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '30',
            '-b:v', '400k',
            '-maxrate', '400k',
            '-bufsize', '800k',
            '-r', '10',
            '-an',
            '-y',
            compressed_path
        ]

        print(f"[LLF-LLM] Compressing video (first 5s only) with ffmpeg...")
        result = subprocess.run(compress_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"[LLF-LLM] FFmpeg compression failed: {result.stderr}")
            final_path = video_path
        else:
            compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
            print(f"[LLF-LLM] Compressed video size: {compressed_size_mb:.2f}MB "
                  f"(reduced {((original_size_mb - compressed_size_mb) / original_size_mb * 100):.1f}%)")
            final_path = compressed_path

    except FileNotFoundError:
        print(f"[LLF-LLM] Warning: ffmpeg not found, using original video without compression")
        final_path = video_path
    except subprocess.TimeoutExpired:
        print(f"[LLF-LLM] Warning: ffmpeg timeout, using original video")
        final_path = video_path
    except Exception as e:
        print(f"[LLF-LLM] Warning: compression failed ({str(e)}), using original video")
        final_path = video_path

    try:
        with open(final_path, "rb") as f:
            video_bytes = f.read()
            base64_data = base64.b64encode(video_bytes).decode("utf-8")

        base64_size_mb = len(base64_data) / (1024 * 1024)
        print(f"[LLF-LLM] Final video base64 size: {base64_size_mb:.2f}MB")

        if base64_size_mb > 10.0:
            print(f"[LLF-LLM] Warning: Base64 size is very large ({base64_size_mb:.2f}MB), may cause API issues")

        return base64_data

    finally:
        try:
            if temp_original and os.path.exists(temp_original):
                os.remove(temp_original)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLFaigcLLM:
    """OpenAI-compatible LLM API node. Supports text, up to 8 images, and video inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": False, "default": "https://ai.t8star.cn/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (["自定义..."] + CHAT_MODELS,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "video": ("VIDEO",),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "skip_error": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run_llm"
    CATEGORY = "LLFaigc/LLM"

    def run_llm(self, api_baseurl, api_key, model, prompt,
                custom_model="",
                image_1=None, image_2=None, image_3=None, image_4=None,
                image_5=None, image_6=None, image_7=None, image_8=None,
                video=None, temperature=0.6, skip_error=False):

        # Resolve model name
        if model == "自定义...":
            if not custom_model.strip():
                error_msg = "Model is set to '自定义...' but custom_model is empty"
                print(f"[LLF-LLM] {error_msg}")
                if not skip_error:
                    raise ValueError(error_msg)
                return (error_msg,)
            resolved_model = custom_model.strip()
        else:
            resolved_model = model

        print(f"[LLF-LLM] Using model: {resolved_model}")

        if OpenAI is None:
            if not skip_error:
                raise RuntimeError(
                    "[LLFaigcLLM] Error: openai package not installed. "
                    "Please install it with: pip install openai"
                )
            return ("Error: openai package not installed. Please install it with: pip install openai",)

        client = OpenAI(api_key=api_key, base_url=api_baseurl)

        # Collect all images
        images = []
        for i, img in enumerate([image_1, image_2, image_3, image_4,
                                  image_5, image_6, image_7, image_8], 1):
            if img is not None:
                images.append((i, img))
        print(f"[LLF-LLM] Total images provided: {len(images)}")

        # Build messages
        try:
            # Build user content parts
            user_content = []

            # Add text prompt first
            if prompt.strip():
                user_content.append({"type": "text", "text": prompt.strip()})

            # Priority: video > images
            if video is not None:
                print(f"[LLF-LLM] Processing video input...")
                base64_video = encode_video_b64(video)
                print(f"[LLF-LLM] Video base64 size: {len(base64_video) / (1024*1024):.2f}MB")
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
                })
            elif images:
                for idx, img in images:
                    print(f"[LLF-LLM] Encoding image_{idx}...")
                    b64 = encode_image_b64(img)
                    print(f"[LLF-LLM] Image_{idx} base64 size: {len(b64) / (1024*1024):.2f}MB")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

            # If no multimodal content, use plain text
            if len(user_content) == 1 and user_content[0]["type"] == "text":
                messages = [
                    {'role': 'user', 'content': user_content[0]["text"]},
                ]
            else:
                messages = [
                    {'role': 'user', 'content': user_content},
                ]

            # Debug log (strip base64)
            debug_messages = copy.deepcopy(messages)
            for msg in debug_messages:
                if isinstance(msg.get('content'), list):
                    for item in msg['content']:
                        if item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            item['image_url']['url'] = url[:60] + f"...[{len(url)} chars]"
                        elif item.get('type') == 'video_url':
                            url = item['video_url']['url']
                            item['video_url']['url'] = url[:60] + f"...[{len(url)} chars]"
            print(f"[LLF-LLM] Request messages: {json.dumps(debug_messages, indent=2, ensure_ascii=False)}")

        except Exception as e:
            error_msg = f"Error encoding inputs: {str(e)}"
            print(f"[LLF-LLM] {error_msg}")
            import traceback
            traceback.print_exc()
            if not skip_error:
                raise
            return (error_msg,)

        # Call API
        try:
            print(f"[LLF-LLM] Calling API: {api_baseurl} | Model: {resolved_model}")
            completion = client.chat.completions.create(
                model=resolved_model, messages=messages, temperature=temperature
            )

            if completion is not None and hasattr(completion, 'choices'):
                result = completion.choices[0].message.content
                print(f"[LLF-LLM] Response received: {len(result)} chars")
            else:
                result = 'Error: No response from API'
                print(f"[LLF-LLM] {result}")

        except Exception as e:
            result = f'Error calling API: {str(e)}'
            print(f"[LLF-LLM] API call failed: {str(e)}")
            import traceback
            traceback.print_exc()
            if not skip_error:
                raise

        return (result,)
