"""
LLFaigc - GPT-Image-2 FAL node (ported from Comfyui-zhenzhen).

Self-contained: no dependency on zhenzhen's Comfly.py, utils.py, or get_config().
Uses fal.ai Queue API via https://ai.t8star.cn/fal proxy.
Supports text-to-image and image editing with mask.

API config is provided via the ``api_config`` (RH_OPENAPI_CONFIG) input.
"""

import base64
import io
import time

import comfy
import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO

from ..core.api_key import get_config


# ---------------------------------------------------------------------------
# Helpers (self-contained, no external plugin deps)
# ---------------------------------------------------------------------------

_LOG_PREFIX = "LLFaigc-GPT-Image-2-FAL"


def _pil2tensor(image):
    """Single PIL Image -> tensor [1, H, W, 3] in [0,1]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _tensor2pil(image_tensor):
    """ComfyUI IMAGE tensor -> list of PIL Images."""
    images = []
    batch_size = image_tensor.shape[0]
    for i in range(batch_size):
        img = image_tensor[i]
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    return images


def _image_to_base64(image_tensor):
    """Convert IMAGE tensor to base64 data URI."""
    if image_tensor is None:
        return None
    pil_image = _tensor2pil(image_tensor)[0]
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLFaigcGptImage2Fal:
    """GPT-Image-2 via fal.ai Queue API.

    Uses base_url/fal as proxy for https://queue.fal.run.
    Supports text-to-image (generate) and image editing (edit).

    api_config: connect RH API Config node to provide base_url and api_key.
    """

    _IMAGE_SIZE_CHOICES = [
        "auto",
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "custom",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["generate", "edit"], {"default": "edit"}),
            },
            "optional": {
                "api_config": ("RH_OPENAPI_CONFIG",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "mask": ("IMAGE",),
                "quality": (["high", "medium", "low", "auto"], {"default": "auto"}),
                "image_size": (cls._IMAGE_SIZE_CHOICES, {"default": "custom"}),
                "custom_width": ("INT", {"default": 3840, "min": 256, "max": 3840, "step": 16}),
                "custom_height": ("INT", {"default": 2160, "min": 256, "max": 3840, "step": 16}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "image_way": (["image_url", "base64"], {"default": "image_url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "poll_interval": ("INT", {"default": 2, "min": 1, "max": 10}),
                "max_poll_attempts": ("INT", {"default": 60, "min": 10, "max": 300}),
                "skip_error": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls")
    FUNCTION = "process"
    CATEGORY = "LLFaigc/个人节点"

    def __init__(self):
        self.api_key = ""
        self.base_url = ""
        self.timeout = 300

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _upload_image_to_url(self, image_tensor):
        """Upload IMAGE tensor to files API and return URL."""
        if image_tensor is None:
            return None
        try:
            pil_image = _tensor2pil(image_tensor)[0]
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            file_content = buffered.getvalue()
            files = {"file": ("image.png", file_content, "image/png")}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.base_url}/v1/files",
                headers=headers,
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            if "url" in result:
                return result["url"]
            print(f"[{_LOG_PREFIX}] Unexpected file upload response: {result}")
            return None
        except Exception as e:
            print(f"[{_LOG_PREFIX}] Error uploading image: {e}")
            return None

    def process(
        self,
        prompt,
        mode="edit",
        api_config=None,
        image1=None, image2=None, image3=None, image4=None,
        mask=None,
        quality="auto", image_size="custom",
        custom_width=3840, custom_height=2160,
        num_images=1, output_format="png", image_way="image_url",
        seed=0, poll_interval=2, max_poll_attempts=60,
        skip_error=False,
    ):
        # Resolve api_key and base_url from api_config
        config = get_config(api_config)
        self.api_key = config["api_key"]
        self.base_url = config["base_url"].rstrip("/")

        # Default fallback image
        all_images = [image1, image2, image3, image4]
        default_image = next((img for img in all_images if img is not None), None)
        if default_image is None:
            default_image = _pil2tensor(Image.new("RGB", (1024, 1024), color="white"))

        try:
            if not self.api_key:
                err = "API key not provided. Connect api_config or set in Settings."
                if not skip_error:
                    raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                return (default_image, err, "")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)

            # Determine endpoint
            fal_base = f"{self.base_url}/fal"
            if mode == "edit":
                endpoint = "openai/gpt-image-2/edit"
            else:
                endpoint = "openai/gpt-image-2"
            api_url = f"{fal_base}/{endpoint}"

            # Build payload
            payload = {
                "prompt": prompt,
                "quality": quality,
                "num_images": num_images,
                "output_format": output_format,
            }

            if image_size == "custom":
                w = (custom_width // 16) * 16
                h = (custom_height // 16) * 16
                payload["image_size"] = {"width": w, "height": h}
            elif image_size != "auto" or mode == "generate":
                payload["image_size"] = image_size

            if seed > 0:
                payload["seed"] = seed

            # Process input images
            image_urls = []
            input_images = [img for img in all_images if img is not None]

            if input_images:
                pbar.update_absolute(10)
                for idx, img in enumerate(input_images):
                    if image_way == "base64":
                        img_data = _image_to_base64(img)
                    else:
                        img_data = self._upload_image_to_url(img)
                    if img_data:
                        image_urls.append(img_data)
                        print(f"[{_LOG_PREFIX}] Image {idx+1}/{len(input_images)} prepared")
                    pbar.update_absolute(10 + int((idx + 1) / len(input_images) * 10))

            if image_urls:
                payload["image_urls"] = image_urls

            # Process mask
            if mask is not None:
                if image_way == "base64":
                    mask_data = _image_to_base64(mask)
                else:
                    mask_data = self._upload_image_to_url(mask)
                if mask_data:
                    payload["mask_image_url"] = mask_data
                    print(f"[{_LOG_PREFIX}] Mask image prepared")

            pbar.update_absolute(25)
            print(f"[{_LOG_PREFIX}] Submitting to {api_url} (mode={mode}, quality={quality}, size={image_size})")

            # Submit request
            response = requests.post(
                api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                err = f"API Error: {response.status_code} - {response.text[:500]}"
                if not skip_error:
                    raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                return (default_image, err, "")

            result = response.json()
            pbar.update_absolute(30)

            # Check if result is immediate (sync mode or direct response)
            if "images" in result and result["images"]:
                result_data = result
            else:
                # Queue mode: poll for result
                request_id = result.get("request_id")
                response_url = result.get("response_url", "")

                if not request_id:
                    err = f"No request_id in response: {str(result)[:300]}"
                    if not skip_error:
                        raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                    return (default_image, err, "")

                # Fix response_url to use our proxy
                if "queue.fal.run" in response_url:
                    response_url = response_url.replace("https://queue.fal.run", fal_base)
                if not response_url:
                    response_url = f"{fal_base}/{endpoint}/requests/{request_id}"

                print(f"[{_LOG_PREFIX}] Queued, request_id={request_id}, polling...")

                result_data = None
                for attempt in range(max_poll_attempts):
                    progress = 30 + min(60, int((attempt + 1) / max_poll_attempts * 60))
                    pbar.update_absolute(progress)
                    time.sleep(poll_interval)

                    try:
                        poll_resp = requests.get(
                            response_url,
                            headers=self._get_headers(),
                            timeout=self.timeout,
                        )
                        if poll_resp.status_code != 200:
                            continue
                        poll_data = poll_resp.json()

                        if "images" in poll_data and poll_data["images"]:
                            result_data = poll_data
                            break

                        status = poll_data.get("status", "")
                        if status in ("FAILED", "CANCELLED"):
                            err = f"Task {status}: {poll_data.get('error', 'unknown')}"
                            if not skip_error:
                                raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                            return (default_image, err, "")

                    except requests.exceptions.RequestException as e:
                        print(f"[{_LOG_PREFIX}] Poll error: {e}")
                        continue

                if result_data is None:
                    err = f"Timeout: no result after {max_poll_attempts * poll_interval}s"
                    if not skip_error:
                        raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                    return (default_image, err, "")

            pbar.update_absolute(90)

            # Download result images
            images_list = result_data.get("images", [])
            if not images_list:
                err = "No images in result"
                if not skip_error:
                    raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
                return (default_image, err, "")

            generated_tensors = []
            url_list = []

            for i, img_info in enumerate(images_list):
                img_url = img_info.get("url", "")
                if not img_url:
                    continue
                url_list.append(img_url)
                try:
                    img_resp = requests.get(img_url, timeout=self.timeout)
                    if img_resp.status_code == 200:
                        pil_img = Image.open(BytesIO(img_resp.content))
                        generated_tensors.append(_pil2tensor(pil_img))
                        print(f"[{_LOG_PREFIX}] Downloaded image {i+1}/{len(images_list)}")
                except Exception as e:
                    print(f"[{_LOG_PREFIX}] Error downloading image {i+1}: {e}")

            if generated_tensors:
                combined = torch.cat(generated_tensors, dim=0)
                pbar.update_absolute(100)
                urls_str = "\n".join(url_list)
                info = (
                    f"Model: gpt-image-2 (fal)\n"
                    f"Mode: {mode}\n"
                    f"Quality: {quality}\n"
                    f"Size: {image_size}\n"
                    f"Images: {len(generated_tensors)}"
                )
                return (combined, info, urls_str)

            err = "Failed to download any result images"
            if not skip_error:
                raise RuntimeError(f"[{_LOG_PREFIX}] {err}")
            return (default_image, err, "")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"[{_LOG_PREFIX}] {error_message}")
            import traceback
            traceback.print_exc()
            if not skip_error:
                raise
            return (default_image, error_message, "")
