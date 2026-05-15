"""
LLFaigc - GPT-Image-2 node (Official + FAL mode in one node).

Self-contained: no dependency on zhenzhen's Comfly.py, utils.py, or Comflyapi.json.
API key is provided via the node's ``api_key`` string input.

Supports two API modes:
  - official: multipart form POST to /v1/images/edits (sync or async)
  - fal: JSON POST to /fal/openai/gpt-image-2 with queue polling

Supports multi-prompt batch mode: one prompt per line, each line produces
images that are concatenated into a single output batch.
"""

import base64
import io
import math
import re
import time

import comfy
import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO
from comfy.utils import common_upscale

# ---------------------------------------------------------------------------
# Helpers (self-contained, no external plugin deps)
# ---------------------------------------------------------------------------

_BASE_URL = "https://ai.t8star.cn"


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


def _downscale_input(image):
    """Downscale image so long-edge * short-edge <= 1536*1024."""
    samples = image.movedim(-1, 1)
    total = int(1536 * 1024)
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = common_upscale(samples, width, height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s


def _split_prompts(text):
    """Split text into individual prompts using double-newline as delimiter.

    Two consecutive newlines (blank line) separates prompts.
    Single newlines within a prompt are preserved.
    Leading/trailing whitespace is stripped from each prompt.
    Empty prompts are skipped.
    """
    if not text or not text.strip():
        return []
    # Normalize line endings, then split on double-newline
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\n+', normalized)
    return [block.strip() for block in blocks if block.strip()]


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLFaigcGptImage2Official:
    """GPT-Image-2 unified node (Official + FAL mode).

    Supports multi-prompt batch: fill in multiple prompts (one per line),
    and the node will call the API for each prompt and combine all results
    into a single IMAGE batch output.

    Model selection auto-switches API mode:
      - "gpt-image-2" / "gpt-image-2-all": Official (multipart form)
      - "gpt-image-2-fal": FAL (JSON + queue polling)
    """

    _ASPECT_RATIO_CHOICES = [
        "auto",
        "1:1", "3:2", "2:3",
        "4:3", "3:4",
        "5:4", "4:5",
        "16:9", "9:16",
        "2:1", "1:2",
        "21:9", "9:21",
    ]

    _RESOLUTION_CHOICES = ["auto", "1k", "2k", "4k"]

    _MODEL_CHOICES = ["gpt-image-2", "gpt-image-2-all", "gpt-image-2-fal"]

    _SIZE_MAP = {
        # 1:1
        ("1:1", "1k"): "1024x1024", ("1:1", "2k"): "2048x2048", ("1:1", "4k"): "2880x2880",
        # 16:9
        ("16:9", "1k"): "1280x720", ("16:9", "2k"): "2560x1440", ("16:9", "4k"): "3840x2160",
        # 9:16
        ("9:16", "1k"): "720x1280", ("9:16", "2k"): "1440x2560", ("9:16", "4k"): "2160x3840",
        # 4:3
        ("4:3", "1k"): "1152x864", ("4:3", "2k"): "2304x1728", ("4:3", "4k"): "3264x2448",
        # 3:4
        ("3:4", "1k"): "864x1152", ("3:4", "2k"): "1728x2304", ("3:4", "4k"): "2448x3264",
        # 3:2
        ("3:2", "1k"): "1248x832", ("3:2", "2k"): "2496x1664", ("3:2", "4k"): "3504x2336",
        # 2:3
        ("2:3", "1k"): "832x1248", ("2:3", "2k"): "1664x2496", ("2:3", "4k"): "2336x3504",
        # 5:4
        ("5:4", "1k"): "1120x896", ("5:4", "2k"): "2240x1792", ("5:4", "4k"): "3200x2560",
        # 4:5
        ("4:5", "1k"): "896x1120", ("4:5", "2k"): "1792x2240", ("4:5", "4k"): "2560x3200",
        # 21:9
        ("21:9", "1k"): "1456x624", ("21:9", "2k"): "3024x1296", ("21:9", "4k"): "3696x1584",
        # 9:21
        ("9:21", "1k"): "624x1456", ("9:21", "2k"): "1296x3024", ("9:21", "4k"): "1584x3696",
        # 2:1
        ("2:1", "1k"): "2048x1024", ("2:1", "2k"): "2688x1344", ("2:1", "4k"): "3840x1920",
        # 1:2
        ("1:2", "1k"): "1024x2048", ("1:2", "2k"): "1344x2688", ("1:2", "4k"): "1920x3840",
    }

    # FAL image_size presets
    _FAL_IMAGE_SIZE_CHOICES = [
        "auto",
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9",
        "custom",
    ]

    @staticmethod
    def _parse_size_wh(size_str):
        m = re.match(r"^(\d+)x(\d+)$", size_str.strip())
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    @classmethod
    def _validate_gpt_image2_size(cls, size_str):
        if size_str == "auto":
            return True, None
        w, h = cls._parse_size_wh(size_str)
        if w is None:
            return False, "size format must be WxH, e.g. 1024x1024"
        if max(w, h) > 3840:
            return False, "long edge must be <= 3840px"
        lo, hi = min(w, h), max(w, h)
        if hi / lo > 3.0 + 1e-9:
            return False, "aspect ratio must not exceed 3:1"
        px = w * h
        if px < 655360 or px > 8294400:
            return False, "total pixels must be 655,360 ~ 8,294,400"
        return True, None

    @classmethod
    def _get_size_from_params(cls, aspect_ratio, resolution):
        if aspect_ratio == "auto":
            return "auto", None
        size = cls._SIZE_MAP.get((aspect_ratio, resolution))
        if size is None:
            return None, f"unsupported combo: {aspect_ratio} x {resolution}"
        return size, None

    # ----- ComfyUI interface -----

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls._MODEL_CHOICES, {"default": "gpt-image-2"}),
                "aspect_ratio": (cls._ASPECT_RATIO_CHOICES, {"default": "auto"}),
                "resolution": (cls._RESOLUTION_CHOICES, {"default": "1k"}),
            },
            "optional": {
                "prompts": ("STRING", {"multiline": True, "default": "", "tooltip": "Separate prompts with a blank line (double Enter). Single newlines within a prompt are preserved."}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "image15": ("IMAGE",),
                "image16": ("IMAGE",),
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "background": (["auto", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "async_mode": ("BOOLEAN", {"default": True}),
                "webhook": ("STRING", {"default": ""}),
                "max_poll_attempts": ("INT", {"default": 300, "min": 10, "max": 1000}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 60}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
                "initial_timeout": ("INT", {"default": 900, "min": 60, "max": 1200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "skip_error": ("BOOLEAN", {"default": False}),
                # FAL-specific params
                "fal_mode": (["generate", "edit"], {"default": "edit"}),
                "fal_image_size": (cls._FAL_IMAGE_SIZE_CHOICES, {"default": "custom"}),
                "fal_custom_width": ("INT", {"default": 3840, "min": 256, "max": 3840, "step": 16}),
                "fal_custom_height": ("INT", {"default": 2160, "min": 256, "max": 3840, "step": 16}),
                "fal_num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "fal_image_way": (["image_url", "base64"], {"default": "image_url"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "LLFaigc/个人节点"

    def __init__(self):
        self.api_key = ""
        self.timeout = 300
        self.session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _auth_headers_bearer(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers_json(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def make_request_with_retry(self, url, data=None, files=None, max_retries=5, initial_timeout=300):
        for attempt in range(1, max_retries + 1):
            current_timeout = min(initial_timeout * (1.5 ** (attempt - 1)), 1200)
            try:
                if files is not None:
                    response = self.session.post(
                        url,
                        headers=self._auth_headers_bearer(),
                        data=data,
                        files=files,
                        timeout=current_timeout,
                    )
                else:
                    response = self.session.post(
                        url,
                        headers=self._headers_json(),
                        json=data,
                        timeout=current_timeout,
                    )
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except requests.exceptions.ConnectionError:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (400, 401, 403):
                    raise
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 60))

    def get_headers_multipart(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _blank_input_file(self):
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1024), color="white").save(buf, format="PNG")
        buf.seek(0)
        return ("blank.png", buf, "image/png")

    # ===================================================================
    # FAL mode helpers
    # ===================================================================

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
                f"{_BASE_URL}/v1/files",
                headers=headers,
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            if "url" in result:
                return result["url"]
            print(f"  [FAL] Unexpected file upload response: {result}")
            return None
        except Exception as e:
            print(f"  [FAL] Error uploading image: {e}")
            return None

    def _generate_single_fal(
        self, prompt,
        image1, image2, image3, image4, image5,
        image6, image7, image8, image9, image10,
        image11, image12, image13, image14, image15, image16,
        mask,
        fal_mode, fal_image_size, fal_custom_width, fal_custom_height,
        fal_num_images, fal_image_way,
        poll_interval, max_poll_attempts, seed,
        quality, output_format,
    ):
        """FAL mode: JSON payload -> queue polling."""
        fal_base = f"{_BASE_URL}/fal"
        if fal_mode == "edit":
            endpoint = "openai/gpt-image-2/edit"
        else:
            endpoint = "openai/gpt-image-2"
        api_url = f"{fal_base}/{endpoint}"

        # Build JSON payload
        payload = {
            "prompt": prompt,
            "quality": quality,
            "num_images": fal_num_images,
            "output_format": output_format,
        }

        if fal_image_size == "custom":
            w = (fal_custom_width // 16) * 16
            h = (fal_custom_height // 16) * 16
            payload["image_size"] = {"width": w, "height": h}
        elif fal_image_size != "auto" or fal_mode == "generate":
            payload["image_size"] = fal_image_size

        if seed > 0:
            payload["seed"] = seed

        # Collect input images (FAL uses up to 4 images)
        all_images = [image1, image2, image3, image4]
        input_images = [img for img in all_images if img is not None]

        image_urls = []
        if input_images:
            for idx, img in enumerate(input_images):
                if fal_image_way == "base64":
                    img_data = _image_to_base64(img)
                else:
                    img_data = self._upload_image_to_url(img)
                if img_data:
                    image_urls.append(img_data)
                    print(f"  [FAL] Image {idx+1}/{len(input_images)} prepared")

        if image_urls:
            payload["image_urls"] = image_urls

        # Process mask
        if mask is not None:
            if fal_image_way == "base64":
                mask_data = _image_to_base64(mask)
            else:
                mask_data = self._upload_image_to_url(mask)
            if mask_data:
                payload["mask_image_url"] = mask_data
                print(f"  [FAL] Mask image prepared")

        print(f"  [FAL] Submitting to {api_url} (mode={fal_mode}, quality={quality}, size={fal_image_size})")

        # Submit request
        response = requests.post(
            api_url,
            headers=self._headers_json(),
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"[FAL] API Error: {response.status_code} - {response.text[:500]}")

        result = response.json()

        # Check if result is immediate
        if "images" in result and result["images"]:
            result_data = result
        else:
            # Queue mode: poll for result
            request_id = result.get("request_id")
            response_url = result.get("response_url", "")

            if not request_id:
                raise RuntimeError(f"[FAL] No request_id in response: {str(result)[:300]}")

            # Fix response_url to use our proxy
            if "queue.fal.run" in response_url:
                response_url = response_url.replace("https://queue.fal.run", fal_base)
            if not response_url:
                response_url = f"{fal_base}/{endpoint}/requests/{request_id}"

            print(f"  [FAL] Queued, request_id={request_id}, polling...")

            result_data = None
            for attempt in range(max_poll_attempts):
                time.sleep(poll_interval)
                try:
                    poll_resp = requests.get(
                        response_url,
                        headers=self._headers_json(),
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
                        raise RuntimeError(f"[FAL] Task {status}: {poll_data.get('error', 'unknown')}")

                except requests.exceptions.RequestException as e:
                    print(f"  [FAL] Poll error: {e}")
                    continue

            if result_data is None:
                raise RuntimeError(f"[FAL] Timeout: no result after {max_poll_attempts * poll_interval}s")

        # Download result images
        images_list = result_data.get("images", [])
        if not images_list:
            raise RuntimeError("[FAL] No images in result")

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
            except Exception as e:
                print(f"  [FAL] Error downloading image {i+1}: {e}")

        if not generated_tensors:
            raise RuntimeError("[FAL] Failed to download any result images")

        combined = torch.cat(generated_tensors, dim=0)
        urls_str = "\n".join(url_list)
        info = (
            f"  Mode: fal ({fal_mode})\n"
            f"  Quality: {quality} | Size: {fal_image_size}\n"
            f"  Images: {len(generated_tensors)}"
        )
        return combined, url_list[0] if url_list else "", info

    # ===================================================================
    # Official mode helpers (original logic)
    # ===================================================================

    def _build_official_edits_multipart(
        self, prompt, image1, image2, image3, image4, image5,
        image6, image7, image8, image9, image10,
        image11, image12, image13, image14, image15, image16,
        mask, n, quality, size, background,
        output_format, output_compression, moderation, response_format="url", model="gpt-image-2",
    ):
        input_images = []
        for img in [image1, image2, image3, image4, image5,
                     image6, image7, image8, image9, image10,
                     image11, image12, image13, image14, image15, image16]:
            if img is not None:
                input_images.append(img)

        if mask is not None and len(input_images) == 0:
            raise Exception("Must provide at least one input image when using mask")

        files = {}

        if len(input_images) == 0:
            files["image"] = self._blank_input_file()
            total_images = 1
        else:
            image_list = []
            for img_tensor in input_images:
                batch_size = img_tensor.shape[0]
                for i in range(batch_size):
                    single_image = img_tensor[i: i + 1]
                    scaled_image = _downscale_input(single_image).squeeze()
                    image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="PNG")
                    img_byte_arr.seek(0)
                    image_list.append(("image_{}.png".format(len(image_list)), img_byte_arr, "image/png"))

            total_images = len(image_list)

            if total_images == 1:
                files["image"] = image_list[0]
            else:
                files["image[]"] = image_list

        if mask is not None:
            if total_images != 1:
                raise Exception("Mask requires exactly one input image")
            first_img = input_images[0]
            if mask.shape[1:] != first_img.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            _batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
            scaled_mask = _downscale_input(rgba_mask.unsqueeze(0)).squeeze()
            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_byte_arr = io.BytesIO()
            mask_img.save(mask_byte_arr, format="PNG")
            mask_byte_arr.seek(0)
            files["mask"] = ("mask.png", mask_byte_arr, "image/png")

        data = {
            "prompt": prompt,
            "model": model,
            "n": str(n),
            "quality": quality,
            "moderation": moderation,
            "size": size,
        }
        if background != "auto":
            data["background"] = background
        if output_compression != 100:
            data["output_compression"] = str(output_compression)
        if output_format != "png":
            data["output_format"] = output_format
        if response_format != "url":
            data["response_format"] = response_format

        if "image[]" in files:
            request_files = []
            for file_tuple in files["image[]"]:
                request_files.append(("image", file_tuple))
            if "mask" in files:
                request_files.append(("mask", files["mask"]))
        else:
            request_files = []
            if "image" in files:
                request_files.append(("image", files["image"]))
            if "mask" in files:
                request_files.append(("mask", files["mask"]))

        return data, request_files

    def _decode_b64_url_one(self, b64_json, image_url, max_retries, initial_timeout):
        if b64_json:
            b64_data = b64_json
            if b64_data.startswith("data:image"):
                b64_data = b64_data.split(",", 1)[-1]
            elif b64_data.startswith("data:image/png;base64,"):
                b64_data = b64_data[len("data:image/png;base64,"):]
            image_data = base64.b64decode(b64_data)
            pil_img = Image.open(BytesIO(image_data))
            return _pil2tensor(pil_img)
        if image_url:
            for download_attempt in range(1, max_retries + 1):
                try:
                    img_response = requests.get(
                        image_url,
                        timeout=min(initial_timeout * (1.5 ** (download_attempt - 1)), 900),
                    )
                    img_response.raise_for_status()
                    pil_img = Image.open(BytesIO(img_response.content))
                    return _pil2tensor(pil_img)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    if download_attempt == max_retries:
                        return None
                    time.sleep(min(2 ** (download_attempt - 1), 60))
        return None

    def _async_official(
        self,
        prompt,
        image1, image2, image3, image4, image5,
        image6, image7, image8, image9, image10,
        image11, image12, image13, image14, image15, image16,
        mask,
        max_poll_attempts,
        poll_interval,
        webhook,
        n, quality, size, background,
        output_format, output_compression, moderation,
        response_format, model, max_retries, initial_timeout,
    ):
        data, request_files = self._build_official_edits_multipart(
            prompt, image1, image2, image3, image4, image5,
            image6, image7, image8, image9, image10,
            image11, image12, image13, image14, image15, image16,
            mask, n, quality, size, background,
            output_format, output_compression, moderation, response_format, model,
        )
        url = f"{_BASE_URL}/v1/images/edits?async=true"
        if webhook.strip():
            url += f"&webhook={webhook.strip()}"

        response = requests.post(
            url,
            headers=self.get_headers_multipart(),
            data=data,
            files=request_files,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API Error: {response.status_code} - {response.text}")

        submit_result = response.json()
        task_id = submit_result.get("task_id") or submit_result.get("data")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {submit_result}")

        print(f"  [Prompt] \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\" -> Task ID: {task_id}")

        query_url = f"{_BASE_URL}/v1/images/tasks/{task_id}"
        image_url_first = ""
        final_result = None

        for attempts in range(1, max_poll_attempts + 1):
            time.sleep(poll_interval)
            try:
                status_response = requests.get(
                    query_url, headers=self.get_headers_multipart(), timeout=self.timeout
                )
                if status_response.status_code != 200:
                    print(f"  Status check failed: {status_response.status_code}")
                    continue
                status_data = status_response.json()
                inner = status_data.get("data", {}) if isinstance(status_data, dict) else {}
                status = inner.get("status", "")

                if status == "SUCCESS":
                    result_data = inner.get("data", {})
                    data_array = (
                        result_data.get("data", []) if isinstance(result_data, dict) else []
                    )
                    tensors = []
                    for item in data_array or []:
                        u = item.get("url", "") or ""
                        bj = item.get("b64_json", "") or ""
                        if u and not image_url_first:
                            image_url_first = u
                        t = self._decode_b64_url_one(bj, u, max_retries, initial_timeout)
                        if t is not None:
                            tensors.append(t)
                    if not tensors:
                        raise RuntimeError("Async task SUCCESS but no decodable image in data")
                    final_result = status_data
                    combined = torch.cat(tensors, dim=0)
                    return (combined, image_url_first, task_id, final_result)
                if status == "FAILURE":
                    fail_reason = inner.get("fail_reason", "Unknown error")
                    raise RuntimeError(f"Task failed: {fail_reason}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"  Error polling task status: {str(e)}")
        raise RuntimeError(f"Failed to get image after {max_poll_attempts} poll attempts")

    def _items_to_tensors(self, result, max_retries=5, initial_timeout=300):
        out = []
        for item in result.get("data", []) or []:
            if "b64_json" in item and item["b64_json"]:
                b64_data = item["b64_json"]
                if b64_data.startswith("data:image"):
                    b64_data = b64_data.split(",", 1)[-1]
                elif b64_data.startswith("data:image/png;base64,"):
                    b64_data = b64_data[len("data:image/png;base64,"):]
                image_data = base64.b64decode(b64_data)
                pil_img = Image.open(BytesIO(image_data))
                out.append(_pil2tensor(pil_img))
            elif "url" in item and item["url"]:
                for download_attempt in range(1, max_retries + 1):
                    try:
                        img_response = requests.get(
                            item["url"],
                            timeout=min(initial_timeout * (1.5 ** (download_attempt - 1)), 900),
                        )
                        img_response.raise_for_status()
                        pil_img = Image.open(BytesIO(img_response.content))
                        out.append(_pil2tensor(pil_img))
                        break
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                        if download_attempt == max_retries:
                            break
                        time.sleep(min(2 ** (download_attempt - 1), 60))
        return out

    def _edits(
        self, prompt, image1, image2, image3, image4, image5,
        image6, image7, image8, image9, image10,
        image11, image12, image13, image14, image15, image16,
        mask, n, quality, size, background,
        output_format, output_compression, moderation, response_format, model,
        max_retries, initial_timeout,
    ):
        data, request_files = self._build_official_edits_multipart(
            prompt, image1, image2, image3, image4, image5,
            image6, image7, image8, image9, image10,
            image11, image12, image13, image14, image15, image16,
            mask, n, quality, size, background,
            output_format, output_compression, moderation, response_format, model,
        )
        response = self.make_request_with_retry(
            f"{_BASE_URL}/v1/images/edits",
            data=data,
            files=request_files,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
        )
        return response.json()

    # -----------------------------------------------------------------------
    # Generate images for a single prompt (returns tensor batch)
    # -----------------------------------------------------------------------

    def _generate_single(
        self, prompt, aspect_ratio, resolution,
        image1, image2, image3, image4, image5,
        image6, image7, image8, image9, image10,
        image11, image12, image13, image14, image15, image16,
        mask, model, n, quality, background,
        output_format, output_compression, moderation,
        response_format, async_mode, webhook,
        max_poll_attempts, poll_interval,
        max_retries, initial_timeout,
    ):
        """Call API for a single prompt and return (tensor_batch, image_url, info_str)."""

        size, error_msg = self._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            return None, "", error_msg

        ok, err_msg = self._validate_gpt_image2_size(size)
        if not ok:
            return None, "", err_msg

        try:
            if async_mode:
                combined, image_url, task_id, final_result = self._async_official(
                    prompt,
                    image1, image2, image3, image4, image5,
                    image6, image7, image8, image9, image10,
                    image11, image12, image13, image14, image15, image16,
                    mask,
                    max_poll_attempts, poll_interval, webhook,
                    n, quality, size, background,
                    output_format, output_compression, moderation,
                    response_format, model, max_retries, initial_timeout,
                )

                info_lines = [
                    f"  Mode: async (task {task_id})",
                    f"  Model: {model} | Size: {size} | Quality: {quality}",
                ]
                if final_result:
                    inner = final_result.get("data", {})
                    inner_data = inner.get("data", {}) if isinstance(inner, dict) else {}
                    if isinstance(inner_data, dict) and "usage" in inner_data:
                        info_lines.append(f"  Tokens: {inner_data['usage'].get('total_tokens', 'N/A')}")
                return combined, image_url or "", "\n".join(info_lines)

            result = self._edits(
                prompt, image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
                image11, image12, image13, image14, image15, image16,
                mask, n, quality, size, background,
                output_format, output_compression, moderation, response_format, model,
                max_retries, initial_timeout,
            )

            if "data" not in result or not result["data"]:
                return None, "", f"No image data in response: {result}"

            tensors = self._items_to_tensors(result, max_retries, initial_timeout)
            if not tensors:
                return None, "", "No images decoded from response"

            combined = torch.cat(tensors, dim=0)

            info_lines = [
                f"  Mode: sync | Model: {model} | Size: {size} | Quality: {quality}",
            ]
            if "usage" in result:
                u = result["usage"]
                if isinstance(u, dict):
                    if "total_tokens" in u:
                        info_lines.append(f"  Tokens: {u['total_tokens']}")
            return combined, "", "\n".join(info_lines)

        except Exception as e:
            import traceback
            print(f"  ERROR for prompt \"{prompt[:60]}...\": {e}")
            print(traceback.format_exc())
            return None, "", f"Error: {e}"

    # -----------------------------------------------------------------------
    # Main entry: generate
    # -----------------------------------------------------------------------

    def generate(
        self, prompts, model="gpt-image-2", aspect_ratio="1:1", resolution="1k",
        image1=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
        image11=None, image12=None, image13=None, image14=None, image15=None, image16=None,
        mask=None, api_key="",
        n=1, quality="auto", background="auto",
        output_format="png", output_compression=100, moderation="auto",
        response_format="url",
        async_mode=True, webhook="", max_poll_attempts=300, poll_interval=5,
        max_retries=5, initial_timeout=900, seed=0, skip_error=False,
        fal_mode="edit", fal_image_size="custom", fal_custom_width=3840, fal_custom_height=2160,
        fal_num_images=1, fal_image_way="image_url",
    ):
        if api_key.strip():
            self.api_key = api_key

        # Auto-detect API mode from model name
        is_fal = model == "gpt-image-2-fal"
        api_mode = "fal" if is_fal else "official"

        blank = Image.new("RGB", (1024, 1024), color="white")
        blank_t = _pil2tensor(blank)

        if not self.api_key:
            msg = "API key not provided"
            print(msg)
            return (blank_t, "", msg)

        # Parse multi-line prompts
        prompt_list = _split_prompts(prompts)
        if not prompt_list:
            msg = "No prompts provided. Enter at least one prompt (one per line)."
            print(msg)
            return (blank_t, "", msg)

        total_prompts = len(prompt_list)
        print(f"**LLFaigc GPT-Image-2** Starting batch with {total_prompts} prompt(s) [mode={api_mode}]")

        pbar = comfy.utils.ProgressBar(100)

        all_tensors = []
        all_urls = []
        all_info = []

        for idx, single_prompt in enumerate(prompt_list):
            print(f"[{idx + 1}/{total_prompts}] Generating: \"{single_prompt[:80]}{'...' if len(single_prompt) > 80 else ''}\"")

            if api_mode == "fal":
                # FAL mode
                result_tensor, result_url, result_info = self._generate_single_fal(
                    single_prompt,
                    image1, image2, image3, image4, image5,
                    image6, image7, image8, image9, image10,
                    image11, image12, image13, image14, image15, image16,
                    mask,
                    fal_mode, fal_image_size, fal_custom_width, fal_custom_height,
                    fal_num_images, fal_image_way,
                    poll_interval, max_poll_attempts, seed,
                    quality, output_format,
                )
            else:
                # Official mode
                # Validate size once (same for all prompts)
                size, error_msg = self._get_size_from_params(aspect_ratio, resolution)
                if error_msg:
                    print(error_msg)
                    return (blank_t, "", error_msg)

                result_tensor, result_url, result_info = self._generate_single(
                    single_prompt, aspect_ratio, resolution,
                    image1, image2, image3, image4, image5,
                    image6, image7, image8, image9, image10,
                    image11, image12, image13, image14, image15, image16,
                    mask, model, n, quality, background,
                    output_format, output_compression, moderation,
                    response_format, async_mode, webhook,
                    max_poll_attempts, poll_interval,
                    max_retries, initial_timeout,
                )

            if result_tensor is not None:
                all_tensors.append(result_tensor)
                if result_url:
                    all_urls.append(result_url)
            else:
                print(f"  [!] Prompt #{idx + 1} failed: {result_info}")
                if not skip_error:
                    if all_tensors:
                        pass  # Return what we have so far
                    else:
                        return (blank_t, "", result_info)

            # Update progress bar
            progress = int((idx + 1) / total_prompts * 100)
            pbar.update_absolute(progress)

            # Build info for this prompt
            info_entry = f"[{idx + 1}/{total_prompts}] \"{single_prompt[:60]}{'...' if len(single_prompt) > 60 else ''}\"\n{result_info}"
            all_info.append(info_entry)

        # Combine all results - normalize to same size first
        if not all_tensors:
            msg = "All prompts failed to generate images."
            print(msg)
            return (blank_t, "", "\n".join(all_info) + "\n" + msg)

        # Find the most common size among all tensors
        size_counts = {}
        for t in all_tensors:
            h, w = t.shape[1], t.shape[2]
            size_counts[(h, w)] = size_counts.get((h, w), 0) + t.shape[0]
        target_size = max(size_counts, key=size_counts.get) if size_counts else (1024, 1024)

        # Resize all tensors to target_size
        normalized = []
        for t in all_tensors:
            h, w = t.shape[1], t.shape[2]
            if (h, w) != target_size:
                # [B, H, W, C] -> [B, C, H, W] for common_upscale
                t_chw = t.movedim(-1, 1)
                t_resized = common_upscale(t_chw, target_size[1], target_size[0], "lanczos", "disabled")
                normalized.append(t_resized.movedim(1, -1))
            else:
                normalized.append(t)

        combined = torch.cat(normalized, dim=0)

        summary = (
            f"**LLFaigc GPT-Image-2** Complete\n"
            f"API Mode: {api_mode}\n"
            f"Total prompts: {total_prompts}\n"
            f"Successful: {len(all_tensors)}\n"
            f"Total images generated: {combined.shape[0]}\n"
            f"Model: {model}\n"
            f"Aspect Ratio: {aspect_ratio}\n"
            f"Resolution: {resolution}\n"
            f"Output: {output_format}\n"
            f"{'---' * 10}\n"
            + "\n".join(all_info)
        )

        return (combined, all_urls[0] if all_urls else "", summary)
