"""
LLFaigc - GPT-Image-2 Official node (ported from Comfyui-zhenzhen).

Self-contained: no dependency on zhenzhen's Comfly.py, utils.py, or Comflyapi.json.
API key is provided via the node's ``api_key`` string input.
"""

import io
import math
import re
import time

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


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLFaigcGptImage2Official:
    """GPT-Image-2 official API node (zhenzhen-compatible, self-contained)."""

    _ASPECT_RATIO_CHOICES = [
        "auto",
        "1:1", "3:2", "2:3",
        "4:3", "3:4",
        "5:4", "4:5",
        "16:9", "9:16",
        "2:1", "1:2",
        "21:9", "9:21",
    ]

    _RESOLUTION_CHOICES = ["1k", "2k", "4k"]

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
                "prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (cls._ASPECT_RATIO_CHOICES, {"default": "auto"}),
                "resolution": (cls._RESOLUTION_CHOICES, {"default": "1k"}),
            },
            "optional": {
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
                "model": (["gpt-image-2", "gpt-image-2-all"], {"default": "gpt-image-2"}),
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "LLFaigc/\u56fe\u50cf\u751f\u6210"

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
        import base64
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
        pbar,
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

        pbar.update_absolute(10)
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

        print(f"Task submitted. Task ID: {task_id}")
        pbar.update_absolute(20)

        query_url = f"{_BASE_URL}/v1/images/tasks/{task_id}"
        final_result = None
        image_url_first = ""

        for attempts in range(1, max_poll_attempts + 1):
            time.sleep(poll_interval)
            try:
                status_response = requests.get(
                    query_url, headers=self.get_headers_multipart(), timeout=self.timeout
                )
                if status_response.status_code != 200:
                    print(f"Status check failed: {status_response.status_code}")
                    continue
                status_data = status_response.json()
                inner = status_data.get("data", {}) if isinstance(status_data, dict) else {}
                status = inner.get("status", "")
                progress_str = inner.get("progress", "0%")
                try:
                    if isinstance(progress_str, str) and progress_str.endswith("%"):
                        progress_value = int(progress_str[:-1])
                        pbar_value = min(95, 20 + int(progress_value * 0.75))
                        pbar.update_absolute(pbar_value)
                except (ValueError, AttributeError):
                    pass

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
                    pbar.update_absolute(100)
                    return (combined, image_url_first, task_id, final_result)
                if status == "FAILURE":
                    fail_reason = inner.get("fail_reason", "Unknown error")
                    raise RuntimeError(f"Task failed: {fail_reason}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"Error polling task status: {str(e)}")
        raise RuntimeError(f"Failed to get image after {max_poll_attempts} poll attempts")

    def _items_to_tensors(self, result, max_retries=5, initial_timeout=300):
        import base64
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
        max_retries, initial_timeout, pbar,
    ):
        data, request_files = self._build_official_edits_multipart(
            prompt, image1, image2, image3, image4, image5,
            image6, image7, image8, image9, image10,
            image11, image12, image13, image14, image15, image16,
            mask, n, quality, size, background,
            output_format, output_compression, moderation, response_format, model,
        )
        pbar.update_absolute(20)
        response = self.make_request_with_retry(
            f"{_BASE_URL}/v1/images/edits",
            data=data,
            files=request_files,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
        )
        pbar.update_absolute(60)
        return response.json()

    def generate(
        self, prompt, aspect_ratio="1:1", resolution="1k",
        image1=None, image2=None, image3=None, image4=None, image5=None,
        image6=None, image7=None, image8=None, image9=None, image10=None,
        image11=None, image12=None, image13=None, image14=None, image15=None, image16=None,
        mask=None, api_key="", model="gpt-image-2",
        n=1, quality="auto", background="auto",
        output_format="png", output_compression=100, moderation="auto",
        response_format="url",
        async_mode=True, webhook="", max_poll_attempts=300, poll_interval=5,
        max_retries=5, initial_timeout=900, seed=0, skip_error=False,
    ):
        if api_key.strip():
            self.api_key = api_key

        blank = Image.new("RGB", (1024, 1024), color="white")
        blank_t = _pil2tensor(blank)

        if not self.api_key:
            msg = "API key not provided"
            print(msg)
            return (blank_t, "", msg)

        size, error_msg = self._get_size_from_params(aspect_ratio, resolution)
        if error_msg:
            print(error_msg)
            return (blank_t, "", error_msg)

        input_images = [
            img for img in [image1, image2, image3, image4, image5,
                            image6, image7, image8, image9, image10,
                            image11, image12, image13, image14, image15, image16]
            if img is not None
        ]
        num_input_images = len(input_images)

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        def _info_common(mode_line):
            s = f"**LLFaigc GPT-Image-2 (official)** {mode_line}\n"
            s += f"Model: {model}\n"
            s += f"Prompt: {prompt}\n"
            s += f"Aspect Ratio: {aspect_ratio}\n"
            s += f"Resolution: {resolution}\n"
            s += f"Actual Size: {size}\n"
            s += f"Quality: {quality}\n"
            s += f"Input Images: {num_input_images}\n"
            w, h = self._parse_size_wh(size)
            if w is not None and h is not None and w * h > 2560 * 1440:
                s += "(experimental output: total pixels > 2560x1440)\n"
            if background != "auto":
                s += f"Background: {background}\n"
            s += f"Output: {output_format}\n"
            return s

        try:
            ok, err_msg = self._validate_gpt_image2_size(size)
            if not ok:
                print(err_msg)
                return (blank_t, "", err_msg)

            if async_mode:
                combined, image_url, task_id, final_result = self._async_official(
                    prompt,
                    image1, image2, image3, image4, image5,
                    image6, image7, image8, image9, image10,
                    image11, image12, image13, image14, image15, image16,
                    mask,
                    pbar, max_poll_attempts, poll_interval, webhook,
                    n, quality, size, background,
                    output_format, output_compression, moderation,
                    response_format, model, max_retries, initial_timeout,
                )
                mode = "async: POST /v1/images/edits?async=true, GET /v1/images/tasks/{task_id}"
                info = _info_common(mode)
                info += f"Task ID: {task_id}\n"
                if image_url:
                    info += f"Image URL: {image_url}\n"
                if final_result:
                    inner = final_result.get("data", {})
                    inner_data = inner.get("data", {}) if isinstance(inner, dict) else {}
                    if isinstance(inner_data, dict) and "usage" in inner_data:
                        usage = inner_data["usage"]
                        info += f"Total Tokens: {usage.get('total_tokens', 'N/A')}\n"
                return (combined, image_url or "", info)

            result = self._edits(
                prompt, image1, image2, image3, image4, image5,
                image6, image7, image8, image9, image10,
                image11, image12, image13, image14, image15, image16,
                mask, n, quality, size, background,
                output_format, output_compression, moderation, response_format, model,
                max_retries, initial_timeout, pbar,
            )
            mode = "sync: /v1/images/edits (multipart" + (
                ", blank ref" if num_input_images == 0 else f", {num_input_images} images"
            ) + (", mask" if mask is not None else "") + ")"

            if "data" not in result or not result["data"]:
                msg = f"No image data in response: {result}"
                print(msg)
                return (blank_t, "", msg)

            tensors = self._items_to_tensors(result, max_retries, initial_timeout)
            pbar.update_absolute(95)

            if not tensors:
                msg = "No images decoded from response"
                print(msg)
                return (blank_t, "", msg)

            combined = torch.cat(tensors, dim=0)
            pbar.update_absolute(100)

            info = _info_common(mode)
            if "usage" in result:
                u = result["usage"]
                if isinstance(u, dict):
                    if "total_tokens" in u:
                        info += f"Total tokens: {u['total_tokens']}\n"
                    if "input_tokens" in u:
                        info += f"Input tokens: {u['input_tokens']}\n"
                    if "output_tokens" in u:
                        info += f"Output tokens: {u['output_tokens']}\n"

            return (combined, "", info)

        except Exception as e:
            error_message = f"LLFaigc GPT-Image-2 error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            print(error_message)
            if not skip_error:
                raise
            return (blank_t, "", error_message)
