import json

from .model_catalog import MODEL_CATALOG, MODEL_KEYS, RUNNINGHUB_API_DOC, RUNNINGHUB_PRICE_DOC
from .runninghub_client import RunningHubClient, RunningHubError


CREATE_ENDPOINTS = (
    "/task/openapi/create",
    "/openapi/task/create",
    "/api/openapi/task/create",
    "/api/task/create",
)

STATUS_ENDPOINTS = (
    "/task/openapi/status",
    "/openapi/task/status",
    "/api/openapi/task/status",
    "/api/task/status",
)


def _pretty_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _merge_json(default_payload, raw_json):
    payload = dict(default_payload)
    raw_json = (raw_json or "").strip()
    if not raw_json:
        return payload
    try:
        user_payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RunningHubError(f"raw_json 不是合法 JSON: {exc}") from exc
    if not isinstance(user_payload, dict):
        raise RunningHubError("raw_json 必须是 JSON object。")
    payload.update(user_payload)
    return payload


def _extract_task_id(result):
    if not isinstance(result, dict):
        return ""
    candidates = [
        result.get("taskId"),
        result.get("task_id"),
        result.get("id"),
        result.get("data", {}).get("taskId") if isinstance(result.get("data"), dict) else None,
        result.get("data", {}).get("task_id") if isinstance(result.get("data"), dict) else None,
        result.get("data", {}).get("id") if isinstance(result.get("data"), dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value)
    return ""


def _extract_urls(value):
    urls = []

    def walk(item):
        if isinstance(item, dict):
            for inner in item.values():
                walk(inner)
        elif isinstance(item, list):
            for inner in item:
                walk(inner)
        elif isinstance(item, str) and item.startswith(("http://", "https://")):
            urls.append(item)

    walk(value)
    return "\n".join(dict.fromkeys(urls))


class LLFRunningHubModelInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_KEYS,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_name", "kind", "price", "docs")
    FUNCTION = "info"
    CATEGORY = "LLFaigc/RunningHub"

    def info(self, model):
        item = MODEL_CATALOG[model]
        docs = f"API: {RUNNINGHUB_API_DOC}\n价格: {RUNNINGHUB_PRICE_DOC}"
        return (item["label"], item["kind"], item["price"], docs)


class LLFRunningHubCall:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (MODEL_KEYS,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (("1:1", "3:4", "4:3", "9:16", "16:9", "21:9"), {"default": "1:1"}),
                "size": ("STRING", {"multiline": False, "default": "1024x1024"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
                "base_url": ("STRING", {"multiline": False, "default": "https://www.runninghub.cn"}),
                "create_endpoint": ("STRING", {"multiline": False, "default": CREATE_ENDPOINTS[0]}),
                "status_endpoint": ("STRING", {"multiline": False, "default": STATUS_ENDPOINTS[0]}),
                "wait_for_result": ("BOOLEAN", {"default": False}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 30}),
            },
            "optional": {
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "video_url": ("STRING", {"multiline": False, "default": ""}),
                "raw_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("task_id", "urls", "response_json", "payload_json")
    FUNCTION = "call"
    CATEGORY = "LLFaigc/RunningHub"

    def call(
        self,
        api_key,
        model,
        prompt,
        negative_prompt,
        aspect_ratio,
        size,
        duration,
        base_url,
        create_endpoint,
        status_endpoint,
        wait_for_result,
        poll_interval,
        max_wait,
        image_url="",
        video_url="",
        raw_json="",
    ):
        catalog_item = MODEL_CATALOG[model]
        payload = _merge_json(catalog_item["default_payload"], raw_json)
        payload.update(
            {
                "model": catalog_item["label"],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "duration": duration,
            }
        )
        if image_url:
            payload["image_url"] = image_url
        if video_url:
            payload["video_url"] = video_url

        client = RunningHubClient(api_key=api_key, base_url=base_url, timeout=max_wait)
        response = client.create_task(create_endpoint, payload)
        task_id = _extract_task_id(response)

        final_response = response
        if wait_for_result and task_id:
            final_response = client.poll_task(status_endpoint, task_id, poll_interval, max_wait)

        return (task_id, _extract_urls(final_response), _pretty_json(final_response), _pretty_json(payload))


NODE_CLASS_MAPPINGS = {
    "LLFRunningHubModelInfo": LLFRunningHubModelInfo,
    "LLFRunningHubCall": LLFRunningHubCall,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLFRunningHubModelInfo": "LLFaigc RunningHub 模型信息",
    "LLFRunningHubCall": "LLFaigc RunningHub API 调用",
}
