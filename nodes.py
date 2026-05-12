import json
import os

from .model_catalog import MODEL_CATALOG, RUNNINGHUB_API_DOC, RUNNINGHUB_PRICE_DOC
from .runninghub_client import RunningHubClient, RunningHubError


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "models_registry.json")
LIST_OMIT_SENTINEL = "empty"


def _load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _pretty_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


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


def _param_input(param):
    field_type = param.get("type", "STRING")
    field_key = param.get("fieldKey", "")

    if field_type in {"IMAGE", "VIDEO", "AUDIO"}:
        return ("STRING", {"multiline": False, "default": ""})

    if field_type == "LIST":
        options = [str(option["value"]) for option in param.get("options", [])]
        if not options:
            return ("STRING", {"multiline": False, "default": ""})
        seen = set()
        unique = []
        for option in options:
            lowered = option.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(option)
        default = str(param.get("defaultValue", unique[0]))
        if default not in unique:
            default = unique[0]
        return (tuple(unique), {"default": default})

    if field_type == "INT":
        config = {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}
        if param.get("min") is not None:
            config["min"] = int(param["min"])
        if param.get("max") is not None:
            config["max"] = int(param["max"])
        if param.get("step") is not None:
            config["step"] = int(param["step"])
        if param.get("defaultValue") is not None:
            config["default"] = int(param["defaultValue"])
        return ("INT", config)

    if field_type == "FLOAT":
        config = {"default": 0.0, "min": -1.0e12, "max": 1.0e12, "step": 0.01}
        if param.get("min") is not None:
            config["min"] = float(param["min"])
        if param.get("max") is not None:
            config["max"] = float(param["max"])
        if param.get("step") is not None:
            config["step"] = float(param["step"])
        if param.get("defaultValue") is not None:
            config["default"] = float(param["defaultValue"])
        return ("FLOAT", config)

    if field_type == "BOOLEAN":
        default = param.get("defaultValue", False)
        if isinstance(default, str):
            default = default.lower() in {"true", "1", "yes"}
        return ("BOOLEAN", {"default": bool(default)})

    multiline = field_key.lower() in {"prompt", "text", "negativeprompt", "negative_prompt"}
    return ("STRING", {"multiline": multiline, "default": str(param.get("defaultValue") or "")})


def _coerce_value(param, value):
    field_type = param.get("type", "STRING")
    field_key = param.get("fieldKey", "")

    if value is None:
        return None

    if field_type == "LIST" and str(value) == LIST_OMIT_SENTINEL:
        return None

    if field_type in {"IMAGE", "VIDEO", "AUDIO"}:
        if param.get("multipleInputs") or field_key.endswith("Urls") or field_key == "videos":
            items = [item.strip() for item in str(value).replace(",", "\n").splitlines()]
            return [item for item in items if item]
        return str(value).strip() or None

    if field_type == "INT":
        return int(value)
    if field_type == "FLOAT":
        return float(value)
    if field_type == "BOOLEAN":
        return bool(value)
    if field_type == "STRING":
        text = str(value).strip()
        return text or None
    return str(value)


def _build_payload(model_def, kwargs):
    payload = {}
    for param in model_def.get("params", []):
        field_key = param.get("fieldKey")
        if not field_key:
            continue
        value = _coerce_value(param, kwargs.get(field_key))
        if value is not None:
            payload[field_key] = value
    raw_json = kwargs.get("raw_json") or ""
    if raw_json.strip():
        try:
            extra = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RunningHubError(f"raw_json 不是合法 JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise RunningHubError("raw_json 必须是 JSON object。")
        payload.update(extra)
    return payload


def _create_node_class(model_def):
    params = list(model_def.get("params", []))
    class_name = model_def.get("class_name", "LLFRunningHubGeneratedNode")
    endpoint = model_def["endpoint"]
    category = model_def.get("category", "LLFaigc/RunningHub")

    required_params = {}
    optional_params = {}
    for param in params:
        field_key = param.get("fieldKey")
        if not field_key:
            continue
        target = required_params if param.get("required") else optional_params
        target[field_key] = _param_input(param)

    class GeneratedRunningHubNode:
        CATEGORY = category
        FUNCTION = "execute"
        RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
        RETURN_NAMES = ("task_id", "urls", "response_json", "payload_json")
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):
            required = {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                **required_params,
            }
            optional = {
                **optional_params,
                "base_url": ("STRING", {"multiline": False, "default": "https://www.runninghub.cn"}),
                "wait_for_result": ("BOOLEAN", {"default": True}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "max_wait": ("INT", {"default": 600, "min": 30, "max": 7200, "step": 30}),
                "raw_json": ("STRING", {"multiline": True, "default": ""}),
                "skip_error": ("BOOLEAN", {"default": False}),
            }
            return {"required": required, "optional": optional}

        def execute(self, api_key, **kwargs):
            skip_error = bool(kwargs.pop("skip_error", False))
            try:
                return self._execute(api_key, **kwargs)
            except Exception as exc:
                if not skip_error:
                    raise
                response = {"error": str(exc), "endpoint": endpoint}
                return ("", "", _pretty_json(response), "{}")

        def _execute(self, api_key, **kwargs):
            base_url = kwargs.pop("base_url", "https://www.runninghub.cn")
            wait_for_result = bool(kwargs.pop("wait_for_result", True))
            poll_interval = int(kwargs.pop("poll_interval", 5))
            max_wait = int(kwargs.pop("max_wait", 600))
            payload = _build_payload(model_def, kwargs)

            client = RunningHubClient(api_key=api_key, base_url=base_url, timeout=max_wait)
            task_id, submit_response = client.submit_openapi(endpoint, payload)

            final_response = submit_response
            if wait_for_result:
                final_response = client.poll_openapi(task_id, poll_interval, max_wait)

            return (
                task_id,
                _extract_urls(final_response),
                _pretty_json(final_response),
                _pretty_json(payload),
            )

    GeneratedRunningHubNode.__name__ = class_name
    GeneratedRunningHubNode.__qualname__ = class_name
    return GeneratedRunningHubNode


class LLFRunningHubModelInfo:
    @classmethod
    def INPUT_TYPES(cls):
        labels = tuple(f"{item['llf_branch']} / {item['llf_series']} / {item['display_name']}" for item in _load_registry())
        return {"required": {"model": (labels,)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_name", "endpoint", "price", "docs")
    FUNCTION = "info"
    CATEGORY = "LLFaigc/RunningHub/模型信息"

    def info(self, model):
        registry = _load_registry()
        selected = registry[0]
        for item in registry:
            label = f"{item['llf_branch']} / {item['llf_series']} / {item['display_name']}"
            if label == model:
                selected = item
                break
        series_price = ""
        for value in MODEL_CATALOG.values():
            if value["label"] == selected.get("llf_series"):
                series_price = value["price"]
                break
        docs = f"API: {RUNNINGHUB_API_DOC}\n价格: {RUNNINGHUB_PRICE_DOC}"
        return (selected["display_name"], selected["endpoint"], series_price or "以 RunningHub 模型API价格页为准", docs)


def create_all_nodes():
    class_mappings = {"LLFRunningHubModelInfo": LLFRunningHubModelInfo}
    display_mappings = {"LLFRunningHubModelInfo": "LLFaigc RunningHub 模型信息"}
    for model_def in _load_registry():
        internal_name = "LLF_" + model_def.get("class_name", model_def["endpoint"]).replace("-", "_").replace("/", "_")
        class_mappings[internal_name] = _create_node_class(model_def)
        display_mappings[internal_name] = model_def["display_name"].replace("RH ", "LLF ")
    print(f"[LLFaigc] Registered {len(class_mappings) - 1} RunningHub model nodes")
    return class_mappings, display_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = create_all_nodes()
