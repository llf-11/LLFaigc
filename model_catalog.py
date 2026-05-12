RUNNINGHUB_API_DOC = "https://www.runninghub.cn/call-api/api-detail/2004544343584849921?activeTab=api"
RUNNINGHUB_PRICE_DOC = "https://www.runninghub.cn/third-party-fees"


MODEL_CATALOG = {
    "image:g-series": {
        "label": "全能图片G系列",
        "kind": "image",
        "price": "以 RunningHub 模型API价格页为准；公开索引示例：全能图片G-1.5-文生图 4.7 w",
        "default_payload": {
            "model": "全能图片G系列",
            "prompt": "",
            "negative_prompt": "",
            "aspect_ratio": "1:1",
            "size": "1024x1024",
        },
    },
    "image:omni": {
        "label": "全能图片系列",
        "kind": "image",
        "price": "以 RunningHub 模型API价格页为准",
        "default_payload": {
            "model": "全能图片系列",
            "prompt": "",
            "negative_prompt": "",
            "aspect_ratio": "1:1",
            "size": "1024x1024",
        },
    },
    "image:x-series": {
        "label": "全能图片X",
        "kind": "image",
        "price": "以 RunningHub 模型API价格页为准",
        "default_payload": {
            "model": "全能图片X",
            "prompt": "",
            "negative_prompt": "",
            "aspect_ratio": "1:1",
            "size": "1024x1024",
        },
    },
    "video:s-series": {
        "label": "全能视频S系列",
        "kind": "video",
        "price": "以 RunningHub 模型API价格页为准；公开索引示例：全能视频S 9.8 w",
        "default_payload": {
            "model": "全能视频S系列",
            "prompt": "",
            "image_url": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
    },
    "video:kling": {
        "label": "可灵系列",
        "kind": "video",
        "price": "以 RunningHub 模型API价格页为准；公开索引示例：可灵文生视频o3-pro 38.5 w",
        "default_payload": {
            "model": "可灵系列",
            "prompt": "",
            "image_url": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
    },
    "video:wan": {
        "label": "Wan Video Models系列",
        "kind": "video",
        "price": "以 RunningHub 模型API价格页为准",
        "default_payload": {
            "model": "Wan Video Models系列",
            "prompt": "",
            "image_url": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
    },
    "video:x-series": {
        "label": "全能视频X系列",
        "kind": "video",
        "price": "以 RunningHub 模型API价格页为准",
        "default_payload": {
            "model": "全能视频X系列",
            "prompt": "",
            "image_url": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
    },
    "video:seedance-2": {
        "label": "Seedance2.0系列",
        "kind": "video",
        "price": "以 RunningHub 模型API价格页为准",
        "default_payload": {
            "model": "Seedance2.0系列",
            "prompt": "",
            "image_url": "",
            "duration": 5,
            "aspect_ratio": "16:9",
        },
    },
}


MODEL_KEYS = tuple(MODEL_CATALOG.keys())


def model_label(model_key):
    return MODEL_CATALOG[model_key]["label"]


def model_price(model_key):
    return MODEL_CATALOG[model_key]["price"]
