"""
LLFaigc - filtered RunningHub OpenAPI nodes for ComfyUI.

This package keeps the original HM-RunningHub OpenAPI node runtime and only
filters models_registry.json to the image/video model families requested by
LLFaigc.
"""

from .nodes.node_factory import create_all_nodes
from .nodes.settings_node import RHSettingsNode
from .nodes.assets import (
    NODE_CLASS_MAPPINGS as _asset_class_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as _asset_display_mappings,
)
from .nodes.gpt_image2_node import LLFaigcGptImage2Official
from .nodes.gpt_image2_fal_node import LLFaigcGptImage2Fal
from .nodes.llm_node import LLFaigcLLM
from .nodes.rh_llm_node import LLFaigcRHLLM
from .nodes.rh_llm_chat_node import LLFaigcRHLLMChat

_class_mappings, _display_mappings = create_all_nodes()

NODE_CLASS_MAPPINGS = {
    "LLFaigcSettingsNode": RHSettingsNode,
    "LLFaigcGptImage2Official": LLFaigcGptImage2Official,
    "LLFaigcGptImage2Fal": LLFaigcGptImage2Fal,
    "LLFaigcLLM": LLFaigcLLM,
    "LLFaigcRHLLM": LLFaigcRHLLM,
    "LLFaigcRHLLMChat": LLFaigcRHLLMChat,
    **_asset_class_mappings,
    **_class_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLFaigcSettingsNode": "LLFaigc OpenAPI Settings",
    "LLFaigcGptImage2Official": "LLFaigc GPT-Image-2 Official",
    "LLFaigcGptImage2Fal": "LLFaigc GPT-Image-2 FAL",
    "LLFaigcLLM": "LLF-LLM",
    "LLFaigcRHLLM": "LLF-RH-LLM",
    "LLFaigcRHLLMChat": "LLF-RH-LLM-Chat",
    **_asset_display_mappings,
    **_display_mappings,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
