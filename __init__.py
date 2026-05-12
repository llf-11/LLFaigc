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

_class_mappings, _display_mappings = create_all_nodes()

NODE_CLASS_MAPPINGS = {
    "LLFaigcSettingsNode": RHSettingsNode,
    **_asset_class_mappings,
    **_class_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLFaigcSettingsNode": "LLFaigc OpenAPI Settings",
    **_asset_display_mappings,
    **_display_mappings,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
