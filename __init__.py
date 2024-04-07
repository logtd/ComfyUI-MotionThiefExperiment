from .nodes.ref_apply_node import ApplyRefMotionNode
from .nodes.ref_settings_node import MotionRefSettingsDefaultNode, MotionRefSettingsCustomNode


NODE_CLASS_MAPPINGS = {
    "ApplyRefMotionNode": ApplyRefMotionNode,
    "MotionRefSettingsDefaultNode": MotionRefSettingsDefaultNode,
    "MotionRefSettingsCustomNode": MotionRefSettingsCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyRefMotionNode": "Apply Ref Motion",
    "MotionRefSettingsDefaultNode": "Motion Ref Setting",
    "MotionRefSettingsCustomNode": "Motion Ref Settings (Custom)"
}
