
from ..reference.ref_config import RefSetting


class MotionRefSettingsDefaultNode:
    @classmethod
    def INPUT_TYPES(s):
        d = {"required": { 
            "enabled": ("BOOLEAN", {"default": True }),
        }, "optional": {
            "prev_settings": ("MOTION_REF_SETTINGS", ),
        }}

        return d
    RETURN_TYPES = ("MOTION_REF_SETTINGS",)
    FUNCTION = "add"

    CATEGORY = "reference"

    def add(self, 
              enabled,
                prev_settings=[]):
        if not enabled:
            return (prev_settings, )
        
        input_attentions = [True, True, True, True, True, True, True, True]
        output_attentions = [True, True, True, True, True, True, 
                True, True, True, False, False, False]

        ref_setting = RefSetting(
            input_attentions,
            output_attentions,
            False,
            False,
            False,
            True
        )
        
        ref_settings = [*prev_settings, ref_setting]

        return (ref_settings, )



class MotionRefSettingsCustomNode:
    @classmethod
    def INPUT_TYPES(s):
        inputs = list(range(1, 9))
        outputs = list(range(1, 13))

        d = {"required": { 
            "enabled": ("BOOLEAN", {"default": True }),
            "q_bank": ("BOOLEAN", { "default": False }),
            "k_bank": ("BOOLEAN", { "default": False }),
            "v_bank": ("BOOLEAN", { "default": False }),
            "norm_bank": ("BOOLEAN", { "default": False }),
        }, "optional": {
            "prev_settings": ("MOTION_REF_SETTINGS", ),
        }}

        for i in inputs:
            d['required'][f'input_{i}'] = ("BOOLEAN", { "default": False })

        for i in outputs:
            d['required'][f'output_{i}'] = ("BOOLEAN", { "default": False })

        return d
    RETURN_TYPES = ("MOTION_REF_SETTINGS",)
    FUNCTION = "add"

    CATEGORY = "reference"

    def add(self, 
              enabled,
              q_bank,
                k_bank,
                v_bank,
                norm_bank,
                input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8,
                output_1, output_2, output_3, output_4, output_5, output_6, 
                output_7, output_8, output_9, output_10, output_11, output_12,
                prev_settings=[]):
        if not enabled:
            return (prev_settings, )
        
        input_attentions = [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8]
        output_attentions = [output_1, output_2, output_3, output_4, output_5, output_6, 
                output_7, output_8, output_9, output_10, output_11, output_12]

        ref_setting = RefSetting(
            input_attentions,
            output_attentions,
            q_bank,
            k_bank,
            v_bank,
            norm_bank
        )
        
        ref_settings = [*prev_settings, ref_setting]

        return (ref_settings, )
