from ..utils.module_utils import is_named_module_transformer_block
from .ref_mode import RefMode
from .ref_config import RefConfig


class RefController:
    model = None

    def __init__(self, diffusion_model):
        self.model = diffusion_model

    def set_motion_mode(self, mode: RefMode, config: RefConfig = None):
        model = self.model
        input_modules = list(filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
        output_modules = list(filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

        for _, module in input_modules + output_modules:
            for i in range(len(module.attention_blocks)):
                attn = module.attention_blocks[i]
                attn.ref_off()

        if config is None or len(config.settings) == 0:
            return

        for setting in config.settings:
            for i, (_, module) in enumerate(input_modules):
                for attn_idx in range(len(module.attention_blocks)):
                    attn = module.attention_blocks[attn_idx]
                    if setting.input_attentions[i]:
                        attn.ref_mode = mode
                        if setting.q_mode:
                            attn.q_mode = True
                        if setting.k_mode:
                            attn.k_mode = True
                        if setting.v_mode:
                            attn.v_mode = True
                        if setting.normal_mode:
                            attn.normal_mode = True
            for i, (_, module) in enumerate(output_modules):
                for attn_idx in range(len(module.attention_blocks)):
                    attn = module.attention_blocks[attn_idx]
                    if setting.output_attentions[i]:
                        attn.ref_mode = mode
                        if setting.q_mode:
                            attn.q_mode = True
                        if setting.k_mode:
                            attn.k_mode = True
                        if setting.v_mode:
                            attn.v_mode = True
                        if setting.normal_mode:
                            attn.normal_mode = True

    def clear_modules(self):
        model = self.model
        input_modules = list(filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
        output_modules = list(filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

        for _, module in input_modules + output_modules:
            for i in range(len(module.attention_blocks)):
                attn = module.attention_blocks[i]
                attn.ref_off()
                attn.ref_clean()
