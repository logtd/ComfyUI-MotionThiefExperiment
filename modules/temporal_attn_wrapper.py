import torch

from comfy.ldm.modules.attention import optimized_attention

from ..reference.ref_mode import RefMode
from ..utils.module_utils import is_named_module_transformer_block


def get_attention_wrapper(cls):
    superclass = cls.__bases__[0]

    class TemporalAttentionWrapper(superclass):
        is_ref = True

        ref_mode = RefMode.OFF
        q_mode = False
        k_mode = False
        v_mode = False
        normal_mode = False

        q_bank = None
        k_bank = None
        v_bank = None
        normal_bank = None

        def ref_off(self):
            self.ref_mode = RefMode.OFF
            self.q_mode = False
            self.k_mode = False
            self.v_mode = False
            self.normal_mode = False

        def ref_clean(self):
            self.q_bank = None
            self.v_bank = None
            self.k_bank = None
            self.normal_mode = None

        def forward(self, x, context=None, value=None, mask=None, scale_mask=None):
            context = context if context is not None else x
            value = value if value is not None else context
            if self.ref_mode == RefMode.WRITE and self.normal_mode:
                self.normal_bank = x
            elif self.ref_mode == RefMode.READ and self.normal_mode:
                context = torch.cat([context, self.normal_bank], dim=1)
                value = torch.cat([value, self.normal_bank], dim=1)

            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(value)

            if self.ref_mode == RefMode.READ:
                if self.q_mode:
                    q = self.q_bank
                if self.k_mode:
                    k = self.k_bank
                if self.v_mode:
                    v = self.v_bank
            elif self.ref_mode == RefMode.WRITE:
                if self.q_mode:
                    self.q_bank = q
                if self.k_mode:
                    self.k_bank = k
                if self.v_mode:
                    self.v_bank = v
            
            if self.scale is not None:
                k *= self.scale
            # apply scale mask, if present
            if scale_mask is not None:
                k *= scale_mask
            
            out = optimized_attention(q, k, v, self.heads, mask)
            return self.to_out(out)
    
    # HACK: This is for experiment reasons only
    cls.__bases__ = (TemporalAttentionWrapper,)
    
    return cls


def wrap_temporal_attentions(model):
    input_modules = list(filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
    output_modules = list(filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

    for _, module in input_modules + output_modules:
        for i in range(len(module.attention_blocks)):
            attn = module.attention_blocks[i]
            attn.__class__ = get_attention_wrapper(attn.__class__)