from einops import rearrange, repeat
import torch

from comfy.ldm.modules.attention import attention_basic, attention_pytorch, attention_split, attention_sub_quad
from comfy import model_management
from comfy.cli_args import args

from ..reference.ref_mode import RefMode
from ..utils.module_utils import is_named_module_transformer_block


# From ADE
# until xformers bug is fixed, do not use xformers for VersatileAttention! TODO: change this when fix is out
# logic for choosing optimized_attention method taken from comfy/ldm/modules/attention.py
optimized_attention_mm = attention_basic
if model_management.pytorch_attention_enabled():
    optimized_attention_mm = attention_pytorch
else:
    if args.use_split_cross_attention:
        optimized_attention_mm = attention_split
    else:
        optimized_attention_mm = attention_sub_quad


def get_attention_wrapper(cls):
    class TemporalAttentionWrapper(cls):
        old_class = cls
        is_ref = True

        ref_mode = RefMode.OFF
        q_mode = False
        k_mode = False
        v_mode = False
        normal_mode = False
        ref_norm_fidelity = 1

        q_bank = None
        k_bank = None
        v_bank = None
        normal_bank = None
        ref_count = 1

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

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            video_length=None,
            scale_mask=None,
        ):
            if self.attention_mode != "Temporal":
                raise NotImplementedError

            d = hidden_states.shape[1]
            b = hidden_states.shape[0] // video_length
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )

            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(
                    hidden_states).to(hidden_states.dtype)

            encoder_hidden_states = (
                repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                if encoder_hidden_states is not None
                else encoder_hidden_states
            )

            if self.ref_mode == RefMode.READ and self.normal_mode and self.ref_norm_fidelity > 0.0:
                norm_hidden_states = hidden_states.clone()

            hidden_states = self.sub_forward(
                hidden_states,
                encoder_hidden_states,
                value=None,
                mask=attention_mask,
                scale_mask=scale_mask,
                uncond=False
            )  # [8192, 16, 320]

            if self.ref_mode == RefMode.READ and self.normal_mode and self.ref_norm_fidelity > 0.0:
                uc_hidden_states = hidden_states.clone()
                # [(b d) f h]
                uc_mask = torch.Tensor(
                    [1] * d
                    + [0] * d
                ).to(uc_hidden_states.device).bool()

                if encoder_hidden_states is None:
                    encoder_hidden_states = norm_hidden_states

                uc_hidden_states[uc_mask] = self.sub_forward(
                    norm_hidden_states[uc_mask],
                    encoder_hidden_states[uc_mask],
                    value=None,
                    mask=attention_mask,
                    scale_mask=scale_mask,
                    uncond=True
                )
                hidden_states = self.ref_norm_fidelity * uc_hidden_states + \
                    (1.0 - self.ref_norm_fidelity) * hidden_states

            hidden_states = rearrange(
                hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        def sub_forward(self, x, context=None, value=None, mask=None, scale_mask=None, uncond=False):
            context = context if context is not None else x
            value = value if value is not None else context

            context_k = context
            value_v = value
            if self.ref_mode == RefMode.WRITE and self.normal_mode and not uncond:
                self.normal_bank = x
            elif self.ref_mode == RefMode.READ and self.normal_mode and not uncond:
                context_k = torch.cat([context, self.normal_bank], dim=1)
                value_v = torch.cat([value, self.normal_bank], dim=1)

            q = self.to_q(x)
            k = self.to_k(context_k)
            v = self.to_v(value_v)

            if self.ref_mode == RefMode.READ and not uncond:
                if self.q_mode:
                    q = self.q_bank
                if self.k_mode:
                    k = self.k_bank
                if self.v_mode:
                    v = self.v_bank
            elif self.ref_mode == RefMode.WRITE and not uncond:
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

            attn_output = optimized_attention_mm(q, k, v, self.heads, mask)

            return self.to_out(attn_output)

    return TemporalAttentionWrapper


def wrap_temporal_attentions(model):
    # HACK: This is for experiment reasons only
    input_modules = list(
        filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
    middle_modules = list(
        filter(is_named_module_transformer_block, model.middle_block.named_modules()))
    output_modules = list(
        filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

    for _, module in input_modules + output_modules + middle_modules:
        for i in range(len(module.attention_blocks)):
            attn = module.attention_blocks[i]
            attn.__class__ = get_attention_wrapper(attn.__class__)


def unwrap_temporal_attentions(model):
    # HACK: This is for experiment reasons only
    input_modules = list(
        filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
    middle_modules = list(
        filter(is_named_module_transformer_block, model.middle_block.named_modules()))
    output_modules = list(
        filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

    for _, module in input_modules + output_modules + middle_modules:
        for i in range(len(module.attention_blocks)):
            attn = module.attention_blocks[i]
            attn.__class__ = attn.old_class
