import torch

from ..modules.unet_wrapper import setup_ref_unet
from ..modules.temporal_attn_wrapper import wrap_temporal_attentions
from ..reference.ref_config import RefConfig
from ..utils.ref_utils import prepare_ref_latents


class ApplyRefMotionNode:
    @classmethod
    def INPUT_TYPES(s):

        return {"required": { 
            "model": ("MODEL",),
            "ref_latents": ("LATENT",),
            "enabled": ("BOOLEAN", {"default": True }),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "ref_settings": ("MOTION_REF_SETTINGS",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, 
              model, 
              ref_latents, 
              enabled,
              positive, 
              negative,
              start_percent,
              end_percent,
              ref_settings):
        if not enabled:
            return (model, )
        
        model = model.clone()
        transformer_options = model.model_options.get('transformer_options', {})
        model.model_options['transformer_options'] = transformer_options

        ref_latents = prepare_ref_latents(model, ref_latents)
        sampling = model.model.model_sampling

        prompt = torch.cat([negative[0][0]]* 16 +[positive[0][0]]* 16)

        ref_config = RefConfig(
            ref_latents,
            prompt,
            sampling,
            start_percent,
            end_percent,
            ref_settings
        )

        transformer_options['ref_motion_config'] = ref_config

        ref_controller = setup_ref_unet(model)
        if ref_controller is not None:
            model.model_options['ref_controller'] = ref_controller

        base_patch_model_fn = model.patch_model

        def patch_model(*args, **kwargs):
            rtrn = base_patch_model_fn(*args, **kwargs)
            diffusion_model = model.model.diffusion_model
            wrap_temporal_attentions(diffusion_model)
            return rtrn

        # HACK: This is for experiment purposes only
        model.patch_model = patch_model

        return (model, )
