import torch

from ..reference.ref_controller import RefController, RefMode
from ..reference.ref_config import RefConfig
from ..utils.noise_utils import add_noise


def get_unet_wrapper(cls, ref_controller: RefController):
    class RefUNet(cls):
        is_ref = True

        def _ref_motion_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            flow_options = transformer_options.get('flow_options', None)
            ref_config: RefConfig = transformer_options.get(
                'ref_motion_config', None)
            if ref_config is None or len(ref_config.settings) == 0 or (flow_options is not None and flow_options._state == 0):
                return RefMode.OFF, None

            ad_params = transformer_options['ad_params']
            sub_idxs = ad_params.get('sub_idxs', None)
            n_conds = len(transformer_options['cond_or_uncond'])
            n_frames = len(x) // n_conds
            ref_count = ad_params['full_length'] // 16
            if sub_idxs is None:
                sub_idxs = list(range(n_frames))
                ref_count = 1

            ref_latent = ref_config.ref_latents
            if ad_params['full_length'] > len(ref_latent):
                sub_idxs = list(range(n_frames))
                ref_count = 1

            ref_controller.sef_ref_count(1)

            if len(ref_latent) < n_frames:
                raise ValueError(
                    "Motion Reference latent must be at least as long as sample latent.")

            ref_latent = ref_latent[sub_idxs]
            ref_latent = torch.cat([ref_latent]*n_conds)
            sigma = ref_config.sampling.sigma(timesteps)
            start_sigma = ref_config.sampling.percent_to_sigma(
                ref_config.start_percent)
            end_sigma = ref_config.sampling.percent_to_sigma(
                ref_config.end_percent)
            if not (start_sigma >= sigma[0] >= end_sigma):
                return RefMode.OFF, ref_config

            ref_latent_noised = add_noise(ref_latent, torch.randn_like(
                ref_latent), sigma[0]).to(x.device).to(x.dtype)

            transformer_options = transformer_options.copy()
            if 'flow_options' in transformer_options:
                del transformer_options['flow_options']

            positive = ref_config.positive.to(ref_latent_noised.device).half()
            negative = ref_config.negative.to(ref_latent_noised.device).half()

            prompt = torch.cat(([negative] * n_frames) +
                               ([positive] * n_frames))

            ref_controller.set_motion_mode(RefMode.WRITE, ref_config)
            super().forward(ref_latent_noised,
                            timesteps=torch.cat(
                                [timesteps[0].unsqueeze(0)]*(n_conds*n_frames)),
                            context=prompt,
                            y=y,
                            control=None,
                            transformer_options=transformer_options,
                            **kwargs)
            ref_controller.set_motion_mode(RefMode.OFF, ref_config)

            return RefMode.READ, ref_config

        def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            ref_controller.clear_modules()
            try:
                motion_mode, motion_conifg = self._ref_motion_forward(
                    x,
                    timesteps=timesteps,
                    context=context,
                    y=y,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs
                )
                ref_controller.set_motion_mode(motion_mode, motion_conifg)
                output = super().forward(x,
                                         timesteps=timesteps,
                                         context=context,
                                         y=y,
                                         control=control,
                                         transformer_options=transformer_options,
                                         **kwargs)
                ref_controller.set_motion_mode(RefMode.OFF)
                return output
            finally:
                ref_controller.clear_modules()

    return RefUNet


def setup_ref_unet(model):
    if not hasattr(model.model.diffusion_model, 'is_ref'):
        ref_controller = RefController(model.model.diffusion_model)
        model.model.diffusion_model.__class__ = get_unet_wrapper(
            model.model.diffusion_model.__class__, ref_controller)
        return ref_controller
    return None
