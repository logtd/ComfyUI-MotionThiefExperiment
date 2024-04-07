from ..modules.unet_wrapper import get_unet_wrapper
from ..reference.ref_controller import RefController

def setup_ref_unet(model):
    if not hasattr(model.model.diffusion_model, 'is_ref'):
        ref_controller = RefController(model.model.diffusion_model)
        model.model.diffusion_model.__class__ = get_unet_wrapper(model.model.diffusion_model.__class__, ref_controller)
        return ref_controller
    return None