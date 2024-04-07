import comfy.model_management


def prepare_ref_latents(model, ref_latent):
    base_model = model.model
    ref_latent = ref_latent['samples'].clone()
    ref_latent = base_model.process_latent_in(ref_latent)
    device = comfy.model_management.get_torch_device()
    ref_latent = ref_latent.to(device)
    return ref_latent
