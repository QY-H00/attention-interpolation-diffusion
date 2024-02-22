import torch
from diffusion import InterpolationDiffusion

if __name__ == "__main__":
    # Load the model
    model = InterpolationDiffusion()
    
    # Initialize the generator
    channel = model.pipeline.unet.config.in_channels
    height = model.pipeline.default_sample_size * model.pipeline.vae_scale_factor
    width = model.pipeline.default_sample_size * model.pipeline.vae_scale_factor
    torch_device = "cuda"
    generator = torch.cuda.manual_seed(0)
    
    # Load the latent vectors
    latent1 = torch.randn(
        (1, channel, height // model.pipeline.vae_scale_factor, width // model.pipeline.vae_scale_factor),
        generator=generator,
        device=torch_device,
    )
    
    latent2 = torch.randn(
        (1, channel, height // model.pipeline.vae_scale_factor, width // model.pipeline.vae_scale_factor),
        generator=generator,
        device=torch_device,
    )
    
    # Load the prompt
    prompt1 = "A painting of a cat"
    prompt2 = "A painting of a dog"
    
    # Interpolate
    images = model.interpolate(latent1, latent2, prompt1, prompt2, guide_prompt=None, size=3, boost_ratio=0.3)