import os
from PIL import Image
import matplotlib.pyplot as plt
import torch

from diffusion import InterpolationStableDiffusionPipeline
from utils import show_images_horizontally

if __name__ == "__main__":
    # Load the model
    model = InterpolationStableDiffusionPipeline()
    
    # Initialize the generator
    channel = model.pipeline.unet.config.in_channels
    height = model.pipeline.unet.config.sample_size * model.pipeline.vae_scale_factor
    width = model.pipeline.unet.config.sample_size * model.pipeline.vae_scale_factor
    TORCH_DEVICE = "cuda"
    generator = torch.cuda.manual_seed(0)
    
    # Load the latent vectors
    latent1 = torch.randn(
        (1, channel, height // model.pipeline.vae_scale_factor, width // model.pipeline.vae_scale_factor),
        generator=generator,
        device=TORCH_DEVICE,
    )
    
    latent2 = torch.randn(
        (1, channel, height // model.pipeline.vae_scale_factor, width // model.pipeline.vae_scale_factor),
        generator=generator,
        device=TORCH_DEVICE,
    )
    
    # Load the prompt
    prompt1 = "A painting of a cat"
    prompt2 = "A painting of a dog"
    
    # Set the number of timesteps and boost ratio
    num_inference_steps = 50
    boost_ratio = 0.3
    
    # Interpolate
    images = model.interpolate(latent1, latent2, prompt1, prompt2, guide_prompt=None, size=3, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio)
    
    save_dir = os.path.join("results", "{" + prompt1 + "}" + "-" + "{" + prompt2 + "}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Target Directory: ", save_dir)
    show_images_horizontally(images, os.path.join(save_dir, f"steps={num_inference_steps}_boostR={boost_ratio}.png"))