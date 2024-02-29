import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import lpips
from utils import compute_smoothness_and_efficiency, compute_lpips

from diffusion import InterpolationStableDiffusionPipeline
from utils import show_images_horizontally, baysian_prior_selection

if __name__ == "__main__":
    
    root_dir = os.path.join("results", "qualitative")
    os.makedirs(root_dir, exist_ok=True)
    pipe = InterpolationStableDiffusionPipeline()
    
    # Initialize the generator
    vae_scale_factor = 8
    channel = pipe.unet.config.in_channels
    height = pipe.unet.config.sample_size * vae_scale_factor
    width = pipe.unet.config.sample_size * vae_scale_factor
    torch_device = "cuda"
    
    
    prompt1 = "an airplane"
    prompt2 = "a deer"
    file_name = '{' + prompt1[:20] + '}_{' + prompt2[:20] + '}'
    file_path = os.path.join(root_dir, file_name)
    generator = torch.cuda.manual_seed(0)
        
    latent = torch.randn(
        (1, channel, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
        device=torch_device,
    )
    
    num_inference_steps = 25
    lpips_model = lpips.LPIPS(net="vgg").to("cuda")
    
    boost_ratio = 1.0
    early = "vfused"
    late = "self"
    guide_prompt = None
    
    if boost_ratio == 1.0:
        file_suffix = f"_{early}"
    elif boost_ratio == 0.0:
        file_suffix = f"_{late}"
    else:
        file_suffix = f"_{early}_{late}"
    
    alpha, beta = baysian_prior_selection(pipe, latent, latent, prompt1, prompt2, lpips_model, guide_prompt=guide_prompt, size=3, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early=early, late=late)
    images = pipe.interpolate_save_gpu(latent, latent, prompt1, prompt2, guide_prompt=guide_prompt, size=7, 
                                       num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early=early, late=late, alpha=alpha, beta=beta)
    smoothness, efficiency, max_distance = compute_smoothness_and_efficiency(images, lpips_model, device="cuda")
    print(f"{file_suffix}", smoothness, efficiency, max_distance)
    if guide_prompt is not None:
        guide_prompt = guide_prompt[:20]
    show_images_horizontally(images, file_path + f"{file_suffix}_{str(guide_prompt)}.png")