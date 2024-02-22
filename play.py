import torch
from diffusion import InterpolationDiffusion
import matplotlib.pyplot as plt
from PIL import Image
import torch
from diffusion import InterpolationDiffusion
import os

def showImagesHorizontally(list_of_files, output_file):
    number_of_files = len(list_of_files)

    # Create a figure with subplots
    _, axs = plt.subplots(1, number_of_files, figsize=(10, 5))

    for i in range(number_of_files):
        _image = list_of_files[i]
        axs[i].imshow(_image)
        axs[i].axis("off")
        
    # Save the figure as PNG
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.1)

if __name__ == "__main__":
    # Load the model
    model = InterpolationDiffusion()
    
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
    showImagesHorizontally(images, os.path.join(save_dir, f"steps={num_inference_steps}_boostR={boost_ratio}.png"))