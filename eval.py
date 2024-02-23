import os
import random
import torch
from tqdm.auto import tqdm
from diffusion import InterpolationStableDiffusionPipeline
from utils import save_images


def prepare_imgs(corpus, iters=1000, trial_per_iters=5):
    # Load the model
    pipe = InterpolationStableDiffusionPipeline()
    
    # Initialize the generator
    vae_scale_factor = 8
    channel = pipe.unet.config.in_channels
    height = pipe.unet.config.sample_size * vae_scale_factor
    width = pipe.unet.config.sample_size * vae_scale_factor
    TORCH_DEVICE = "cuda"
    generator = torch.cuda.manual_seed(0)
    
    for _ in tqdm(range(iters)):
        prompt1, prompt2 = random.sample(corpus, 2)
        save_dir = '{' + prompt1 + '}_{' + prompt2 + '}'
        save_dir = os.path.join("results", "direct_eval", save_dir)
        for idx in range(trial_per_iters):
            save_dir = os.path.join(save_dir, str(idx+1))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            latent = torch.randn(
                (1, channel, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=TORCH_DEVICE,
            )
            num_inference_steps = 25
            boost_ratio = 0.3
            
            images = pipe.interpolate(latent, latent, prompt1, prompt2, guide_prompt=None, size=5, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio)
            save_images(images, save_dir)
            

if __name__ == "__main__":
    corpus = [
        "banana",
        "pen"]
    prepare_imgs(corpus, 1, 1)

    