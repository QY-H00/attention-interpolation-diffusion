from typing import Union
from types import MethodType
import numpy as np
import torch
from torch import FloatTensor
from diffusers import StableDiffusionPipeline, DDIMScheduler
from interpolation import linear_interpolation, sphere_interpolation, InterpolationAttnProcessorFull


class InterpolationDiffusion:
    '''
    Diffusion that generates interpolated images
    '''
    def __init__(self, repo_name: str="CompVis/stable-diffusion-v1-4", torch_device: str="cuda", num_inference_step=25):
        self.pipeline = StableDiffusionPipeline.from_pretrained(repo_name, cache_dir="cache")
        self.pipeline.to(torch_device)
        ddim_scheduler = DDIMScheduler.from_pretrained(repo_name, subfolder="scheduler")
        ddim_scheduler.set_timesteps = MethodType(ddim_set_timesteps, ddim_scheduler)
        self.pipeline.scheduler = ddim_scheduler
        self.torch_device = torch_device
    
    def interpolate(self, latent1: FloatTensor, latent2: FloatTensor, prompt1: str, prompt2: str, guide_prompt=None, size=3, boost_ratio=0.3, num_inference_steps=25):
        '''
        Interpolate between two generation
        
        Args:
        latent1: FloatTensor, latent vector of the first image
        latent2: FloatTensor, latent vector of the second image
        prompt1: str, text prompt of the first image
        prompt2: str, text prompt of the second image
        guide_prompt: str, text prompt for the interpolation
        size: int, number of interpolations including starting and ending points
        
        Returns:
        List of nterpolated images
        '''
        assert latent1.shape == latent2.shape, "shapes of latent1 and latent2 must match"
        
        # Prepare interpolated inputs
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        latents = sphere_interpolation(latent1, latent2, size)
        guidance_scale = 7.5
        do_classifier_free_guidance = guidance_scale > 1 and self.pipeline.unet.config.time_cond_proj_dim is None
        emb1, _ = self.pipeline.encode_prompt(prompt1, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None, do_classifier_free_guidance=do_classifier_free_guidance)
        emb2, _ = self.pipeline.encode_prompt(prompt2, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None, do_classifier_free_guidance=do_classifier_free_guidance)
        if guide_prompt is not None:
            guide_emb, _ = self.pipeline.encode_prompt(guide_prompt, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None, do_classifier_free_guidance=do_classifier_free_guidance)
            embs_first_half = linear_interpolation(emb1, guide_emb, size // 2)
            embs_second_half = linear_interpolation(guide_emb, emb2, size - size // 2)
            embs = torch.cat([embs_first_half, embs_second_half], dim=0)
        else:
            embs = linear_interpolation(emb1, emb2, size)
        
        # Two-stage interpolation
        num_initialize_step = int(boost_ratio * num_inference_steps)
        num_refine_step = num_inference_steps - num_initialize_step
        initialize_time_step = self.pipeline.scheduler.timesteps[:num_initialize_step]
        refine_time_step = self.pipeline.scheduler.timesteps[num_initialize_step:]
        
        # Stage 1: Spatial intialization
        spatial_interpolate_attn_proc = InterpolationAttnProcessorFull(size=size, is_fused=False, alpha=num_initialize_step, beta=num_initialize_step)
        self.pipeline.unet.set_attn_processor(processor=spatial_interpolate_attn_proc)
        print("Spatial Initialize...")
        latents = self.pipeline(latents=latents, prompt_embeds=embs, guidance_scale=guidance_scale, output_type="latent", timesteps=initialize_time_step, return_dict=False)[0]
        
        # Stage 2: Semantic refinement
        refine_interpolate_attn_proc = InterpolationAttnProcessorFull(size=size, is_fused=True, alpha=num_refine_step, beta=num_refine_step)
        self.pipeline.unet.set_attn_processor(processor=refine_interpolate_attn_proc)
        print("Semantic Refine...")
        latents = self.pipeline(latents=latents, prompt_embeds=embs, guidance_scale=guidance_scale, output_type="latent", timesteps=refine_time_step, return_dict=False)[0]
        
        # Get the list of PIL images
        print("Decode...")
        images = self.pipeline(latents=latents, prompt_embeds=embs, num_inference_steps=0).images
        
        return images


def ddim_set_timesteps(self, num_inference_steps: int=None, timesteps=None, device: Union[str, torch.device] = None):
        """
        Modify DDIM Scheduler to make it aceepts custom timesteps

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        
        if num_inference_steps == 0:
            self.timesteps = torch.tensor([], device=device, dtype=torch.int64)
        elif timesteps is not None:
            self.timesteps = timesteps  
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
                )

            self.timesteps = torch.from_numpy(timesteps).to(device)
        
        
