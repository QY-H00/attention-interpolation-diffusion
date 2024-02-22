import torch
from torch import FloatTensor
from diffusers import StableDiffusionPipeline, DDIMScheduler
from interpolation import linear_interpolation, sphere_interpolation, InterpolationAttnProcessorFull


class InterpolationDiffusion:
    '''
    Diffusion that generates interpolated images
    '''
    def __init__(self, repo_name: str="CompVis/stable-diffusion-v1-4", torch_device: str="cuda", num_inference_step=25):
        self.pipeline = StableDiffusionPipeline.from_pretrained(repo_name)
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(repo_name, subfolder="scheduler")
        self.num_inference_step = num_inference_step
        self.pipeline.scheduler.set_timesteps(num_inference_step)
        self.torch_device = torch_device
    
    def interpolate(self, latent1: FloatTensor, latent2: FloatTensor, prompt1: str, prompt2: str, guide_prompt=None, size=3, boost_ratio=0.3):
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
        
        # Prepare interpolated input
        latents = sphere_interpolation(latent1, latent2, size)
        emb1 = self.pipeline.encode_prompt(prompt1, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None)
        emb2 = self.pipeline.encode_prompt(prompt2, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None)
        if guide_prompt is not None:
            guide_emb = self.pipeline.encode_prompt(guide_prompt, device=self.torch_device, num_images_per_prompt=1, negative_prompt=None)
            embs_first_half = linear_interpolation(emb1, guide_emb, size // 2)
            embs_second_half = linear_interpolation(guide_emb, emb2, size - size // 2)
            embs = torch.cat([embs_first_half, embs_second_half], dim=0)
        else:
            embs = linear_interpolation(emb1, emb2, size)
        
        # Two-stage interpolation
        num_initialize_step = int(boost_ratio * self.num_inference_step)
        num_refine_step = self.num_inference_step - num_initialize_step
        
        # Stage 1: Spatial intialization
        initialize_time_step = self.pipeline.scheduler.timesteps[:num_initialize_step]
        spatial_interpolate_attn_proc = InterpolationAttnProcessorFull(size=size, is_fused=False, alpha=num_initialize_step, beta=num_initialize_step)
        self.pipeline.unet.set_attn_processor(processor=spatial_interpolate_attn_proc)
        latents = self.pipeline(latents=latents, prompt_embeds=embs, output_type="latent", timesteps=initialize_time_step, return_dict=False)[0]
        
        # Stage 2: Semantic refinement
        refine_time_step = self.pipeline.scheduler.timesteps[num_initialize_step:]
        refine_interpolate_attn_proc = InterpolationAttnProcessorFull(size=size, is_fused=True, alpha=num_refine_step, beta=num_refine_step)
        self.pipeline.unet.set_attn_processor(processor=refine_interpolate_attn_proc)
        latents = self.pipeline(latents=latents, prompt_embeds=embs, output_type="latent", timesteps=refine_time_step, return_dict=False)[0]
        
        # Get the list of PIL images
        images = self.pipeline(latents=latents, prompt_embeds=embs, num_inference_timestep=0).images
        
        return images
        
        
