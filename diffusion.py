from typing import Union
from types import MethodType
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import FloatTensor
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from interpolation import linear_interpolation, sphere_interpolation, InterpolationAttnProcessorWithUncond, InterpolationAttnProcessor


class InterpolationDiffusionGeneral:
    '''
    Diffusion that generates interpolated images
    '''
    def __init__(self, repo_name: str="CompVis/stable-diffusion-v1-4", torch_device: str="cuda"):
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


class InterpolationStableDiffusionPipeline:
    
    def __init__(self, repo_name="CompVis/stable-diffusion-v1-4", device="cuda", frozen=True):
        self.vae = AutoencoderKL.from_pretrained(repo_name, subfolder="vae", use_safetensors=True, cache_dir="weights")
        self.tokenizer = CLIPTokenizer.from_pretrained(repo_name, subfolder="tokenizer", cache_dir="weights")
        self.text_encoder = CLIPTextModel.from_pretrained(
            repo_name, subfolder="text_encoder", use_safetensors=True, cache_dir="weights"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            repo_name, subfolder="unet", use_safetensors=True, cache_dir="weights"
        )
        self.scheduler = DDIMScheduler.from_pretrained(repo_name, subfolder="scheduler", cache_dir="weights")
        self.torch_device = device
        self.vae.to(self.torch_device)
        self.text_encoder.to(self.torch_device)
        self.unet.to(self.torch_device)
        self.guidance_scale = 7.5  # Scale for classifier-free guidance
        
        if frozen:
            for param in self.unet.parameters():
                param.requires_grad = False

            for param in self.text_encoder.parameters():
                param.requires_grad = False

            for param in self.vae.parameters():
                param.requires_grad = False

    def prompt_to_embedding(self, prompt):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def retrieve_from_latent(self, latents, text_embeddings, timesteps=25):
        self.scheduler.set_timesteps(timesteps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        return image

    def interpolate(self, latent1, latent2, prompt1, prompt2, guide_prompt=None, size=10, num_inference_steps=25, boost_ratio=0.5, early="cross", late="self"):
        # Prepare interpolated inputs
        self.scheduler.set_timesteps(num_inference_steps)
        latents = sphere_interpolation(latent1, latent2, size)
        embs1 = self.prompt_to_embedding(prompt1)
        emb1 = embs1[0:1]
        uncond_emb1 = embs1[1:2]
        embs2 = self.prompt_to_embedding(prompt2)
        emb2 = embs2[0:1]
        uncond_emb2 = embs2[1:2]
        if guide_prompt is not None:
            guide_embs = self.prompt_to_embedding(guide_prompt)
            guide_emb = guide_embs[0:1]
            uncond_guide_emb = guide_embs[1:2]
            embs_first_half = linear_interpolation(emb1, guide_emb, size // 2)
            embs_second_half = linear_interpolation(guide_emb, emb2, size - size // 2)
            embs = torch.cat([embs_first_half, embs_second_half], dim=0)
            uncond_embs_first_half = linear_interpolation(uncond_emb1, uncond_guide_emb, size // 2)
            uncond_embs_second_half = linear_interpolation(uncond_guide_emb, uncond_emb2, size - size // 2)
            uncond_embs = torch.cat([uncond_embs_first_half, uncond_embs_second_half], dim=0)
        else:
            embs = linear_interpolation(emb1, emb2, size)
            uncond_embs = linear_interpolation(uncond_emb1, uncond_emb2, size)

        i = 0
        boost_step = int(num_inference_steps * boost_ratio)
        for t in tqdm(self.scheduler.timesteps):
            i += 1
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = self.scheduler.scale_model_input(latents, timestep=t)
            # predict the noise residual
            with torch.no_grad():
                if i < boost_step:
                    if early == "cross":
                        interpolate_attn_proc = InterpolationAttnProcessor(size=size, is_fused=False, alpha=num_inference_steps - boost_step, beta=num_inference_steps - boost_step)
                    elif early == "fused":
                        interpolate_attn_proc = InterpolationAttnProcessor(size=size, is_fused=True, alpha=num_inference_steps - boost_step, beta=num_inference_steps - boost_step)
                    else:
                        raise ValueError("Invalid early parameter")
                else:
                    if late == "self":
                        interpolate_attn_proc = AttnProcessor()
                    elif late == "fused":
                        interpolate_attn_proc = InterpolationAttnProcessor(size=size, is_fused=True, alpha=num_inference_steps - boost_step, beta=num_inference_steps - boost_step)
                    else:
                        raise ValueError("Invalid early parameter")
                    
                self.unet.set_attn_processor(processor=interpolate_attn_proc)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=embs).sample
                attn_proc = AttnProcessor()
                self.unet.set_attn_processor(processor=attn_proc)
                noise_uncond = self.unet(latent_model_input, t, encoder_hidden_states=uncond_embs).sample
            # perform guidance
            noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        images = (image / 2 + 0.5).clamp(0, 1)
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
        
        
