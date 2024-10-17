from typing import Optional

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    SchedulerMixin,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm.auto import tqdm

from interpolation import (
    InnerInterpolatedAttnProcessor,
    OuterInterpolatedAttnProcessor,
    generate_beta_tensor,
    linear_interpolation,
    slerp,
    spherical_interpolation,
)
from transformers import CLIPTextModel, CLIPTokenizer


class InterpolationStableDiffusionPipeline:
    """
    Diffusion Pipeline that generates interpolated images
    """

    def __init__(
        self,
        repo_name: str = "CompVis/stable-diffusion-v1-4",
        scheduler_name: str = "ddim",
        frozen: bool = True,
        guidance_scale: float = 7.5,
        scheduler: Optional[SchedulerMixin] = None,
        cache_dir: Optional[str] = None,
    ):
        # Initialize the generator
        self.vae = AutoencoderKL.from_pretrained(
            repo_name, subfolder="vae", use_safetensors=True, cache_dir=cache_dir
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            repo_name, subfolder="tokenizer", cache_dir=cache_dir
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            repo_name,
            subfolder="text_encoder",
            use_safetensors=True,
            cache_dir=cache_dir,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            repo_name, subfolder="unet", use_safetensors=True, cache_dir=cache_dir
        )

        # Initialize the scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif scheduler_name == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                repo_name, subfolder="scheduler", cache_dir=cache_dir
            )
        elif scheduler_name == "unipc":
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                repo_name, subfolder="scheduler", cache_dir=cache_dir
            )
        else:
            raise ValueError(
                "Invalid scheduler name (ddim, unipc) and not specify scheduler."
            )

        # Setup device

        self.guidance_scale = guidance_scale  # Scale for classifier-free guidance

        if frozen:
            for param in self.unet.parameters():
                param.requires_grad = False

            for param in self.text_encoder.parameters():
                param.requires_grad = False

            for param in self.vae.parameters():
                param.requires_grad = False

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.unet.to(*args, **kwargs)

    def generate_latent(
        self, generator: Optional[torch.Generator] = None, torch_device: str = "cpu"
    ) -> torch.FloatTensor:
        """
        Generates a random latent tensor.

        Args:
            generator (Optional[torch.Generator], optional): Generator for random number generation. Defaults to None.
            torch_device (str, optional): Device to store the tensor. Defaults to "cpu".

        Returns:
            torch.FloatTensor: Random latent tensor.
        """
        channel = self.unet.config.in_channels
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size
        if generator is None:
            latent = torch.randn(
                (1, channel, height, width),
                device=torch_device,
            )
        else:
            latent = torch.randn(
                (1, channel, height, width),
                generator=generator,
                device=torch_device,
            )
        return latent

    @torch.no_grad()
    def prompt_to_embedding(
        self, prompt: str, negative_prompt: str = ""
    ) -> torch.FloatTensor:
        """
        Prepare the text prompt for the diffusion process

        Args:
            prompt: str, text prompt
            negative_prompt: str, negative text prompt

        Returns:
            FloatTensor, text embeddings
        """

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[
            0
        ]

        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.torch_device)
        )[0]

        text_embeddings = torch.cat([text_embeddings, uncond_embeddings])
        return text_embeddings

    @torch.no_grad()
    def interpolate(
        self,
        latent_start: torch.FloatTensor,
        latent_end: torch.FloatTensor,
        prompt_start: str,
        prompt_end: str,
        guide_prompt: Optional[str] = None,
        negative_prompt: str = "",
        size: int = 7,
        num_inference_steps: int = 25,
        warmup_ratio: float = 0.5,
        early: str = "fused_outer",
        late: str = "self",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Interpolate between two generation

        Args:
            latent_start: FloatTensor, latent vector of the first image
            latent_end: FloatTensor, latent vector of the second image
            prompt_start: str, text prompt of the first image
            prompt_end: str, text prompt of the second image
            guide_prompt: str, text prompt for the interpolation
            negative_prompt: str, negative text prompt
            size: int, number of interpolations including starting and ending points
            num_inference_steps: int, number of inference steps in scheduler
            warmup_ratio: float, ratio of warmup steps
            early: str, warmup interpolation methods
            late: str, late interpolation methods
            alpha: float, alpha parameter for beta distribution
            beta: float, beta parameter for beta distribution
            guidance_scale: Optional[float], scale for classifier-free guidance
        Returns:
            Numpy array of interpolated images, shape (size, H, W, 3)
        """
        # Specify alpha and beta
        self.torch_device = self.unet.device
        if alpha is None:
            alpha = num_inference_steps
        if beta is None:
            beta = num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        self.scheduler.set_timesteps(num_inference_steps)

        # Prepare interpolated latents and embeddings
        latents = spherical_interpolation(latent_start, latent_end, size)
        embs_start = self.prompt_to_embedding(prompt_start, negative_prompt)
        emb_start = embs_start[0:1]
        uncond_emb_start = embs_start[1:2]
        embs_end = self.prompt_to_embedding(prompt_end, negative_prompt)
        emb_end = embs_end[0:1]
        uncond_emb_end = embs_end[1:2]

        # Perform prompt guidance if it is specified
        if guide_prompt is not None:
            guide_embs = self.prompt_to_embedding(guide_prompt, negative_prompt)
            guide_emb = guide_embs[0:1]
            uncond_guide_emb = guide_embs[1:2]
            embs = torch.cat([emb_start] + [guide_emb] * (size - 2) + [emb_end], dim=0)
            uncond_embs = torch.cat(
                [uncond_emb_start] + [uncond_guide_emb] * (size - 2) + [uncond_emb_end],
                dim=0,
            )
        else:
            embs = linear_interpolation(emb_start, emb_end, size=size)
            uncond_embs = linear_interpolation(
                uncond_emb_start, uncond_emb_end, size=size
            )

        # Specify the interpolation methods
        pure_inner_attn_proc = InnerInterpolatedAttnProcessor(
            size=size,
            is_fused=False,
            alpha=alpha,
            beta=beta,
        )
        fused_inner_attn_proc = InnerInterpolatedAttnProcessor(
            size=size,
            is_fused=True,
            alpha=alpha,
            beta=beta,
        )
        pure_outer_attn_proc = OuterInterpolatedAttnProcessor(
            size=size,
            is_fused=False,
            alpha=alpha,
            beta=beta,
        )
        fused_outer_attn_proc = OuterInterpolatedAttnProcessor(
            size=size,
            is_fused=True,
            alpha=alpha,
            beta=beta,
        )
        self_attn_proc = AttnProcessor2_0()
        procs_dict = {
            "pure_inner": pure_inner_attn_proc,
            "fused_inner": fused_inner_attn_proc,
            "pure_outer": pure_outer_attn_proc,
            "fused_outer": fused_outer_attn_proc,
            "self": self_attn_proc,
        }

        # Denoising process
        i = 0
        warmup_step = int(num_inference_steps * warmup_ratio)
        for t in tqdm(self.scheduler.timesteps):
            i += 1
            latent_model_input = self.scheduler.scale_model_input(latents, timestep=t)
            with torch.no_grad():
                # Change attention module
                if i < warmup_step:
                    interpolate_attn_proc = procs_dict[early]
                else:
                    interpolate_attn_proc = procs_dict[late]
                self.unet.set_attn_processor(processor=interpolate_attn_proc)

                # Predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=embs
                ).sample
                attn_proc = AttnProcessor2_0()
                self.unet.set_attn_processor(processor=attn_proc)
                noise_uncond = self.unet(
                    latent_model_input, t, encoder_hidden_states=uncond_embs
                ).sample
            # perform guidance
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the images
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        images = (image / 2 + 0.5).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return images

    @torch.no_grad()
    def interpolate_save_gpu(
        self,
        latent_start: torch.FloatTensor,
        latent_end: torch.FloatTensor,
        prompt_start: str,
        prompt_end: str,
        guide_prompt: Optional[str] = None,
        negative_prompt: str = "",
        size: int = 7,
        num_inference_steps: int = 25,
        warmup_ratio: float = 0.5,
        early: str = "fused_outer",
        late: str = "self",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        init: str = "linear",
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Interpolate between two generation

        Args:
            latent_start: FloatTensor, latent vector of the first image
            latent_end: FloatTensor, latent vector of the second image
            prompt_start: str, text prompt of the first image
            prompt_end: str, text prompt of the second image
            guide_prompt: str, text prompt for the interpolation
            negative_prompt: str, negative text prompt
            size: int, number of interpolations including starting and ending points
            num_inference_steps: int, number of inference steps in scheduler
            warmup_ratio: float, ratio of warmup steps
            early: str, warmup interpolation methods
            late: str, late interpolation methods
            alpha: float, alpha parameter for beta distribution
            beta: float, beta parameter for beta distribution
            init: str, interpolation initialization methods

        Returns:
            Numpy array of interpolated images, shape (size, H, W, 3)
        """
        self.torch_device = self.unet.device
        # Specify alpha and beta
        if alpha is None:
            alpha = num_inference_steps
        if beta is None:
            beta = num_inference_steps
        betas = generate_beta_tensor(size, alpha=alpha, beta=beta)
        final_images = None

        # Generate interpolated images one by one
        for i in range(size - 2):
            it = betas[i + 1].item()
            if init == "denoising":
                images = self.denoising_interpolate(
                    latent_start,
                    prompt_start,
                    prompt_end,
                    negative_prompt,
                    interpolated_ratio=it,
                    timesteps=num_inference_steps,
                )
            else:
                images = self.interpolate_single(
                    it,
                    latent_start,
                    latent_end,
                    prompt_start,
                    prompt_end,
                    guide_prompt=guide_prompt,
                    num_inference_steps=num_inference_steps,
                    warmup_ratio=warmup_ratio,
                    early=early,
                    late=late,
                    negative_prompt=negative_prompt,
                    init=init,
                    guidance_scale=guidance_scale,
                )
            if size == 3:
                return images
            if i == 0:
                final_images = images[:2]
            elif i == size - 3:
                final_images = np.concatenate([final_images, images[1:]], axis=0)
            else:
                final_images = np.concatenate([final_images, images[1:2]], axis=0)
        return final_images

    def interpolate_single(
        self,
        it,
        latent_start: torch.FloatTensor,
        latent_end: torch.FloatTensor,
        prompt_start: str,
        prompt_end: str,
        guide_prompt: str = None,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        warmup_ratio: float = 0.5,
        early: str = "fused_outer",
        late: str = "self",
        init="linear",
        guidance_scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Interpolates between two latent vectors and generates a sequence of images.

        Args:
            it (float): Interpolation factor between latent_start and latent_end.
            latent_start (torch.FloatTensor): Starting latent vector.
            latent_end (torch.FloatTensor): Ending latent vector.
            prompt_start (str): Starting prompt for text conditioning.
            prompt_end (str): Ending prompt for text conditioning.
            guide_prompt (str, optional): Guiding prompt for text conditioning. Defaults to None.
            negative_prompt (str, optional): Negative prompt for text conditioning. Defaults to "".
            num_inference_steps (int, optional): Number of inference steps. Defaults to 25.
            warmup_ratio (float, optional): Ratio of warm-up steps. Defaults to 0.5.
            early (str, optional): Early attention processing method. Defaults to "fused_outer".
            late (str, optional): Late attention processing method. Defaults to "self".
            init (str, optional): Initialization method for interpolation. Defaults to "linear".
            guidance_scale (Optional[float], optional): Scale for classifier-free guidance. Defaults to None.
        Returns:
            numpy.ndarray: Sequence of generated images.
        """
        self.torch_device = self.unet.device
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Prepare interpolated inputs
        self.scheduler.set_timesteps(num_inference_steps)

        embs_start = self.prompt_to_embedding(prompt_start, negative_prompt)
        emb_start = embs_start[0:1]
        uncond_emb_start = embs_start[1:2]
        embs_end = self.prompt_to_embedding(prompt_end, negative_prompt)
        emb_end = embs_end[0:1]
        uncond_emb_end = embs_end[1:2]

        latent_t = slerp(latent_start, latent_end, it)
        if guide_prompt is not None:
            embs_guide = self.prompt_to_embedding(guide_prompt, negative_prompt)
            emb_t = embs_guide[0:1]
        else:
            if init == "linear":
                emb_t = torch.lerp(emb_start, emb_end, it)
            else:
                emb_t = slerp(emb_start, emb_end, it)
        if init == "linear":
            uncond_emb_t = torch.lerp(uncond_emb_start, uncond_emb_end, it)
        else:
            uncond_emb_t = slerp(uncond_emb_start, uncond_emb_end, it)

        latents = torch.cat([latent_start, latent_t, latent_end], dim=0)
        embs = torch.cat([emb_start, emb_t, emb_end], dim=0)
        uncond_embs = torch.cat([uncond_emb_start, uncond_emb_t, uncond_emb_end], dim=0)

        # Specifiy the attention processors
        pure_inner_attn_proc = InnerInterpolatedAttnProcessor(
            t=it,
            is_fused=False,
        )
        fused_inner_attn_proc = InnerInterpolatedAttnProcessor(
            t=it,
            is_fused=True,
        )
        pure_outer_attn_proc = OuterInterpolatedAttnProcessor(
            t=it,
            is_fused=False,
        )
        fused_outer_attn_proc = OuterInterpolatedAttnProcessor(
            t=it,
            is_fused=True,
        )
        self_attn_proc = AttnProcessor2_0()
        procs_dict = {
            "pure_inner": pure_inner_attn_proc,
            "fused_inner": fused_inner_attn_proc,
            "pure_outer": pure_outer_attn_proc,
            "fused_outer": fused_outer_attn_proc,
            "self": self_attn_proc,
        }

        i = 0
        warmup_step = int(num_inference_steps * warmup_ratio)
        for t in tqdm(self.scheduler.timesteps):
            i += 1
            latent_model_input = self.scheduler.scale_model_input(latents, timestep=t)
            # predict the noise residual
            with torch.no_grad():
                # Warmup
                if i < warmup_step:
                    interpolate_attn_proc = procs_dict[early]
                else:
                    interpolate_attn_proc = procs_dict[late]
                self.unet.set_attn_processor(processor=interpolate_attn_proc)
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=embs
                ).sample
                attn_proc = AttnProcessor2_0()
                self.unet.set_attn_processor(processor=attn_proc)
                noise_uncond = self.unet(
                    latent_model_input, t, encoder_hidden_states=uncond_embs
                ).sample
            # perform guidance
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the images
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        images = (image / 2 + 0.5).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return images

    def denoising_interpolate(
        self,
        latents: torch.FloatTensor,
        text_1: str,
        text_2: str,
        negative_prompt: str = "",
        interpolated_ratio: float = 1,
        timesteps: int = 25,
    ) -> np.ndarray:
        """
        Performs denoising interpolation on the given latents.

        Args:
            latents (torch.Tensor): The input latents.
            text_1 (str): The first text prompt.
            text_2 (str): The second text prompt.
            negative_prompt (str, optional): The negative text prompt. Defaults to "".
            interpolated_ratio (int, optional): The ratio of interpolation between text_1 and text_2. Defaults to 1.
            timesteps (int, optional): The number of timesteps for diffusion. Defaults to 25.

        Returns:
            numpy.ndarray: The interpolated images.
        """
        self.unet.set_attn_processor(processor=AttnProcessor2_0())
        start_emb = self.prompt_to_embedding(text_1)
        end_emb = self.prompt_to_embedding(text_2)
        neg_emb = self.prompt_to_embedding(negative_prompt)
        uncond_emb = neg_emb[0:1]
        emb_1 = start_emb[0:1]
        emb_2 = end_emb[0:1]
        self.scheduler.set_timesteps(timesteps)
        i = 0
        for t in tqdm(self.scheduler.timesteps):
            i += 1
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = self.scheduler.scale_model_input(latents, timestep=t)
            # predict the noise residual
            with torch.no_grad():
                if i < timesteps * interpolated_ratio:
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=emb_1
                    ).sample
                else:
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=emb_2
                    ).sample
                noise_uncond = self.unet(
                    latent_model_input, t, encoder_hidden_states=uncond_emb
                ).sample
            # perform guidance
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_pred - noise_uncond
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        images = (image / 2 + 0.5).clamp(0, 1)
        images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return images
