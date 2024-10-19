import argparse
import json
import os

import torch
from PIL import Image

from pipeline_interpolated_sdxl import InterpolationStableDiffusionXLPipeline
from pipeline_interpolated_sdxl import (
    InterpolationStableDiffusionXLPipeline as InterpolationStableDiffusionXLPipelineIP,
)
from prior import BetaPriorPipeline
from utils import image_grids, show_images_horizontally


def parse_args():
    parser = argparse.ArgumentParser(description="Interpolated SDXL Playground")
    parser.add_argument("--model", type=str, default="playgroundai/playground-v2.5-1024px-aesthetic")
    parser.add_argument("--prompt", type=str, default="exp/aes.json")
    parser.add_argument("--mode", type=str, default="text_to_text") # mode can be "text_to_text", "text_to_image", "image_to_image"
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dtype = torch.float16
    if args.mode == "text_to_text":
        xl_pipe = InterpolationStableDiffusionXLPipeline.from_pretrained(
            args.model, torch_dtype=dtype
        )
    elif args.mode == "text_to_image":
        xl_pipe = InterpolationStableDiffusionXLPipelineIP.from_pretrained(
            args.model, torch_dtype=dtype, variant="fp16"
        )
        xl_pipe.load_interpolated_ip_adapter("ozzygt/sdxl-ip-adapter", "", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors", early="scale_control")
    elif args.mode == "image_to_image":
        xl_pipe = InterpolationStableDiffusionXLPipelineIP.from_pretrained(
            args.model, torch_dtype=dtype, variant="fp16"
        )
        xl_pipe.load_interpolated_ip_adapter("ozzygt/sdxl-ip-adapter", "", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    xl_pipe.to("cuda")
    beta_pipe = BetaPriorPipeline(xl_pipe)

    prompt_pth = args.prompt
    with open(prompt_pth, "r") as f:
        prompt_dict = json.load(f)
    folder_name = prompt_pth.split("/")[-1].split(".")[0]
    PREFIX = prompt_dict["prefix"]
    NEGATIVE_PROMPT = prompt_dict["negative"]
    PROMPT_START_LIST = prompt_dict["start"]
    PROMPT_END_LIST = prompt_dict["end"]

    generator = torch.cuda.manual_seed(1002)
    size = xl_pipe.default_sample_size
    latent_start = torch.randn((1, 4, size, size,), device="cuda", dtype=dtype, generator=generator)
    latent_end = torch.randn((1, 4, size, size,), device="cuda", dtype=dtype, generator=generator)

    if os.path.exists(f"results/{folder_name}") is False:
        os.makedirs(f"results/{folder_name}")

    idx = 0
    interpolation_size = 7

    if args.mode == "text_to_text":
        for prompt_start, prompt_end in zip(PROMPT_START_LIST, PROMPT_END_LIST):
            prompt_a = PREFIX + prompt_start
            prompt_b = PREFIX + prompt_end
            images = beta_pipe.generate_interpolation(
                prompt_a,
                prompt_b,
                NEGATIVE_PROMPT,
                latent_end,
                latent_end,
                num_inference_steps=50,
                exploration_size=32,
                interpolation_size=interpolation_size,
                output_type="np",
                warmup_ratio=1.0
            )
            print("Coefs:", beta_pipe.xs)
            show_images_horizontally(images, f"results/{folder_name}/{idx}.jpg")
            idx += 1
    elif args.mode == "text_to_image":
        TEXT_PROMPT_START_LIST = prompt_dict["text_start"]
        for _, image_prompt_end in zip(PROMPT_START_LIST, PROMPT_END_LIST):
            image_b = Image.open(image_prompt_end)
            print(image_b)
            text_a = PREFIX + TEXT_PROMPT_START_LIST[idx]
            text_b = text_a
            images = beta_pipe.generate_interpolation(
                text_a,
                text_b,
                NEGATIVE_PROMPT,
                latent_start,
                latent_start,
                num_inference_steps=28,
                exploration_size=10,
                interpolation_size=7,
                image_start=None,
                image_end=image_b,
                output_type="np",
                warmup_ratio=0.2,
                init_alpha=1,
                init_beta=1
            )
            show_images_horizontally(images, f"results/{folder_name}/{idx}.jpg")
            idx += 1
    else:
        TEXT_PROMPT_START_LIST = prompt_dict["text_start"]
        TEXT_PROMPT_END_LIST = prompt_dict["text_end"]
        for image_prompt_start, image_prompt_end in zip(PROMPT_START_LIST, PROMPT_END_LIST):
            image_a = Image.open(image_prompt_start)
            image_b = Image.open(image_prompt_end)
            text_a = PREFIX + TEXT_PROMPT_START_LIST[idx]
            text_b = PREFIX + TEXT_PROMPT_END_LIST[idx]
            images = beta_pipe.generate_interpolation(
                text_a,
                text_b,
                NEGATIVE_PROMPT,
                latent_end,
                latent_end,
                num_inference_steps=28,
                exploration_size=15,
                interpolation_size=5,
                image_start=image_a,
                image_end=image_b,
                output_type="pil",
                warmup_ratio=0.1
            )
            images[0] = image_a.resize((1024, 1024))
            images[-1] = image_b.resize((1024, 1024))
            images.reverse()
            # show_images_horizontally(images, f"results/{folder_name}/{idx}.jpg")
            out_images = image_grids(images, rows=1, cols=5)
            out_images.save(f"results/{folder_name}/{idx}.jpg")
            idx += 1
