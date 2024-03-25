import os
import random
import uuid
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
import user_history
from PIL import Image
from scipy.stats import beta as beta_distribution

from pipeline_interpolated_stable_diffusion import InterpolationStableDiffusionPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

title = r"""
<h1 align="center">PAID: (Prompt-guided) Attention Interpolation of Text-to-Image Diffusion</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/QY-H00/attention-interpolation-diffusion/tree/public' target='_blank'><b>PAID: (Prompt-guided) Attention Interpolation of Text-to-Image Diffusion</b></a>.<br>
How to use:<br>
1. Input prompt 1 and prompt 2. 
2. (Optional) Input the guidance prompt and negative prompt.
3. (Optional) Change the interpolation parameters and check the Beta distribution.
4. Click the <b>Generate</b> button to begin generating images.
5. Enjoy! 😊"""

article = r"""
---
✒️ **Citation**
<br>
If you found this demo/our paper useful, please consider citing:
```bibtex
@article{he024paid,
    title={PAID},
    author={He, Qiyuan and Wang, Jinghao and Liu, Ziwei and Angle, Yao},
    journal={},
    year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue in our <a href='https://github.com/QY-H00/attention-interpolation-diffusion/tree/public' target='_blank'><b>Github Repo</b></a> or directly reach us out at <b>qhe@u.nus.edu.sg</b>.
"""

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = False
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"
PREVIEW_IMAGES = False

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = InterpolationStableDiffusionPipeline(
    repo_name="runwayml/stable-diffusion-v1-5",
    guidance_scale=10.0,
    scheduler_name="unipc",
)
pipeline.to(device, dtype=dtype)


def change_model_fn(model_name: str) -> None:
    global pipeline
    name_mapping = {
        "SD1.4-521": "CompVis/stable-diffusion-v1-4",
        "SD1.5-512": "runwayml/stable-diffusion-v1-5",
        "SD2.1-768": "stabilityai/stable-diffusion-2-1",
        "SDXL-1024": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    if "XL" not in model_name:
        pipeline = InterpolationStableDiffusionPipeline(
            repo_name=name_mapping[model_name],
            guidance_scale=10.0,
            scheduler_name="unipc",
        ).to(device, dtype=dtype)
    else:
        pipeline = InterpolationStableDiffusionPipeline.from_pretrained(
            name_mapping[model_name], torch_dtype=dtype
        ).to(device, dtype=dtype)


def save_image(img, index):
    unique_name = f"{index}.png"
    img = Image.fromarray(img)
    img.save(unique_name)
    return unique_name


def generate_beta_tensor(
    size: int, alpha: float = 3.0, beta: float = 3.0
) -> torch.FloatTensor:
    prob_values = [i / (size - 1) for i in range(size)]
    inverse_cdf_values = beta_distribution.ppf(prob_values, alpha, beta)
    return inverse_cdf_values


def plot_gemma_fn(alpha: float, beta: float, size: int) -> pd.DataFrame:
    beta_ppf = generate_beta_tensor(size=size, alpha=alpha, beta=beta)
    return pd.DataFrame(
        {
            "interpolation index": [i for i in range(size)],
            "coefficient": beta_ppf.tolist(),
        }
    )


def get_example() -> list:
    case = [
        [
            "./examples/yann-lecun_resize.jpg",
            None,
            "a man",
            "Spring Festival",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/musk_resize.jpeg",
            "./examples/poses/pose2.jpg",
            "a man flying in the sky in Mars",
            "Mars",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/sam_resize.png",
            "./examples/poses/pose4.jpg",
            "a man doing a silly pose wearing a suite",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            "./examples/schmidhuber_resize.png",
            "./examples/poses/pose3.jpg",
            "a man sit on a chair",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/kaifu_resize.png",
            "./examples/poses/pose.jpg",
            "a man",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]
    return case


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    print("randomizing seed")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.no_grad()
def generate(
    prompt1: str,
    prompt2: str,
    guidance_prompt: Optional[str] = None,
    negative_prompt: str = "",
    warmup_ratio: int = 8,
    guidance_scale: float = 10,
    early: str = "fused_outer",
    late: str = "self",
    alpha: float = 4.0,
    beta: float = 4.0,
    interpolation_size: int = 3,
    seed: int = 0,
    same_latent: bool = True,
    num_inference_steps: int = 50,
    progress=gr.Progress(),
):
    global pipeline
    generator = torch.Generator().manual_seed(seed)
    latent1 = pipeline.generate_latent(generator=generator)
    latent1 = latent1.to(device=pipeline.unet.device, dtype=pipeline.unet.dtype)
    if same_latent:
        latent2 = latent1.clone()
    else:
        latent2 = pipeline.generate_latent(generator=generator)
        latent2 = latent2.to(device=pipeline.unet.device, dtype=pipeline.unet.dtype)
    betas = generate_beta_tensor(size=interpolation_size, alpha=alpha, beta=beta)
    for i in progress.tqdm(
        range(interpolation_size - 2),
        desc=(
            f"Generating {interpolation_size-2} images"
            if interpolation_size > 3
            else "Generating 1 image"
        ),
    ):
        it = betas[i + 1].item()
        images = pipeline.interpolate_single(
            it,
            latent1,
            latent2,
            prompt1,
            prompt2,
            guide_prompt=guidance_prompt,
            num_inference_steps=num_inference_steps,
            warmup_ratio=warmup_ratio,
            early=early,
            late=late,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        )
        if interpolation_size == 3:
            final_images = images
            break
        if i == 0:
            final_images = images[:2]
        elif i == interpolation_size - 3:
            final_images = np.concatenate([final_images, images[1:]], axis=0)
        else:
            final_images = np.concatenate([final_images, images[1:2]], axis=0)
    # Save images
    # for image in output:
    #     user_history.save_image(
    #         profile=profile,
    #         image=image,
    #         label=prompt1 + " " + prompt2,
    #         metadata={
    #             "negative_prompt": negative_prompt,
    #             "seed": seed,
    #             "width": width,
    #             "height": height,
    #             "prior_guidance_scale": prior_guidance_scale,
    #             "decoder_num_inference_steps": decoder_num_inference_steps,
    #             "decoder_guidance_scale": decoder_guidance_scale,
    #             "interpolation_size": interpolation_size,
    #         },
    #     )
    uuids = str(uuid.uuid4())
    image_paths = [
        save_image(img, uuids + f"{index}") for index, img in enumerate(final_images)
    ]
    return image_paths


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Group():
        prompt1 = gr.Text(
            label="Prompt 1",
            max_lines=3,
            placeholder="Enter the First Prompt",
            interactive=True,
            value="A photo of dog, best quality, extremely detailed",
        )
        prompt2 = gr.Text(
            label="Prompt 2",
            max_lines=3,
            placeholder="Enter the Second prompt",
            interactive=True,
            value="A photo of cat, best quality, extremely detailed",
        )
        result = gr.Gallery(label="Result", show_label=False, rows=1)
    with gr.Accordion("Advanced options", open=True):
        with gr.Group():
            with gr.Column():
                interpolation_size = gr.Slider(
                    label="Interpolation Size",
                    minimum=3,
                    maximum=20,
                    step=1,
                    value=3,
                    info="Interpolation size includes the start and end images",
                )
                alpha = gr.Slider(
                    label="alpha",
                    minimum=0,
                    maximum=50,
                    step=0.1,
                    value=4.0,
                )
                beta = gr.Slider(
                    label="beta",
                    minimum=0,
                    maximum=50,
                    step=0.1,
                    value=4.0,
                )
            gamma_plot = gr.LinePlot(
                x="interpolation index",
                y="coefficient",
                title="Beta Distribution with Sampled Points",
                height=400,
                width=900,
                overlay_point=True,
                tooltip=["coefficient", "interpolation index"],
                interactive=False,
                show_label=False,
            )
        with gr.Group():
            guidance_prompt = gr.Text(
                label="Guidance prompt",
                max_lines=3,
                placeholder="Enter a Guidance Prompt",
                interactive=True,
            )
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=3,
                placeholder="Enter a Negative Prompt",
                interactive=True,
                value="monochrome, lowres, bad anatomy, worst quality, low quality",
            )
        with gr.Row():
            model_choice = gr.Dropdown(
                ["SD1.4-521", "SD1.5-512", "SD2.1-768", "SDXL-1024"],
                label="Model",
                value="SD1.5-512",
                interactive=True,
            )
        with gr.Row():
            warmup_ratio = gr.Slider(
                label="Warmup Ratio",
                minimum=0.02,
                maximum=1,
                step=0.01,
                value=0.16,
                interactive=True,
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0,
                maximum=50,
                step=0.1,
                value=10,
                interactive=True,
            )
        num_inference_steps = gr.Slider(
            label="Inference Steps",
            minimum=25,
            maximum=50,
            step=1,
            value=50,
            interactive=True,
        )
        with gr.Row():
            with gr.Column():
                early = gr.Dropdown(
                    label="Early stage attention type",
                    choices=[
                        "pure_inner",
                        "fused_inner",
                        "pure_outer",
                        "fused_outer",
                        "self",
                    ],
                    value="fused_outer",
                    type="value",
                    interactive=True,
                )
                late = gr.Dropdown(
                    label="Late stage attention type",
                    choices=[
                        "pure_inner",
                        "fused_inner",
                        "pure_outer",
                        "fused_outer",
                        "self",
                    ],
                    value="self",
                    type="value",
                    interactive=True,
                )
            with gr.Column():
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                same_latent = gr.Checkbox(
                    label="Same latent",
                    value=True,
                    info="Use the same latent for start and end images",
                )
    generate_button = gr.Button("Generate", variant="primary")
    gr.Examples(
        examples=get_example(),
        inputs=[
            prompt1,
            prompt2,
            guidance_prompt,
            negative_prompt,
            warmup_ratio,
            guidance_scale,
            early,
            late,
            alpha,
            beta,
            interpolation_size,
            seed,
            same_latent,
            num_inference_steps,
        ],
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    alpha.change(
        fn=plot_gemma_fn, inputs=[alpha, beta, interpolation_size], outputs=gamma_plot
    )
    beta.change(
        fn=plot_gemma_fn, inputs=[alpha, beta, interpolation_size], outputs=gamma_plot
    )
    interpolation_size.change(
        fn=plot_gemma_fn, inputs=[alpha, beta, interpolation_size], outputs=gamma_plot
    )
    model_choice.change(fn=change_model_fn, inputs=[model_choice], outputs=None)

    inputs = [
        prompt1,
        prompt2,
        guidance_prompt,
        negative_prompt,
        warmup_ratio,
        guidance_scale,
        early,
        late,
        alpha,
        beta,
        interpolation_size,
        seed,
        same_latent,
        num_inference_steps,
    ]
    generate_button.click(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    gr.Markdown(article)

with gr.Blocks(css="style.css") as demo_with_history:
    with gr.Tab("App"):
        demo.render()
    with gr.Tab("Past generations"):
        user_history.render()

if __name__ == "__main__":
    demo_with_history.queue(max_size=20).launch()
