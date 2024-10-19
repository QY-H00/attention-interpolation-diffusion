import os
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pipeline_interpolated_stable_diffusion import InterpolationStableDiffusionPipeline
from scipy.stats import beta as beta_distribution

from pipeline_interpolated_sdxl import InterpolationStableDiffusionXLPipeline


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
@misc{he2024aid,
      title={AID: Attention Interpolation of Text-to-Image Diffusion},
      author={Qiyuan He and Jinghao Wang and Ziwei Liu and Angela Yao},
      year={2024},
      eprint={2403.17924},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = InterpolationStableDiffusionPipeline(
    repo_name="runwayml/stable-diffusion-v1-5",
    guidance_scale=10.0,
    scheduler_name="unipc",
)
pipeline.to(device, dtype=torch.float32)


def change_model_fn(model_name: str) -> None:
    global device
    name_mapping = {
        "SD1.4-521": "CompVis/stable-diffusion-v1-4",
        "SD2.1-768": "stabilityai/stable-diffusion-2-1",
        "SDXL-1024": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    if "XL" not in model_name:
        globals()["pipeline"] = InterpolationStableDiffusionPipeline(
            repo_name=name_mapping[model_name],
            guidance_scale=10.0,
            scheduler_name="unipc",
        )
        globals()["pipeline"].to(device, dtype=torch.float32)
    else:
        if device == torch.device("cpu"):
            dtype = torch.float32
        else:
            dtype = torch.float16
        globals()["pipeline"] = InterpolationStableDiffusionXLPipeline.from_pretrained(
            name_mapping[model_name], torch_dtype=dtype
        )
        globals()["pipeline"].to(device)


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
    beta_ppf = generate_beta_tensor(size=size, alpha=int(alpha), beta=int(beta))
    return pd.DataFrame(
        {
            "interpolation index": list(range(size)),
            "coefficient": beta_ppf.tolist(),
        }
    )


def get_example() -> list[list[str | float | int]]:
    case = [
        [
            "A photo of dog, best quality, extremely detailed",
            "A photo of car, best quality, extremely detailed",
            3,
            6,
            3,
            "A car with dog furry texture, best quality, extremely detailed",
            "monochrome, lowres, bad anatomy, worst quality, low quality",
            "SD1.5-512",
            6.1 / 50,
            10,
            50,
            "fused_inner",
            "self",
            1002,
            True,
        ],
        [
            "A photo of dog, best quality, extremely detailed",
            "A photo of car, best quality, extremely detailed",
            7,
            8,
            8,
            "A toy named dog-car, best quality, extremely detailed",
            "monochrome, lowres, bad anatomy, worst quality, low quality",
            "SD1.5-512",
            8.1 / 50,
            10,
            50,
            "fused_inner",
            "self",
            1002,
            True,
        ],
        [
            "anime artwork a Pikachu sitting on the grass, dramatic, anime style, key visual, vibrant, studio anime, highly detailed",
            "anime artwork a beautiful girl, dramatic, anime style, key visual, vibrant, studio anime, highly detailed",
            7,
            10,
            6,
            None,
            "photo, photorealistic, realism, ugly, messy background",
            "SDXL-1024",
            25 / 50,
            10,
            50,
            "fused_outer",
            "self",
            1002,
            False,
        ],
        [
            "vaporwave synthwave style Los Angeles street. cyberpunk, neon, vibes, stunningly beautiful, crisp, detailed, sleek, ultramodern, high contrast, cinematic composition",
            "cinematic film still, stormtrooper taking aim. shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainyCopied!",
            7,
            530,
            602,
            None,
            "photo, photorealistic, realism, ugly, messy background",
            "SDXL-1024",
            25 / 50,
            10,
            50,
            "fused_outer",
            "self",
            1002,
            False,
        ],
    ]
    return case


def change_generate_button_fn(enable: int) -> gr.Button:
    if enable == 0:
        return gr.Button(interactive=False, value="Switching Model...")
    else:
        return gr.Button(interactive=True, value="Generate")


def dynamic_gallery_fn(interpolation_size: int):
    return gr.Gallery(
        label="Result", show_label=False, rows=1, columns=interpolation_size
    )


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
) -> np.ndarray:
    global pipeline
    generator = (
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available()
        else torch.manual_seed(seed)
    )
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
            latent_start=latent1,
            latent_end=latent2,
            prompt_start=prompt1,
            prompt_end=prompt2,
            guide_prompt=guidance_prompt,
            num_inference_steps=num_inference_steps,
            warmup_ratio=warmup_ratio,
            early=early,
            late=late,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        )
        if hasattr(images, "images"):
            # for sdxl
            images = np.array(images.images)
        if interpolation_size == 3:
            final_images = images
            break
        if i == 0:
            final_images = images[:2]
        elif i == interpolation_size - 3:
            final_images = np.concatenate([final_images, images[1:]], axis=0)
        else:
            final_images = np.concatenate([final_images, images[1:2]], axis=0)
    return final_images


interpolation_size = None

with gr.Blocks(css="style.css") as demo:
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
            value="A photo of car, best quality, extremely detaile",
        )
        result = gr.Gallery(label="Result", show_label=False, rows=1, columns=3)
    generate_button = gr.Button(value="Generate", variant="primary")
    with gr.Accordion("Advanced options", open=True):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    interpolation_size = gr.Slider(
                        label="Interpolation Size",
                        minimum=3,
                        maximum=15,
                        step=1,
                        value=3,
                        info="Interpolation size includes the start and end images",
                    )
                    alpha = gr.Slider(
                        label="alpha",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=6.0,
                    )
                    beta = gr.Slider(
                        label="beta",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=3.0,
                    )
                gamma_plot = gr.LinePlot(
                    x="interpolation index",
                    y="coefficient",
                    title="Beta Distribution with Sampled Points",
                    height=500,
                    width=400,
                    overlay_point=True,
                    tooltip=["coefficient", "interpolation index"],
                    interactive=False,
                    show_label=False,
                )
                gamma_plot.change(
                    plot_gemma_fn,
                    inputs=[
                        alpha,
                        beta,
                        interpolation_size,
                    ],
                    outputs=gamma_plot,
                )
        with gr.Group():
            guidance_prompt = gr.Text(
                label="Guidance prompt",
                max_lines=3,
                placeholder="Enter a Guidance Prompt",
                interactive=True,
                value="A photo of a dog driving a car, logical, best quality, extremely detailed",
            )
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=3,
                placeholder="Enter a Negative Prompt",
                interactive=True,
                value="monochrome, lowres, bad anatomy, worst quality, low quality",
            )
        with gr.Row():
            with gr.Column():
                warmup_ratio = gr.Slider(
                    label="Warmup Ratio",
                    minimum=0.02,
                    maximum=1,
                    step=0.01,
                    value=0.122,
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
        num_inference_steps = gr.Slider(
            label="Inference Steps",
            minimum=25,
            maximum=50,
            step=1,
            value=50,
            interactive=True,
        )
        with gr.Row():
            model_choice = gr.Dropdown(
                ["SD1.4-521", "SD1.5-512", "SD2.1-768", "SDXL-1024"],
                label="Model",
                value="SD1.5-512",
                interactive=True,
                info="SDXL will run on float16 while the rest will run on float32.",
            )
            with gr.Column():
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=1002,
                )
                same_latent = gr.Checkbox(
                    label="Same latent",
                    value=True,
                    info="Use the same latent for start and end images",
                    show_label=True,
                )

    gr.Examples(
        examples=get_example(),
        inputs=[
            prompt1,
            prompt2,
            interpolation_size,
            alpha,
            beta,
            guidance_prompt,
            negative_prompt,
            model_choice,
            warmup_ratio,
            guidance_scale,
            num_inference_steps,
            early,
            late,
            seed,
            same_latent,
        ],
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
    model_choice.change(
        fn=change_generate_button_fn,
        inputs=gr.Number(0, visible=False),
        outputs=generate_button,
    ).then(fn=change_model_fn, inputs=model_choice).then(
        fn=change_generate_button_fn,
        inputs=gr.Number(1, visible=False),
        outputs=generate_button,
    )
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
        fn=dynamic_gallery_fn,
        inputs=interpolation_size,
        outputs=result,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    gr.Markdown(article)

demo.launch()
