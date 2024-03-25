import os
import random
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
import user_history
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
4. Click the <b>Submit</b> button to begin customization.
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

dtype = torch.bfloat16
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
pipeline = InterpolationStableDiffusionPipeline(
    repo_name="runwayml/stable-diffusion-v1-5",
    guidance_scale=10.0,
    scheduler_name="unipc",
).to(device, dtype=dtype)


def change_model_fn(model_name: str) -> None:
    global pipeline
    name_mapping = {
        "SD1.4-521": "CompVis/stable-diffusion-v1-4",
        "SD1.5-512": "runwayml/stable-diffusion-v1-5",
        "SD2.1-768": "stabilityai/stable-diffusion-2-1",
        "SDXL-1024": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    pipeline = InterpolationStableDiffusionPipeline(
        repo_name=name_mapping[model_name],
        guidance_scale=10.0,
        scheduler_name="unipc",
    ).to(device, dtype=dtype)


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


def generate(
    prompt1: str,
    prompt2: str,
    negative_prompt: str = "",
    guidance_prompt: Optional[str] = None,
    seed: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    print("prior_num_inference_steps: ", prior_num_inference_steps)
    prior_output = prior_pipeline(
        prompt1=prompt1,
        prompt2=prompt2,
        height=height,
        width=width,
        num_inference_steps=prior_num_inference_steps,
        negative_prompt=negative_prompt,
        guidance_scale=prior_guidance_scale,
        interpolation_size=interpolation_size,
        generator=generator,
    )

    decoder_output = decoder_pipeline(
        image_embeddings=prior_output.image_embeddings,
        prompt1=prompt1,
        prompt2=prompt2,
        num_inference_steps=decoder_num_inference_steps,
        guidance_scale=decoder_guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="pil",
    ).images
    print(decoder_output)
    # Save images
    for image in decoder_output:
        user_history.save_image(
            profile=profile,
            image=image,
            label=prompt1 + " " + prompt2,
            metadata={
                "negative_prompt": negative_prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "prior_guidance_scale": prior_guidance_scale,
                "decoder_num_inference_steps": decoder_num_inference_steps,
                "decoder_guidance_scale": decoder_guidance_scale,
                "interpolation_size": interpolation_size,
            },
        )

    yield decoder_output[0]


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Group():
        prompt1 = gr.Text(
            label="Prompt 1",
            max_lines=3,
            placeholder="Enter the First Prompt",
            interactive=True,
        )
        prompt2 = gr.Text(
            label="Prompt 2",
            max_lines=3,
            placeholder="Enter the Second prompt",
            interactive=True,
        )
        result = gr.Image(label="Result", show_label=False)
    with gr.Accordion("Advanced options", open=True):
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
        )
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
                    maximum=20,
                    step=0.1,
                    value=4.0,
                )
                beta = gr.Slider(
                    label="beta",
                    minimum=0,
                    maximum=30,
                    step=0.1,
                    value=4.0,
                )
            gamma_plot = gr.LinePlot(
                x="interpolation index",
                y="coefficient",
                title="Beta Distribution with Sampled Points",
                height=400,
                width=300,
                overlay_point=True,
                tooltip=["coefficient", "interpolation index"],
                interactive=False,
            )
        with gr.Row():
            model_choice = gr.Dropdown(
                ["SD1.4-521", "SD1.5-512", "SD2.1-768", "SDXL-1024"],
                label="Model",
                value="SD1.5-512",
                interactive=True,
            )
        with gr.Row():
            warmup_step = gr.Slider(
                label="Warmup Step",
                minimum=1,
                maximum=50,
                step=1,
                value=8,
                interactive=True,
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0,
                maximum=50,
                step=0.1,
                value=7.5,
                interactive=True,
            )
        with gr.Row():
            late = gr.Dropdown(
                label="Late stage attention",
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
            early = gr.Dropdown(
                label="Early stage attention",
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
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
    submit_button = gr.Button("Submit", variant="primary")
    gr.Examples(
        examples=get_example(),
        inputs=[prompt1, prompt2, guidance_prompt, negative_prompt],
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
        seed,
        alpha,
        beta,
        interpolation_size,
    ]
    gr.on(
        triggers=[
            submit_button.click,
        ],
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )
    gr.Markdown(article)

with gr.Blocks(css="style.css") as demo_with_history:
    with gr.Tab("App"):
        demo.render()
    with gr.Tab("Past generations"):
        user_history.render()

if __name__ == "__main__":
    demo_with_history.queue(max_size=20).launch()
