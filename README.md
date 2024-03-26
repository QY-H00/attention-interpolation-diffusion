<p align="center">
  <img src="asset/logo.png"  height=350>
</p>

### <div align="center">PAID: (Prompt-guided) Attention Interpolation of Text-to-Image Diffusion<div>

<div align="center">
<a><img herf=https://arxiv.org/abs/xxxx src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=25px></a>
<a herf=https://huggingface.co/spaces/king159/PAID><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Space-276cb4.svg height=25px></a>
<a herf=https://colab.research.google.com/drive/1zC-iOVu_raiFdUAD-TQ76GPKAuIj4hIt?usp=sharing><img src= https://img.shields.io/badge/Google%20Colab-8f2628.svg?logo=googlecolab height=25px></a>
<a herf=https://qy-h00.github.io/attention-interpolation-diffusion/><img src= https://img.shields.io/badge/GitHub%20Project%20Page-bb8a2e.svg?logo=github height=25px></a>
</div>

<p align="center">
  <br>
  <a href="https://qy-h00.github.io" target="_blank">He Qiyuan</a><sup>1</sup>,&nbsp;
  <a href="https://king159.github.io/" target="_blank">Wang Jinghao</a><sup>2</sup>,&nbsp;
  <a href="https://liuziwei7.github.io/" target="_blank">Liu Ziwei</a><sup>2</sup>,&nbsp;
  <a href="https://www.comp.nus.edu.sg/~ayao//" target="_blank">Angela Yao</a><sup>1,&#x2709</sup>;
  </sup></a>
  <br>
  <a herf=https://cvml.comp.nus.edu.sg>Computer Vision & Machine Learning Group, National University of Singapore</a> <sup>1</sup>
  <br>
  S-Lab, Nanyang Technological University <sup>2</sup>
  <br>
  <sup>&#x2709;</sup> Corresponding Author
</p>

## 📌 Release

[03/2024] Code and paper are publicly available.

## 📑 Abstract

<b>TL;DR: <font color="red">AID</font> (Attention Interpolation via Diffusion)</b> is the first method that enables the text-to-image diffusion model to generate interpolation between different conditions with high consistency, smoothness and fidelity. Its variant, <font color="blue">PAID</font>, provides further control of the interpolation via prompt guidance.

## ▶️ PAID Results

<p align="center">
<img src="sdxl_example/1.png">
</p>

<p align="center">
<img src="sdxl_example/2.png">
</p>

<p align="center">
<img src="sdxl_example/3.png">
</p>

<p align="center">
<img src="sdxl_example/4.png">
</p>

<p align="center">
<img src="sdxl_example/5.png">
</p>

<p align="center">
<img src="sdxl_example/6.png">
</p>

<p align="center">
<img src="sdxl_example/7.png">
</p>

<p align="center">
<img src="sdxl_example/8.png">
</p>

<p align="center">
<img src="sdxl_example/9.png">
</p>

## 🏍️ Google Colab

Directly try PAID [here](https://colab.research.google.com/drive/1zC-iOVu_raiFdUAD-TQ76GPKAuIj4hIt?usp=sharing) using Google's Free GPU!

## 🚗 Local Setup using Jupyter Notebook

1. Clone the repository and install the requirements:

``` bash
git clone https://github.com/QY-H00/attention-interpolation-diffusion.git
cd attention-interpolation-diffusion
pip install requirements.txt
```

2. Go to `play.ipynb` and `play_sdxl.ipynb` for fun!

## 🛳️ Local Setup using Gradio

1. install Gradio

``` bash
pip install gradio
```

2. Launch the Gradio interface

``` bash
gradio gradio_src/app.py
```

## 🎲 Customized Interpolation

Our method offers users customized and diverse configurations to experiment with, allowing them to freely adjust settings and achieve a wide range of interesting interpolation results. Here are some examples:

### Prompt guidance

#### 1. "A dog driving car"

<p align="center">
<img src="example/dog_car_1.png">
</p>

#### 2. "A car with dog furry texture"

<p align="center">
<img src="example/dog_car_2.png">
</p>

#### 3. "A toy named dog-car"

<p align="center">
<img src="example/dog_car_3.png">
</p>

#### 4. "A painting of car and dog drawn by Vincent van Gogh"

<p align="center">
<img src="example/dog_car_4.png">
</p>

### $\alpha$ and $\beta$ of the Beta prior

#### 1. $\alpha=1, \beta=1$

<p align="center">
<img src="example/shark_fox_1.png">
</p>

#### 2. $\alpha=1, \beta=8$

<p align="center">
<img src="example/shark_fox_2.png">
</p>

#### 3. $\alpha=8, \beta=1$

<p align="center">
<img src="example/shark_fox_3.png">
</p>

## 📝 Supporting Models

| Model Name            |  Link                                             |
|-----------------------|-------------------------------------------------------------|
| Stable Diffusion 1.4-512  | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)   |
| Stable Diffusion 1.5-512  | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| Stable Diffusion 2.1-768  | [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| Stable Diffusion XL-1024   | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
|Animagine XL 3.1 |   [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)|

## ✒️Citation

If you found this repository/our paper useful, please consider citing:

``` bibtex
@article{he2024paid,
  title={PAID:(Prompt-guided) Attention Interpolation of Text-to-Image},
  author={He, Qiyuan and Wang, Jinghao and Liu, Ziwei and Angle, Yao},
  journal={},
  year={2024}
}
```

## ❤️ Acknowledgement

We thank the following repositories for their great work: [diffusers](https://github.com/huggingface/diffusers), [transformers](https://github.com/huggingface/transformers).

## ➕️ More Results with SD1.5

### Realist Style

<p align="center">
Pikachu -> Gundam
<img src="example/pikachu_gundam.png">
</p>

<p align="center">
Computer -> Phone
<img src="example/computer_phone.png">
</p>

### Anime Style

<p align="center">
Ninja -> Cat
<img src="example/ninja_cat.png">
</p>

<p align="center">
Ninja -> Dog
<img src="example/ninja_dog.png">
</p>

### Oil-Painting Style

<p align="center">
Starry night -> Mona Lisas
<img src="example/starry_mona.png">
</p>

<p align="center">
SkyCraper -> Town
<img src="example/skycraper_town.png">
</p>