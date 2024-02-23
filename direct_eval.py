import os
import random
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusion import InterpolationStableDiffusionPipeline
from utils import save_image, load_and_process_images, compute_smoothness_and_efficiency, compute_fidelity, load_inception_model


def sort_source_and_interpolated(root_dir):
    source_dir = os.path.join(root_dir, "source")
    interpolated_dir = os.path.join(root_dir, "interpolated")
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(interpolated_dir):
        os.makedirs(interpolated_dir)
    pairwise_dir = os.path.join(root_dir, "pairwise")
    for pair_dir in os.listdir(pairwise_dir):
        if os.path.isfile(os.path.join(pairwise_dir, pair_dir)):
            continue
        for trial_dir in os.listdir(os.path.join(pairwise_dir, pair_dir)):
            if os.path.isfile(os.path.join(pairwise_dir, pair_dir, trial_dir)):
                continue
            size = len(os.listdir(os.path.join(pairwise_dir, pair_dir, trial_dir)))
            source_1_path = os.path.join(pairwise_dir, pair_dir, trial_dir, "1.jpg")
            source_1 = Image.open(source_1_path)
            source_1.save(os.path.join(source_dir, f"{pair_dir}_{trial_dir}_1.jpg"))
            source_2_path = os.path.join(pairwise_dir, pair_dir, trial_dir, f"{size}.jpg")
            source_2 = Image.open(source_2_path)
            source_2.save(os.path.join(source_dir, f"{pair_dir}_{trial_dir}_{size}.jpg"))
            for i in range(2, size):
                img_path = os.path.join(pairwise_dir, pair_dir, trial_dir, f"{i}.jpg")
                img = Image.open(img_path)
                img.save(os.path.join(interpolated_dir, f"{pair_dir}_{trial_dir}_{i}.jpg"))

def load_corpus(corpus_name):
    if corpus_name == "laion5b":
        corpus_name = "laion5b_meta_aes_6plus.jsonl"
        corpus_path = f"corpus/{corpus_name}"
        df = pd.read_json(path_or_buf=corpus_path, lines=True)
        captions_list = df['caption'].tolist()
        return captions_list
    elif corpus_name == "cifar100":
        list_of_strings = []
        corpus_name = 'cifar100.txt'
        corpus_path = f"corpus/{corpus_name}"
        with open(corpus_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                words_in_line = stripped_line.split(', ')
                list_of_strings = list_of_strings + words_in_line
        return list_of_strings
    else:
        raise ValueError("Invalid corpus name.")


def create_results_dir(method_name, corpus_name):
    '''
    Create the directory to save the results.
    
    Args:
    - method_name: The name of the method.
    
    Returns:
    - The directory to save the results of corresponding method.
    '''
    root_dir = "results"
    exp_dir = "direct_eval"
    method_dir = os.path.join(root_dir, exp_dir, method_name, corpus_name)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)
    return method_dir


def save_images(images, save_dir):
    '''
    Save the image as JPG.
    
    Args:
    - images: The input images as numpy array with shape (N, H, W, C).
    - save_dir: The directory to save the images.
    '''
    size = images.shape[0]
    for i in range(size):
        image = images[i]
        filename = os.path.join(save_dir, f"{i+1}.jpg")
        save_image(image, filename)


def eval_imgs(pairwise_dir):
    S = 0
    E = 0
    num_pairs = len(os.listdir(pairwise_dir))
    for pair_dir in os.listdir(pairwise_dir):
        if os.path.isfile(os.path.join(pairwise_dir, pair_dir)):
            continue
        pair_dir = os.path.join(pairwise_dir, pair_dir)
        num_tirals = len(os.listdir(pair_dir))
        for trial_dir in os.listdir(pair_dir):
            if os.path.isfile(os.path.join(pair_dir, trial_dir)):
                continue
            trial_dir = os.path.join(pair_dir, trial_dir)
            images = load_and_process_images(trial_dir)
            smoothness, efficiency = compute_smoothness_and_efficiency(images)
            print(f"Smoothness: {smoothness:.4f}, Efficiency: {efficiency:.4f}")
            S += smoothness
            E += efficiency
    num_sample = num_pairs * num_tirals
    S /= num_sample
    E /= num_sample
    eval_file = os.path.join(pairwise_dir, "eval.txt")
    with open(eval_file, 'w') as f:
        f.write(f"Smoothness: {S:.4f}, Efficiency: {E:.4f}")
    return S, E


def prepare_imgs(corpus_name="laion5b", method_name="attention", boost_ratio=0.3, early="cross", late="fused", iters=1000, trial_per_iters=5):
    # Load the corpus
    captions_list = load_corpus(corpus_name)
    
    # Load the model
    pipe = InterpolationStableDiffusionPipeline()
    
    # Initialize the generator
    vae_scale_factor = 8
    channel = pipe.unet.config.in_channels
    height = pipe.unet.config.sample_size * vae_scale_factor
    width = pipe.unet.config.sample_size * vae_scale_factor
    torch_device = "cuda"
    
    # Set the random seed
    random.seed(0)
    generator = torch.cuda.manual_seed(0)
    
    # Prepare the results directory, save each pair of images in a separate directory
    if method_name == "attention":
        right_ratio = 1 - boost_ratio
        method_name_spec = method_name + f"_boost_{boost_ratio}_{early}_{right_ratio:.1f}_{late}"
    else:
        method_name_spec = method_name
    results_dir = create_results_dir(method_name_spec, corpus_name)
    pairwise_dir = os.path.join(results_dir, "pairwise")
    os.makedirs(pairwise_dir, exist_ok=True)
    
    for _ in tqdm(range(iters)):
        prompt1, prompt2 = random.sample(captions_list, 2)
        
        # Create the directory to save the images
        pair_dir = '{' + prompt1[:20] + '}_{' + prompt2[:20] + '}'
        pair_dir = os.path.join(pairwise_dir, pair_dir)
        os.makedirs(pair_dir, exist_ok=True)
        
        prompt_txt = os.path.join(pair_dir, "prompts.txt")
        with open(prompt_txt, 'w') as f:
            f.write(prompt1 + "\n" + prompt2)
        
        for idx in range(trial_per_iters):
            trial_dir = os.path.join(pair_dir, str(idx+1))
            os.makedirs(trial_dir, exist_ok=True)
            
            latent = torch.randn(
                (1, channel, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=torch_device,
            )
            
            num_inference_steps = 25
            
            if method_name == "attention":
                images = pipe.interpolate(latent, latent, prompt1, prompt2, guide_prompt=None, size=5, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early=early, late=late)
                save_images(images, trial_dir)
            elif method_name == "embedding":
                images = pipe.interpolate(latent, latent, prompt1, prompt2, guide_prompt=None, size=5, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early="self", late="self")
                save_images(images, trial_dir)
            else:
                raise ValueError("Invalid method name.")
    
    return results_dir
            

if __name__ == "__main__":
    root_dir = prepare_imgs(corpus_name="cifar100", method_name="attention", boost_ratio=1.0, early="fused", iters=300, trial_per_iters=5)
    eval_imgs(os.path.join(root_dir, "pairwise"))
    sort_source_and_interpolated(root_dir)
    # python -m pytorch_fid path/to/source path/to/interpolated
    # 8177 attention 0.3 0.7 cross fused
    # 8133 embedding
    # 8184 attention 0.0 1.0 cross fused
    # 8185 attention 1.0 0.0 fused fused
