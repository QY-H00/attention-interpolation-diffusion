import os
import random
import torch
from tqdm.auto import tqdm
from diffusion import InterpolationStableDiffusionPipeline
from utils import save_image, load_and_process_images, compute_smoothness_and_efficiency, compute_fidelity, load_inception_model, sort_source_and_interpolated


def create_results_dir(method_name):
    '''
    Create the directory to save the results.
    
    Args:
    - method_name: The name of the method.
    
    Returns:
    - The directory to save the results of corresponding method.
    '''
    root_dir = "results"
    exp_dir = "direct_eval"
    method_dir = os.path.join(root_dir, exp_dir, method_name)
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


def eval_imgs(direct_eval_dir):
    S = 0
    E = 0
    num_pairs = len(os.listdir(direct_eval_dir))
    for pair_dir in os.listdir(direct_eval_dir):
        pair_dir = os.path.join(direct_eval_dir, pair_dir)
        num_tirals = len(os.listdir(pair_dir))
        for trial_dir in os.listdir(pair_dir):
            trial_dir = os.path.join(pair_dir, trial_dir)
            images = load_and_process_images(trial_dir)
            smoothness, efficiency = compute_smoothness_and_efficiency(images)
            print(f"Smoothness: {smoothness:.4f}, Efficiency: {efficiency:.4f}")
            S += smoothness
            E += efficiency
    num_sample = num_pairs * num_tirals
    S /= num_sample
    E /= num_sample
    print(f"Average Smoothness: {S:.4f}, Average Efficiency: {E:.4f}")
    return S, E


def prepare_imgs(corpus_name="laion5b", method_name="attention", boost_ratio=0.3, early="cross", late="fused", iters=1000, trial_per_iters=5):
    # Load the corpus
    corpus = load_corpus(corpus_name)
    
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
        method_name = method_name + f"_boost_{boost_ratio}_{early}_{1-boost_ratio}_{late}"
    results_dir = create_results_dir(method_name)
    pairwise_dir = os.path.join(results_dir, "pairwise")
    os.makedirs(pairwise_dir, exist_ok=True)
    
    for _ in tqdm(range(iters)):
        prompt1, prompt2 = random.sample(corpus, 2)
        
        # Create the directory to save the images
        pair_dir = '{' + prompt1 + '}_{' + prompt2 + '}'
        pair_dir = os.path.join(pairwise_dir, pair_dir)
        os.makedirs(pair_dir, exist_ok=True)
        
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
            

if __name__ == "__main__":
    corpus = [
        "a man",
        "a lion",
        "a banana",
        "a cake",
        "a watermelon",
        "a cat",
        "a dog"]
    # prepare_imgs(corpus, 3, 5)
    # eval_imgs("results/direct_eval")
    sort_source_and_interpolated()

    