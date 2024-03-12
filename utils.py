import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Normalize
from lpips import LPIPS
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from diffusion import InterpolationStableDiffusionPipeline


def baysian_prior_selection(
    interpolation_pipe: InterpolationStableDiffusionPipeline,
    latent1: torch.FloatTensor,
    latent2: torch.FloatTensor,
    prompt1: str,
    prompt2: str,
    lpips_model: LPIPS,
    guide_prompt: str | None = None,
    negative_prompt: str = "",
    size: int = 3,
    num_inference_steps: int = 25,
    warmup_ratio: float = 1,
    early: str = "vfused",
    late: str = "self",
    target_score: float = 0.9,
    n_iter: int = 15,
    p_min: float | None = None,
    p_max: float | None = None
) -> tuple:
    '''
    Select the alpha and beta parameters for the interpolation using Bayesian optimization.
    
    Args:
        interpolation_pipe (any): The interpolation pipeline.
        latent1 (torch.FloatTensor): The first source latent vector.
        latent2 (torch.FloatTensor): The second source latent vector.
        prompt1 (str): The first source prompt.
        prompt2 (str): The second source prompt.
        lpips_model (any): The LPIPS model used to compute perceptual distances.
        guide_prompt (str | None, optional): The guide prompt for the interpolation, if any. Defaults to None.
        negative_prompt (str, optional): The negative prompt for the interpolation, default to empty string. Defaults to "".
        size (int, optional): The size of the interpolation sequence. Defaults to 3.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 25.
        warmup_ratio (float, optional): The warmup ratio. Defaults to 1.
        early (str, optional): The early fusion method. Defaults to "vfused".
        late (str, optional): The late fusion method. Defaults to "self".
        target_score (float, optional): The target score. Defaults to 0.9.
        n_iter (int, optional): The maximum number of iterations. Defaults to 15.
        p_min (float, optional): The minimum value of alpha and beta. Defaults to None.
        p_max (float, optional): The maximum value of alpha and beta. Defaults to None.
    Returns:
        tuple: A tuple containing the selected alpha and beta parameters.
    '''

    def get_smoothness(alpha, beta):
        '''
        Black-box objective function of Baysian Optimization.
        Get the smoothness of the interpolated sequence with the given alpha and beta.
        '''
        if alpha < beta and large_alpha_prior:
            return 0
        if alpha > beta and not large_alpha_prior:
            return 0
        if alpha == beta:
            return init_smoothness
        interpolation_sequence = interpolation_pipe.interpolate_save_gpu(
            latent1,
            latent2,
            prompt1,
            prompt2,
            guide_prompt=guide_prompt,
            negative_prompt=negative_prompt,                
            size=size,
            num_inference_steps=num_inference_steps,
            warmup_ratio=warmup_ratio,
            early=early,
            late=late,
            alpha=alpha,
            beta=beta
        )
        smoothness, _, _ = compute_smoothness_and_consistency(interpolation_sequence, lpips_model)
        return smoothness
   
    # Add prior into selection of alpha and beta
    # We firstly compute the interpolated images with t=0.5
    images = interpolation_pipe.interpolate_single(
        0.5,
        latent1,
        latent2,
        prompt1,
        prompt2,
        guide_prompt=guide_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        warmup_ratio=warmup_ratio,
        early=early,
        late=late
    )
    # We compute the perceptual distances of the interpolated images (t=0.5) to the source image
    distances = compute_lpips(images, lpips_model)
    # We compute the init_smoothness as the smoothness when alpha=beta to avoid recomputation
    init_smoothness, _, _ = compute_smoothness_and_consistency(images, lpips_model)
    # If perceptual distance to the first source image is smaller, alpha should be larger than beta
    large_alpha_prior = distances[0] < distances[1]

    # Baysian optimization configuration
    num_warmup_steps = warmup_ratio * num_inference_steps
    if p_min is None:
        p_min = 1
    if p_max is None:
        p_max = num_warmup_steps
    pbounds = {'alpha': (p_min, p_max), 'beta': (p_min, p_max)}
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.1)
    optimizer = BayesianOptimization(
        f=get_smoothness,
        pbounds=pbounds,
        random_state=1,
        bounds_transformer=bounds_transformer,
        allow_duplicate_points=True
    )
    alpha_init = [p_min, (p_min + p_max) / 2, p_max]
    beta_init = [p_min, (p_min + p_max) / 2, p_max]
  
    # Initial probing
    for alpha in alpha_init:
        for beta in beta_init:
            optimizer.probe(params={'alpha': alpha, 'beta': beta}, lazy=False)
            latest_result = optimizer.res[-1]  # Get the last result
            latest_score = latest_result['target']
            if latest_score >= target_score:
                return alpha, beta

    # Start optimization
    for _ in range(n_iter):  # Max iterations
        optimizer.maximize(init_points=0, n_iter=1)  # One iteration at a time
        max_score = optimizer.max['target']  # Get the highest score so far
        if max_score >= target_score:
            print(f"Stopping early, target of {target_score} reached.")
            break  # Exit the loop if target is reached or exceeded
   
    results = optimizer.max
    alpha = results['params']['alpha']
    beta = results['params']['beta']
    return alpha, beta


def show_images_horizontally(
    list_of_files: np.array,
    output_file: str | None = None,
    interact: bool = False
    ) -> None:
    '''
    Visualize the list of images horizontally and save the figure as PNG.
    
    Args:
        list_of_files: The list of images as numpy array with shape (N, H, W, C).
        output_file: The output file path to save the figure as PNG.
        interact: Whether to show the figure interactively in Jupyter Notebook or not in Python.
    '''
    number_of_files = len(list_of_files)

    heights = [a[0].shape[0] for a in list_of_files]
    widths = [a.shape[1] for a in list_of_files[0]]

    fig_width = 8.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)

    # Create a figure with subplots
    _, axs = plt.subplots(1, number_of_files, figsize=(fig_width * number_of_files, fig_height))
    plt.tight_layout()
    for i in range(number_of_files):
        _image = list_of_files[i]
        axs[i].imshow(_image)
        axs[i].axis("off")
        
    # Save the figure as PNG
    if interact:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.25)
        
        
def save_image(image: np.array, file_name: str) -> None:
    '''
    Save the image as JPG.
    
    Args:
        image: The input image as numpy array with shape (H, W, C).
        file_name: The file name to save the image.
    '''
    image = Image.fromarray(image)
    image.save(file_name)


def load_and_process_images(load_dir: str) -> np.array:
    '''
    Load and process the images into numpy array from the directory.
    
    Args:
        load_dir: The directory to load the images.
    
    Returns:
        images: The images as numpy array with shape (N, H, W, C).
    '''
    images = []
    print(load_dir)
    filenames = sorted(os.listdir(load_dir), key=lambda x: int(x.split('.')[0]))  # Ensure the files are sorted numerically
    for filename in filenames:
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(load_dir, filename))
            img_array = np.asarray(img) / 255.0  # Convert to numpy array and scale pixel values to [0, 1]
            images.append(img_array)
    return images


def compute_lpips(images: np.array, lpips_model: LPIPS) -> np.array:
    '''
    Compute the LPIPS of the input images.
    
    Args:
        images: The input images as numpy array with shape (N, H, W, C).
        lpips_model: The LPIPS model used to compute perceptual distances.
        
    Returns:
        distances: The LPIPS of the input images.
    '''
    # Get device of lpips_model
    device = next(lpips_model.parameters()).device
    device = str(device)
    
    # Change the input images into tensor
    images = torch.tensor(images).to(device).float()
    images = torch.permute(images, (0, 3, 1, 2))
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
    
    # Compute the LPIPS between each adjacent input images
    distances = []
    for i in range(images.shape[0]):
        if i == images.shape[0] - 1:
            break
        img1 = images[i].unsqueeze(0)
        img2 = images[i + 1].unsqueeze(0)
        loss = lpips_model(img1, img2)
        distances.append(loss.item())
    distances = np.array(distances)
    return distances


def compute_gini(distances: np.array) -> float:
    '''
    Compute the Gini index of the input distances.
    
    Args:
        distances: The input distances as numpy array.
    
    Returns:
        gini: The Gini index of the input distances.
    '''
    if len(distances) < 2:
        return 0.0  # Gini index is 0 for less than two elements

    # Sort the list of distances
    sorted_distances = sorted(distances)
    n = len(sorted_distances)
    mean_distance = sum(sorted_distances) / n

    # Compute the sum of absolute differences
    sum_of_differences = 0
    for di in sorted_distances:
        for dj in sorted_distances:
            sum_of_differences += abs(di - dj)

    # Normalize the sum of differences by the mean and the number of elements
    gini = sum_of_differences / (2 * n * n * mean_distance)
    return gini


def compute_smoothness_and_consistency(
    images: np.array,
    lpips_model: LPIPS
    ) -> tuple:
    '''
    Compute the smoothness and efficiency of the input images.
    
    Args:
        images: The input images as numpy array with shape (N, H, W, C).
        lpips_model: The LPIPS model used to compute perceptual distances.
    
    Returns:
        smoothness: One minus gini index of LPIPS of consecutive images.
        consistency: The mean LPIPS of consecutive images.
        max_inception_distance: The maximum LPIPS of consecutive images.
    '''
    distances = compute_lpips(images, lpips_model)
    smoothness = 1 - compute_gini(distances)
    consistency = np.mean(distances)
    max_inception_distance = np.max(distances)
    return smoothness, consistency, max_inception_distance


def seperate_source_and_interpolated_images(images: np.array) -> tuple:
    '''
    Seperate the input images into source and interpolated images.
    The input source is the start and end of the images, while the interpolated images are the rest.
    
    Args:
        images: The input images as numpy array with shape (N, H, W, C).
    
    Returns:
        source: The source images as numpy array with shape (2, H, W, C).
        interpolation: The interpolated images as numpy array with shape (N-2, H, W, C).
    '''
    # Check if the array has at least two elements
    if len(images) < 2:
        raise ValueError("The input array should have at least two elements.")

    # Separate the array into two parts
    # First part takes the first and last element
    source = np.array([images[0], images[-1]])
    # Second part takes the rest of the elements
    interpolation = images[1:-1]
    return source, interpolation
