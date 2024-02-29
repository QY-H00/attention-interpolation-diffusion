import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
import torch
import lpips
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from torchvision.models import inception_v3
from torchvision.transforms import Normalize


def baysian_prior_selection(pipe, latent1, latent2, prompt1, prompt2, lpips_model, guide_prompt=None, size=3, num_inference_steps=25, boost_ratio=1.0, early="vfused", late="self", threshold=0.9):
    def get_smoothness(alpha, beta):
        if alpha < beta and large_alpha_prior:
            return 0
        if alpha > beta and not large_alpha_prior:
            return 0
        if alpha == beta:
            return init_smoothness
        images = pipe.interpolate_save_gpu(latent1, latent2, prompt1, prompt2, guide_prompt=guide_prompt, 
                                        size=size, num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early=early, late=late, alpha=alpha, beta=beta)
        smoothness, _, _ = compute_smoothness_and_efficiency(images, lpips_model, device="cuda")
        return smoothness
    
    # More prior on alpha and beta
    images = pipe.interpolate_single(0.5, latent1, latent2, prompt1, prompt2, guide_prompt=guide_prompt, 
                                        num_inference_steps=num_inference_steps, boost_ratio=boost_ratio, early=early, late=late)
    distances = compute_lpips(images, lpips_model)
    init_smoothness, _, _ = compute_smoothness_and_efficiency(images, lpips_model, device="cuda")
    dis_A = distances[0]
    dis_B = distances[1]
    large_alpha_prior = dis_A < dis_B

    # Bounded region of parameter space
    pbounds = {'alpha': (20, 30), 'beta': (20, 30)}
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.1)
    optimizer = BayesianOptimization(
        f=get_smoothness,
        pbounds=pbounds,
        random_state=1,
        bounds_transformer=bounds_transformer
    )
    target_score = 0.95
    n_iter = 15
    alpha_init = [20, 25, 30]
    beta_init = [20, 25, 30]
    
    # Initial probing
    for alpha in alpha_init:
        for beta in beta_init:
            optimizer.probe(params={'alpha': alpha, 'beta': beta}, lazy=False)
            print(optimizer.res)
            latest_result = optimizer.res[-1]  # Get the last result
            latest_score = latest_result['target']
            if latest_score >= target_score:
                return alpha, beta
            
    for _ in range(n_iter):  # Max iterations
        optimizer.maximize(init_points=0, n_iter=1)  # One iteration at a time
        max_score = optimizer.max['target']  # Get the highest score so far
        if max_score >= target_score:
            print(f"Stopping early, target of {target_score} reached.")
            break  # Exit the loop if target is reached or exceeded
    # optimizer.maximize(init_points=0, n_iter=15)
    results = optimizer.max
    alpha = results['params']['alpha']
    beta = results['params']['beta']
    return alpha, beta


def show_images_horizontally(list_of_files, output_file):
    '''
    Visualize the list of images horizontally and save the figure as PNG.
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
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.25)
        
        
def save_image(image, file_name):
    '''
    Save the image as JPG.
    
    Args:
    - image: The input image as numpy array with shape (H, W, C).
    - file_name: The file name to save the image.
    '''
    image = Image.fromarray(image)
    image.save(file_name)


def load_and_process_images(load_dir):
    '''
    Load and process the images into numpy array from the directory.
    
    Args:
    - load_dir: The directory to load the images.
    
    Returns:
    - images: The images as numpy array with shape (N, H, W, C).
    '''
    images = []
    filenames = sorted(os.listdir(load_dir), key=lambda x: int(x.split('.')[0]))  # Ensure the files are sorted numerically
    for filename in filenames:
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(load_dir, filename))
            img_array = np.asarray(img) / 255.0  # Convert to numpy array and scale pixel values to [0, 1]
            images.append(img_array)
    return images


def compute_lpips(images, lpips_model, device="cuda"):
    images = torch.tensor(images).to(device).float()
    images = torch.permute(images, (0, 3, 1, 2))
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
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


def compute_wasserstein_distances(images):
    '''
    Compute the Wasserstein distances between each pair of consecutive images.
    
    Args:
    - images: The input images as numpy array with shape (N, H, W, C).
    
    Returns:
    - distances: The Wasserstein distances as a list of float. (N-1, )
    '''
    distances = []
    for i in range(len(images) - 1):
        img1 = images[i].ravel()  # Flatten the image array
        img2 = images[i + 1].ravel()  # Flatten the next image array
        distance = wasserstein_distance(img1, img2)
        distances.append(distance)
    return distances


def compute_gini(distances):
    if len(distances) < 2:
        return 0.0  # Gini index is 0 for less than two elements

    # Sort the list of distances
    sorted_distances = sorted(distances)
    n = len(sorted_distances)
    mean_distance = sum(sorted_distances) / n

    # Compute the sum of absolute differences
    sum_of_differences = 0
    for i, di in enumerate(sorted_distances):
        for j, dj in enumerate(sorted_distances):
            sum_of_differences += abs(di - dj)

    # Normalize the sum of differences by the mean and the number of elements
    gini = sum_of_differences / (2 * n * n * mean_distance)
    return gini


def compute_smoothness_and_efficiency(images, lpips_model, device="cuda"):
    '''
    Compute the smoothness and efficiency of the input images.
    
    Args:
    - images: The input images as numpy array with shape (N, H, W, C).
    
    Returns:
    - smoothness: One minus gini index of LPIPS of consecutive images.
    - efficiency: Mean of LPIPS of consecutive images.
    '''
    distances = compute_lpips(images, lpips_model, device)
    smoothness = 1 - compute_gini(distances)
    avg_inception_distance = np.mean(distances)
    max_inception_distance = np.amax(distances)
    return smoothness, avg_inception_distance, max_inception_distance


def calculate_fid(act1, act2):
    '''
    Calculate the Frechet Inception Distance (FID) between two sets of activations.
    
    Args:
    - act1: The activations of the first set of images as numpy array with shape (N, D).
    - act2: The activations of the second set of images as numpy array with shape (M, D).
    
    Returns:
    - fid: The Frechet Inception Distance between the two sets of activations.
    '''
    print(act1.shape, act2.shape)
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # Compute the square root of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate FID
    print(np.sum((mu1 - mu2) ** 2))
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def load_inception_model(device="cuda"):
    '''
    Load the pretrained Inception v3 model.
    '''
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    return inception_model


def get_inception_features(images, inception_model, device="cuda"):
    '''
    Get the Inception features of the input images.
    '''
    # Convert numpy images to PyTorch tensors and normalize
    images = torch.tensor(images).to(device).float()
    images = torch.permute(images, (0, 3, 1, 2))
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
    # Resize images to fit Inception v3 size requirements
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # Use Inception model to extract features
    with torch.no_grad():
        features = inception_model(images).detach().cpu().numpy()
    return features


def seperate_source_and_interpolated_images(images):
    '''
    Seperate the input images into source and interpolated images.
    The input source is the start and end of the images, while the interpolated images are the rest.
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


def compute_fidelity(list_images, model, device="cuda"):
    '''
    Compute the Fidelity of the input images.
    '''
    source_features = None
    interpolated_features = None
    for images in list_images:
        source_images, interpolated_images = seperate_source_and_interpolated_images(images)
        if source_features is None:
            source_features = get_inception_features(source_images, model, device)
        else:
            source_features = np.concatenate((source_features, get_inception_features(source_images, model, device)))
        if interpolated_features is None:
            interpolated_features = get_inception_features(interpolated_images, model, device)
        else:
            interpolated_features = np.concatenate((interpolated_features, get_inception_features(interpolated_images, model, device)))

    fid_score = calculate_fid(source_features, interpolated_features)
    return fid_score

