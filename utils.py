import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
import torch
from torchvision.models import inception_v3
from torchvision.transforms import Normalize


def show_images_horizontally(list_of_files, output_file):
    '''
    Visualize the list of images horizontally and save the figure as PNG.
    '''
    number_of_files = len(list_of_files)

    # Create a figure with subplots
    _, axs = plt.subplots(1, number_of_files, figsize=(10, 5))

    for i in range(number_of_files):
        _image = list_of_files[i]
        axs[i].imshow(_image)
        axs[i].axis("off")
        
    # Save the figure as PNG
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.1)
    
    
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
        image = Image.fromarray(image)
        image.save(filename)


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


def compute_smoothness_and_efficiency(images):
    '''
    Compute the smoothness and efficiency of the input images.
    
    Args:
    - images: The input images as numpy array with shape (N, H, W, C).
    
    Returns:
    - smoothness: Variance of the Wasserstein distances of consecutive images.
    - efficiency: Mean of the Wasserstein distances of consecutive images.
    '''
    wasserstein_distances = compute_wasserstein_distances(images)
    smoothness = np.mean(wasserstein_distances)
    efficiency = np.var(wasserstein_distances)
    return smoothness, efficiency


def calculate_fid(act1, act2):
    '''
    Calculate the Frechet Inception Distance (FID) between two sets of activations.
    
    Args:
    - act1: The activations of the first set of images as numpy array with shape (N, D).
    - act2: The activations of the second set of images as numpy array with shape (M, D).
    
    Returns:
    - fid: The Frechet Inception Distance between the two sets of activations.
    '''
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # Compute the square root of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate FID
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
    if images.shape[0] < 2:
        raise ValueError("The input array should have at least two elements.")

    # Separate the array into two parts
    # First part takes the first and last element
    source = np.array([images[0], images[-1]])
    # Second part takes the rest of the elements
    interpolation = images[1:-1]
    return source, interpolation


def compute_fidelity(images, device="cuda"):
    '''
    Compute the Fidelity of the input images.
    '''
    source_images, interpolated_images = seperate_source_and_interpolated_images(images)
    source_features = get_inception_features(source_images, device)
    interpolated_features = get_inception_features(interpolated_images, device)
    fid_score = calculate_fid(source_features, interpolated_features)
    return fid_score

