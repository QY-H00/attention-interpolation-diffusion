import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from scipy.stats import wasserstein_distance
import torch
from torchvision.models import inception_v3
from torchvision.transforms import Normalize
from scipy.linalg import sqrtm


def showImagesHorizontally(list_of_files, output_file):
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


def load_and_process_images(directory):
    images = []
    filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))  # Ensure the files are sorted numerically
    for filename in filenames:
        if filename.endswith('.jpeg'):
            img = Image.open(os.path.join(directory, filename))
            img_array = np.asarray(img) / 255.0  # Convert to numpy array and scale pixel values to [0, 1]
            images.append(img_array)
    return images


def compute_wasserstein_distances(images):
    distances = []
    for i in range(len(images) - 1):
        img1 = images[i].ravel()  # Flatten the image array
        img2 = images[i + 1].ravel()  # Flatten the next image array
        distance = wasserstein_distance(img1, img2)
        distances.append(distance)
    return distances


def compute_smoothness_and_efficiency(images):
    wasserstein_distances = compute_wasserstein_distances(images)
    smoothness = np.mean(wasserstein_distances)
    efficiency = np.var(wasserstein_distances)
    return smoothness, efficiency


def calculate_fid(act1, act2):
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
    # Load pretrained Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    return inception_model


def get_inception_features(images, inception_model, device="cuda"):
    # Convert numpy images to PyTorch tensors and normalize
    images = torch.tensor(images).to(device).float()
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
    # Resize images to fit Inception v3 size requirements
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # Use Inception model to extract features
    with torch.no_grad():
        features = inception_model(images).detach().cpu().numpy()
    return features


def seperate_source_and_interpolated_images(images):
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
    # Compute FID score
    source_images, interpolated_images = seperate_source_and_interpolated_images(images)
    source_features = get_inception_features(source_images, device)
    interpolated_features = get_inception_features(interpolated_images, device)
    fid_score = calculate_fid(source_features, interpolated_features)
    return fid_score


