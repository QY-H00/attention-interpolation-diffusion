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
    smoothness = compute_gini(wasserstein_distances)
    efficiency = np.mean(wasserstein_distances)
    return smoothness, efficiency


def sort_source_and_interpolated():
    root_dir = "results"
    source_dir = os.path.join(root_dir, "source")
    interpolated_dir = os.path.join(root_dir, "interpolated")
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(interpolated_dir):
        os.makedirs(interpolated_dir)
    direct_eval_dir = os.path.join(root_dir, "direct_eval")
    for pair_dir in os.listdir(direct_eval_dir):
        print(pair_dir)
        for trial_dir in os.listdir(os.path.join(direct_eval_dir, pair_dir)):
            size = len(os.listdir(os.path.join(direct_eval_dir, pair_dir, trial_dir)))
            source_1_path = os.path.join(direct_eval_dir, pair_dir, trial_dir, "1.jpg")
            source_1 = Image.open(source_1_path)
            source_1.save(os.path.join(source_dir, f"{pair_dir}_{trial_dir}_1.jpg"))
            source_2_path = os.path.join(direct_eval_dir, pair_dir, trial_dir, f"{size}.jpg")
            source_2 = Image.open(source_2_path)
            source_2.save(os.path.join(source_dir, f"{pair_dir}_{trial_dir}_{size}.jpg"))
            for i in range(2, size):
                img_path = os.path.join(direct_eval_dir, pair_dir, trial_dir, f"{i}.jpg")
                img = Image.open(img_path)
                img.save(os.path.join(interpolated_dir, f"{pair_dir}_{trial_dir}_{i}.jpg"))


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

