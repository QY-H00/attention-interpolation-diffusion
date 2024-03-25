import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lpips import LPIPS
from PIL import Image
from torchvision.transforms import Normalize


def show_images_horizontally(
    list_of_files: np.array, output_file: Optional[str] = None, interact: bool = False
) -> None:
    """
    Visualize the list of images horizontally and save the figure as PNG.

    Args:
        list_of_files: The list of images as numpy array with shape (N, H, W, C).
        output_file: The output file path to save the figure as PNG.
        interact: Whether to show the figure interactively in Jupyter Notebook or not in Python.
    """
    number_of_files = len(list_of_files)

    heights = [a[0].shape[0] for a in list_of_files]
    widths = [a.shape[1] for a in list_of_files[0]]

    fig_width = 8.0  # inches
    fig_height = fig_width * sum(heights) / sum(widths)

    # Create a figure with subplots
    _, axs = plt.subplots(
        1, number_of_files, figsize=(fig_width * number_of_files, fig_height)
    )
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
    """
    Save the image as JPG.

    Args:
        image: The input image as numpy array with shape (H, W, C).
        file_name: The file name to save the image.
    """
    image = Image.fromarray(image)
    image.save(file_name)


def load_and_process_images(load_dir: str) -> np.array:
    """
    Load and process the images into numpy array from the directory.

    Args:
        load_dir: The directory to load the images.

    Returns:
        images: The images as numpy array with shape (N, H, W, C).
    """
    images = []
    print(load_dir)
    filenames = sorted(
        os.listdir(load_dir), key=lambda x: int(x.split(".")[0])
    )  # Ensure the files are sorted numerically
    for filename in filenames:
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(load_dir, filename))
            img_array = (
                np.asarray(img) / 255.0
            )  # Convert to numpy array and scale pixel values to [0, 1]
            images.append(img_array)
    return images


def compute_lpips(images: np.array, lpips_model: LPIPS) -> np.array:
    """
    Compute the LPIPS of the input images.

    Args:
        images: The input images as numpy array with shape (N, H, W, C).
        lpips_model: The LPIPS model used to compute perceptual distances.

    Returns:
        distances: The LPIPS of the input images.
    """
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
    """
    Compute the Gini index of the input distances.

    Args:
        distances: The input distances as numpy array.

    Returns:
        gini: The Gini index of the input distances.
    """
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


def compute_smoothness_and_consistency(images: np.array, lpips_model: LPIPS) -> tuple:
    """
    Compute the smoothness and efficiency of the input images.

    Args:
        images: The input images as numpy array with shape (N, H, W, C).
        lpips_model: The LPIPS model used to compute perceptual distances.

    Returns:
        smoothness: One minus gini index of LPIPS of consecutive images.
        consistency: The mean LPIPS of consecutive images.
        max_inception_distance: The maximum LPIPS of consecutive images.
    """
    distances = compute_lpips(images, lpips_model)
    smoothness = 1 - compute_gini(distances)
    consistency = np.mean(distances)
    max_inception_distance = np.max(distances)
    return smoothness, consistency, max_inception_distance


def separate_source_and_interpolated_images(images: np.array) -> tuple:
    """
    Separate the input images into source and interpolated images.
    The input source is the start and end of the images, while the interpolated images are the rest.

    Args:
        images: The input images as numpy array with shape (N, H, W, C).

    Returns:
        source: The source images as numpy array with shape (2, H, W, C).
        interpolation: The interpolated images as numpy array with shape (N-2, H, W, C).
    """
    # Check if the array has at least two elements
    if len(images) < 2:
        raise ValueError("The input array should have at least two elements.")

    # Separate the array into two parts
    # First part takes the first and last element
    source = np.array([images[0], images[-1]])
    # Second part takes the rest of the elements
    interpolation = images[1:-1]
    return source, interpolation
