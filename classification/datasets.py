import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageEnhance, ImageFilter
def apply_enhancements(image, gaussian_params, brightness_factor, contrast_factor, blur_radius, speckle_params):
    """
    Apply specified enhancements to an image.
    """
    mean, std = gaussian_params
    speckle_amount, = speckle_params

    np_image = np.asarray(image).astype(np.float32)  # Using float32 for operations

    # Generate Gaussian noise
    gaussian = np.random.normal(mean, std,np_image.shape)  # mean=0, std=5

    # Add Gaussian noise to the image
    noisy_image = np_image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are within pixel range
    noisy_image = noisy_image.astype(np.uint8)  # Convert back to uint8

    # Convert back to PIL Image
    image = Image.fromarray(noisy_image)


    # Change brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Change contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Apply Gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    # Add speckle noise
    np_image = np.array(image)
    noise = np.random.randn(*np_image.shape) * speckle_amount
    noisy_image = np_image + np_image * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    image = Image.fromarray(noisy_image)

    return image

def process_image(image):
    """
    Process a list of images with consistent random enhancements.
    """
    # Generate consistent enhancement parameters
    p=[0,5,1.0,1.5,3,0.08]
    scaled_p = [1.5* x for x in p]
    gaussian_params = (scaled_p[0], scaled_p[1])  # mean and std for Gaussian noise
    brightness_factor = scaled_p[2]
    contrast_factor = scaled_p[3]
    blur_radius = scaled_p[4]
    speckle_params = (scaled_p[5],)  # amount for speckle noise
    # gaussian_params = (0, 10)  # mean and std for Gaussian noise
    # brightness_factor = 2.0
    # contrast_factor = 3
    # blur_radius = 6
    # speckle_params = (0.08,)  # amount

    # Apply enhancements to each image
    return apply_enhancements(image, gaussian_params, brightness_factor, contrast_factor, blur_radius, speckle_params)
class ImageDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.transform = transforms
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')


        img = Image.open(img_path[0])
        img = process_image(img)

        label = int(img_path[1])

        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img = Image.fromarray(np.fliplr(np.array(img)),'L')
            else:
                img = Image.fromarray(np.flipud(np.array(img)),'L')

        img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.files)
