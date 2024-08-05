import os
import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.
    
    Parameters:
    image (PIL.Image): Input image.
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.
    
    Returns:
    PIL.Image: Image with added Gaussian noise.
    """
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape)
    noisy_image = np_image + noise

    # Clip the values to be in the valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_image)

def process_images(input_folder, output_folder, mean=0, std=25):
    """
    Processes all images in the input folder by adding Gaussian noise and saves them to the output folder.
    
    Parameters:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save noisy images.
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert('RGB')

            noisy_image = add_gaussian_noise(image, mean, std)
            output_path = os.path.join(output_folder, filename)
            noisy_image.save(output_path)

            print(f"Processed {filename}")


def convert_jpg_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(input_folder, filename))
            png_filename = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(output_folder, png_filename))
            print(f"Converted {filename} to {png_filename}")



def convert():
    input_folder = 'all_datasets/noise_pics/new_pictures/data/natural_images/person'
    output_folder = 'all_datasets/noise_pics/new_pictures/data/natural_images/person'

    convert_jpg_to_png(input_folder, output_folder)


def noise():
    # Define input and output folders
    input_folder = 'all_datasets/noise_pics/new_pictures/data/natural_images/person'
    output_folder = 'all_datasets/noiser_pics/new_pictures/data/natural_images/person'

    # Process the images with Gaussian noise
    process_images(input_folder, output_folder, mean=0, std=25)

convert()
