import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import shutil
import torch

from torchvision.transforms import transforms


################################# Path & Directory #################################
def convert_to_grayscale(source_folder, target_folder):
    """
    Converts all images in the source folder to grayscale and saves them to the target folder.
    
    Parameters:
    source_folder (str): Path to the folder containing original images.
    target_folder (str): Path to the folder where grayscale images will be saved.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for img_name in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_name)
        if is_image_file(img_path):  # Assuming is_image_file checks for image extensions
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(target_folder, img_name), gray_img)


def copy_large_images(input_folder, output_folder, min_size=256):
    """
    Copies all images with dimensions greater than 256x256 pixels from the input folder to the output folder.
    
    Parameters:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save the large images.
    min_size (int): Minimum size (in pixels) for both width and height. Default is 256.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (basic check by extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)
            
            # Get the image dimensions
            height, width = image.shape[:2]
            
            # Check if both dimensions are greater than the specified minimum size
            if height >= min_size and width >= min_size:
                output_path = os.path.join(output_folder, filename)
                shutil.copy(input_path, output_path)
                print(f"Copied: {output_path}")


def pad_image_to_256(image):
    """
    Pads the image to ensure both dimensions are at least 256 pixels.
    Padding is applied symmetrically, and the padded area is filled with black (0).
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The padded image with dimensions of at least 256x256 pixels.
    """
    original_height, original_width = image.shape[:2]
    
    pad_height = max(0, 256 - original_height)
    pad_width = max(0, 256 - original_width)
    
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image


def process_images_in_folder(input_folder, output_folder):
    """
    Pads all images in the input folder to at least 256x256 pixels and saves them to the output folder.
    
    Parameters:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save padded images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        # Full path to the input file
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (basic check by extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)
            
            # Pad the image if necessary
            padded_image = pad_image_to_256(image)
            
            # Full path to the output file
            output_path = os.path.join(output_folder, filename)
            
            # Save the padded image
            cv2.imwrite(output_path, padded_image)
            print(f"Processed and saved: {output_path}")


def calculate_mean_std(directory):
    """
    Calculates the mean and standard deviation of pixel values in grayscale images from a directory.
    
    Parameters:
    directory (str): Path to the directory containing images.
    
    Returns:
    tuple: A tuple containing the mean and standard deviation of pixel values.
    """
    pixel_sum = 0
    pixel_sum_squared = 0
    num_pixels = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if is_image_file(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            pixels = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            pixel_sum += np.sum(pixels)
            pixel_sum_squared += np.sum(pixels ** 2)
            num_pixels += pixels.size

    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_sum_squared / num_pixels) - (mean ** 2))
    return mean, std


def is_image_file(filename):
    """
    Checks if a file is an image based on its extension.
    
    Parameters:
    filename (str): Name of the file to check.
    
    Returns:
    bool: True if the file is an image, False otherwise.
    """
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    """
    Creates a list of image file paths from a directory and its subdirectories.
    
    Parameters:
    dir (str): Path to the directory containing images.
    
    Returns:
    list: A list of file paths to images.
    """
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
    """
    Creates a new experiment directory with an incremental number.
    
    Parameters:
    main_dir (str): Path to the main directory where experiment folders will be created.
    
    Returns:
    dict: A dictionary containing the new directory path and its incremental number.
    """
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Transforms #################################
def get_transforms(args):
    """
    Returns a list of torchvision transforms based on the provided arguments.
    
    Parameters:
    args (Namespace): Argument namespace containing crop, patch_size, mean, and std.
    
    Returns:
    list: A list of torchvision transforms.
    """
    transform_list = [transforms.ToTensor()]
    if args.crop:
        transform_list.append(transforms.CenterCrop(size=args.patch_size))
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


def crop(img, patch_size):
    """
    Crops the center of an image to the specified patch size.
    
    Parameters:
    img (numpy.ndarray): The input image.
    patch_size (int): The size of the patch to crop.
    
    Returns:
    numpy.ndarray: The cropped image patch.
    """
    if img.ndim == 2:
        h, w = img.shape
        return img[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    elif img.ndim == 3:
        c, h, w = img.shape
        return img[:, h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2]
    else:
        raise NotImplementedError('Wrong image dim')


################################# ETC #################################
def denorm(tensor, mean=0.5, std=0.5, max_pixel=1.):
    """
    Denormalizes a tensor by multiplying by standard deviation and adding the mean.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    mean (float): The mean value used for normalization.
    std (float): The standard deviation used for normalization.
    max_pixel (float): The maximum pixel value (default is 1).
    
    Returns:
    torch.Tensor: The denormalized tensor.
    """
    return std*max_pixel*tensor + mean*max_pixel


def tensor_to_numpy(x):
    """
    Converts a PyTorch tensor to a NumPy array.
    
    Parameters:
    x (torch.Tensor): The input tensor.
    
    Returns:
    numpy.ndarray: The converted NumPy array.
    """
    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        return x.transpose((0, 2, 3, 1))
    elif x.ndim == 3:
        return x.transpose((1, 2, 0))
    elif x.ndim == 2:
        return x
    else:
        raise


def plot_tensors(tensor_list, title_list):
    """
    Plots a list of tensors as images with corresponding titles.
    
    Parameters:
    tensor_list (list): List of PyTorch tensors to plot.
    title_list (list): List of titles for each tensor.
    """
    numpy_list = [tensor_to_numpy(t) for t in tensor_list]
    fig = plt.figure(figsize=(4*len(numpy_list), 8))
    rows, cols = 1, len(numpy_list)
    for i in range(len(numpy_list)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(numpy_list[i], cmap='gray')
        plt.title(title_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


class LambdaLR:
    """
    Defines a learning rate schedule with linear decay starting at a specified epoch.
    
    Parameters:
    n_epochs (int): Total number of epochs.
    offset (int): Number of epochs to offset the start of the decay.
    decay_start_epoch (int): Epoch at which to start the decay.
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Calculates the learning rate multiplier based on the current epoch.
        
        Parameters:
        epoch (int): The current epoch.
        
        Returns:
        float: The learning rate multiplier.
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


if __name__ == "__main__":
    input_folder  = '/Users/samynehmad/studies/final_project/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data/all_datasets/Set30'
    output_folder = '/Users/samynehmad/studies/final_project/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data/all_datasets/Set30'
    # copy_large_images(input_folder, output_folder)

    # process_images_in_folder(input_folder, output_folder)

    convert_to_grayscale(input_folder, output_folder)
