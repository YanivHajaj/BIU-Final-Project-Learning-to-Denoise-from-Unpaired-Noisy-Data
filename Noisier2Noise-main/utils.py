import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
from torchvision.transforms import transforms


################################# Path & Directory #################################
def convert_to_grayscale(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for img_name in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_name)
        if is_image_file(img_path):  # Assuming is_image_file checks for image extensions
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(target_folder, img_name), gray_img)


def calculate_mean_std(directory):
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
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
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
    transform_list = [transforms.ToTensor()]
    if args.crop:
        transform_list.append(transforms.CenterCrop(size=args.patch_size))
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


def crop(img, patch_size):
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
    return std*max_pixel*tensor + mean*max_pixel


def tensor_to_numpy(x):
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
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


if __name__ == "__main__":
    convert_to_grayscale('/Users/samynehmad/studies/final_project/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data/all_datasets/Set20', '/Users/samynehmad/studies/final_project/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data/all_datasets/Set20')