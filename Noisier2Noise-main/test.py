import argparse
import random
import time
from glob import glob
import os
import csv

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.DnCNN import DnCNN
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test Nr2N public')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=10, type=int)

# Model parameters
parser.add_argument('--n_epochs', default=180, type=int)

# Test parameters
parser.add_argument('--noise', default='gauss_25', type=str)  # 'gauss_intensity', 'poisson_intensity'
parser.add_argument('--dataset', default='Set12', type=str)  # BSD100, Kodak, Set12
parser.add_argument('--aver_num', default=30, type=int)
parser.add_argument('--alpha', default=1.0, type=float)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4097)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2719)  # ImageNet Gray: 0.2927

opt = parser.parse_args()


class inner_stats:
    def __init__(self, noise, noiser, prediction, op_mean, op_median):
        self.noise = noise

class img_stats:
    def __init__(self, psnr, ssim):
        self.psnr = psnr
        self.ssim = ssim


def generate(args):
    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = DnCNN().to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    img_dir = os.path.join('../all_datasets/', args.dataset)
    save_dir = os.path.join('./results/', args.dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_paths = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

    noise_type = args.noise.split('_')[0]
    noise_intensity = float(args.noise.split('_')[1]) / 255.

    transform = transforms.Compose(get_transforms(args))

    # Initialize CSV for all images
    all_psnr_csv_path = os.path.join(save_dir, 'all_images_PSNR.csv')
    all_ssim_csv_path = os.path.join(save_dir, 'all_images_SSIM.csv')

    all_psnr_data = []
    all_ssim_data = []

    for index, clean255 in enumerate(imgs):
        if args.crop:
            clean255 = crop(clean255, patch_size=args.patch_size)

        clean_numpy = clean255 / 255.
        noisy_numpy = create_noisy_image(clean_numpy, noise_type, noise_intensity)

        img_psnr_data = []
        img_ssim_data = []

        # Prepare CSV paths for the current image
        img_psnr_csv_path = os.path.join(save_dir, 'image_{}_PSNR.csv'.format(index + 1))
        img_ssim_csv_path = os.path.join(save_dir, 'image_{}_SSIM.csv'.format(index + 1))

        for i in range(args.aver_num):
            noisier_numpy = add_noise(noisy_numpy, clean_numpy, noise_type, noise_intensity, args.alpha)
            noisier_tensor = transform(noisier_numpy).unsqueeze(0).type(torch.FloatTensor).to(device)

            prediction = ((1 + args.alpha ** 2) * model(noisier_tensor) - noisier_tensor) / (args.alpha ** 2)
            overlap_mean = torch.mean(prediction, dim=0)
            overlap_median, _ = torch.median(prediction, dim=0)

            if args.normalize:
                prediction, overlap_mean, overlap_median = denorm(prediction, args.mean, args.std), denorm(overlap_mean, args.mean, args.std), denorm(overlap_median, args.mean, args.std)

            prediction_numpy, overlap_mean_numpy, overlap_median_numpy = tensor_to_numpy(prediction).squeeze(), tensor_to_numpy(overlap_mean).squeeze(), tensor_to_numpy(overlap_median).squeeze()

            n_psnr = psnr(clean_numpy, noisy_numpy, data_range=1)
            p_psnr = psnr(clean_numpy, prediction_numpy, data_range=1)
            op_mean_psnr = psnr(clean_numpy, overlap_mean_numpy, data_range=1)
            op_median_psnr = psnr(clean_numpy, overlap_median_numpy, data_range=1)

            n_ssim = ssim(clean_numpy, noisy_numpy, data_range=1)
            p_ssim = ssim(clean_numpy, prediction_numpy, data_range=1)
            op_mean_ssim = ssim(clean_numpy, overlap_mean_numpy, data_range=1)
            op_median_ssim = ssim(clean_numpy, overlap_median_numpy, data_range=1)

            img_psnr_data.append([i, n_psnr, p_psnr, op_mean_psnr, op_median_psnr])
            img_ssim_data.append([i, n_ssim, p_ssim, op_mean_ssim, op_median_ssim])

        # Write the PSNR and SSIM data for the current image to its CSV files
        write_csv(img_psnr_csv_path, ['i', 'noisy_psnr', 'prediction_psnr', 'overlap_psnr_mean', 'overlap_psnr_median'], img_psnr_data)
        write_csv(img_ssim_csv_path, ['i', 'noisy_ssim', 'prediction_ssim', 'overlap_ssim_mean', 'overlap_ssim_median'], img_ssim_data)

        # Aggregate data for all images
        if index == 0:
            all_psnr_data = np.array(img_psnr_data)
            all_ssim_data = np.array(img_ssim_data)
        else:
            all_psnr_data[:, 1:] += np.array(img_psnr_data)[:, 1:]
            all_ssim_data[:, 1:] += np.array(img_ssim_data)[:, 1:]

    # Calculate the mean across all images
    all_psnr_data[:, 1:] /= len(imgs)
    all_ssim_data[:, 1:] /= len(imgs)

    # Write the aggregated PSNR and SSIM data to the all_images CSV files
    write_csv(all_psnr_csv_path, ['i', 'all_img_mean_noisy_psnr', 'all_img_mean_prediction_psnr', 'all_img_mean_overlap_psnr_mean', 'all_img_mean_overlap_psnr_median'], all_psnr_data)
    write_csv(all_ssim_csv_path, ['i', 'all_img_mean_noisy_ssim', 'all_img_mean_prediction_ssim', 'all_img_mean_overlap_ssim_mean', 'all_img_mean_overlap_ssim_median'], all_ssim_data)


def create_noisy_image(clean_numpy, noise_type, noise_intensity):
    if noise_type == 'gauss':
        return clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
    elif noise_type == 'poisson':
        return np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
    else:
        raise NotImplementedError('wrong type of noise')


def add_noise(noisy_numpy, clean_numpy, noise_type, noise_intensity, alpha):
    if noise_type == 'gauss':
        return noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * alpha
    elif noise_type == 'poisson':
        return noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
    else:
        raise NotImplementedError('wrong type of noise')


def write_csv(file_path, header, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == "__main__":
    generate(opt)
