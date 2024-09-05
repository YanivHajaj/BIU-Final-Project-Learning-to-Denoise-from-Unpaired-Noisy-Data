# usage: test.py [-h] [--test_info TEST_INFO] [--gpu_num GPU_NUM] [--seed SEED] [--exp_num EXP_NUM] [--n_epochs N_EPOCHS] [--noise NOISE]
#                [--dataset DATASET] [--exp_rep EXP_REP] [--aver_num AVER_NUM] [--alpha ALPHA] [--trim_op TRIM_OP] [--noisy_input NOISY_INPUT]
#                [--crop CROP] [--patch_size PATCH_SIZE] [--normalize NORMALIZE] [--mean MEAN] [--std STD]

import argparse
import random
import math
from glob import glob
import csv
import os
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.DnCNN import DnCNN
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test Nr2N public')

parser.add_argument('--test_info', default='None info given', type=str)

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=90, type=int)
parser.add_argument('--exp_num', default=10, type=int)

# Model parameters
parser.add_argument('--n_epochs', default=180, type=int)

# Test parameters
parser.add_argument('--noise', default='gauss_25', type=str)  # 'gauss_intensity', 'poisson_intensity'
parser.add_argument('--dataset', default='Set12', type=str)  # BSD100, Kodak, Set12
parser.add_argument('--exp_rep', default=None, type=str)
parser.add_argument('--aver_num', default=10, type=int)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--trim_op', default=0.05, type=float)
parser.add_argument('--noisy_input', type=bool, default=False)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4097)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2719)  # ImageNet Gray: 0.2927

opt = parser.parse_args()


def generate(args):
    #device = torch.device('cuda:{}'.format(args.gpu_num))
    device = torch.device('cpu')

    # Random Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    model = DnCNN().to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Directory
    img_dir  = os.path.join('../all_datasets/', args.dataset)
    save_dir = create_next_experiment_folder(os.path.join('./results/', args.dataset, 'imgs'), opt.exp_rep)

    # Images
    # Load all PNG and JPG images from the dataset directory - transform to grayscale
    img_paths = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    imgs      = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

    # Noise
    noise_type = args.noise.split('_')[0]
    noise_intensity = float(args.noise.split('_')[1]) / 255.0

    # print(f'************ noise intensity from {noise_type} is: **************')
    # print(noise_intensity)

    # Transform
    transform = transforms.Compose(get_transforms(args))

    # Denoising params
    psnr_averages, ssim_averages = {}, {}

    # CSV
    csv_header = ['k', 'noisy', 'prediction', 'overlap_mean', 'overlap_median', 'overlap_trimmed_mean']
    csv_folder = create_next_experiment_folder(os.path.join('./results/', args.dataset, 'csvs'), opt.exp_rep)


    for index, clean255 in enumerate(imgs):
        if args.crop:
            #clean255 = crop(clean255, patch_size=args.patch_size)
            clean255 = crop_and_resize(clean255, output_size=args.patch_size)

        clean_numpy = clean255/255.
        if noise_type == 'gauss':
            if not opt.noisy_input:
                noisy_numpy      = clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
            else:
                noisy_numpy      = clean_numpy

            noisier_numpy_single = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * args.alpha
        elif noise_type == 'poisson':
            if not opt.noisy_input:
                noisy_numpy          = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
            else:
                noisy_numpy      = clean_numpy
            noisier_numpy_single = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
        else:
            raise NotImplementedError('wrong type of noise')
        
        noisy, noisier = transform(noisy_numpy), transform(noisier_numpy_single)
        noisy, noisier = torch.unsqueeze(noisy, dim=0), torch.unsqueeze(noisier, dim=0)
        noisy, noisier = noisy.type(torch.FloatTensor).to(device), noisier.type(torch.FloatTensor).to(device)

        # Noisier Prediction
        prediction = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)

        # Overlap Prediction
        noisier = torch.zeros(size=(args.aver_num, 1, *clean_numpy.shape))

        for i in range(args.aver_num):
            if noise_type == 'gauss':
                noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * args.alpha
            elif noise_type == 'poisson':
                noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
            else:
                raise NotImplementedError('wrong type of noise')
            
            noisier_tensor      = transform(noisier_numpy)
            noisier_tensor      = torch.unsqueeze(noisier_tensor, dim=0)
            noisier[i, :, :, :] = noisier_tensor

        noisier = noisier.type(torch.FloatTensor).to(device)

        # Calculate overlap using mean and median
        overlap_prediction = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)

        overlap_mean      = torch.mean(overlap_prediction, dim=0)
        overlap_median    = torch.quantile(overlap_prediction, q=0.5, dim=0)

        # Trimmed mean per pixel calculation 
        sorted_overlap, _ = torch.sort(overlap_prediction, dim=0)

        num_to_trim = 0

        if i > 15:
            num_to_trim = 2
        elif i > 6:
            num_to_trim = 1


        trim_percent = args.trim_op  # 10% trimming by default
        # num_to_trim = math.floor(trim_percent * sorted_overlap.size(0))
        
        # Ensure that trimming does not empty the tensor
        # if num_to_trim == 0 or 2 * num_to_trim >= sorted_overlap.size(0):
        #     print(f"Trimming would result in an empty tensor or no trimming possible. Skipping trimming for this image.")
        #     overlap_trimmed_mean = torch.mean(sorted_overlap, dim=0)  # No trimming applied
        # else:
        if num_to_trim:
            trimmed_overlap = sorted_overlap[num_to_trim:-num_to_trim, :, :, :]
            overlap_trimmed_mean = torch.mean(trimmed_overlap, dim=0)  # Mean across the batch dimension after trimming
            print(f"Trimmed with num_to_trim : {num_to_trim}")
        else:
            overlap_trimmed_mean = overlap_mean
        # print(f"Calculated overlap_trimmed_mean: {overlap_trimmed_mean}")


        # Change to Numpy
        if args.normalize:
            prediction      = denorm(prediction, mean=args.mean, std=args.std)
            overlap_mean    = denorm(overlap_mean, mean=args.mean, std=args.std)
            overlap_median  = denorm(overlap_median, mean=args.mean, std=args.std)
            overlap_trimmed = denorm(overlap_trimmed_mean, mean=args.mean, std=args.std)
            noisier         = denorm(noisier_numpy_single, mean=args.mean, std=args.std)  # Add denormalization for noisier (the single noisier of the predict, not overlap)

        prediction, overlap_mean, overlap_median, overlap_trimmed                          = tensor_to_numpy(prediction), tensor_to_numpy(overlap_mean), tensor_to_numpy(overlap_median), tensor_to_numpy(overlap_trimmed)
        prediction_numpy, overlap_mean_numpy, overlap_median_numpy, overlap_trimmed_numpy  = np.squeeze(prediction), np.squeeze(overlap_mean), np.squeeze(overlap_median), np.squeeze(overlap_trimmed)
        noisier_numpy                                                                      = np.squeeze(noisier)  # Convert noisier tensor to numpy and squeeze

        image_metrics = calculate_metrics(clean_numpy, noisy_numpy, prediction_numpy, overlap_mean_numpy, overlap_median_numpy, noisier_numpy, overlap_trimmed_numpy)

        for metric_type in ['psnr', 'ssim']:
            averages = psnr_averages if metric_type == 'psnr' else ssim_averages
            for key, value in image_metrics[metric_type].items():
                if key in averages:
                    averages[key] += value / len(imgs)
                else:
                    averages[key] = value / len(imgs)

        # Log PSNR and SSIM values
        print('{}th image | PSNR: noisy:{:.3f}, prediction:{:.3f}, overlap_mean:{:.3f}, overlap_median:{:.3f}, overlap_trimmed_mean:{:.3f}, noisier:{:.3f} | SSIM: noisy:{:.3f}, prediction:{:.3f}, overlap_mean:{:.3f}, overlap_median:{:.3f}, overlap_trimmed_mean:{:.3f}, noisier:{:.3f}'.format(
                index + 1, image_metrics['psnr']['noisy'], image_metrics['psnr']['prediction'], image_metrics['psnr']['overlap_mean'], 
                image_metrics['psnr']['overlap_median'], image_metrics['psnr']['overlap_trim'], image_metrics['psnr']['noisier'], image_metrics['ssim']['noisy'],
                image_metrics['ssim']['prediction'], image_metrics['ssim']['overlap_mean'],
                image_metrics['ssim']['overlap_median'], image_metrics['ssim']['overlap_trim'], image_metrics['ssim']['noisier']))
             
        # write on SCV per image SSIM and PSNR
        # Assuming img_paths[index] is the image file path with an extension
        file_name = os.path.splitext(os.path.basename(img_paths[index]))[0]  # Extracts the base file name without extension

        file_path = f'{csv_folder}PSNR_{index}_{file_name}.csv'  
        csv_data  = [args.aver_num ,image_metrics['psnr']['noisy'], image_metrics['psnr']['prediction'], 
                    image_metrics['psnr']['overlap_mean'], image_metrics['psnr']['overlap_median'], image_metrics['psnr']['overlap_trim']]
        
        write_csv(file_path, csv_data, csv_header)

        file_path = f'{csv_folder}SSIM_{index}_{file_name}.csv'  
        csv_data  = [args.aver_num, image_metrics['ssim']['noisy'], image_metrics['ssim']['prediction'], 
                    image_metrics['ssim']['overlap_mean'], image_metrics['ssim']['overlap_median'], image_metrics['ssim']['overlap_trim']]
        
        write_csv(file_path, csv_data, csv_header)


        # Save sample images (up to 10 images)
        if index <= 10:
            sample_clean, sample_noisy  = 255. * np.clip(clean_numpy, 0., 1.), 255. * np.clip(noisy_numpy, 0., 1.)
            sample_prediction           = 255. * np.clip(prediction_numpy, 0., 1.)
            sample_overlap_mean         = 255. * np.clip(overlap_mean_numpy, 0., 1.)
            sample_overlap_median       = 255. * np.clip(overlap_median_numpy, 0., 1.)
            sample_overlap_trimmed      = 255. * np.clip(overlap_trimmed_numpy, 0., 1.)
            sample_noisier              = 255. * np.clip(noisier_numpy, 0., 1.)  # Prepare noisier image for saving

            cv2.imwrite(os.path.join(save_dir, '{}th_clean.png'.format(index+1)), sample_clean)
            cv2.imwrite(os.path.join(save_dir, '{}th_noisy.png'.format(index+1)), sample_noisy)
            cv2.imwrite(os.path.join(save_dir, '{}th_prediction.png'.format(index+1)), sample_prediction)
            cv2.imwrite(os.path.join(save_dir, '{}th_overlap_mean.png'.format(index+1)), sample_overlap_mean)
            cv2.imwrite(os.path.join(save_dir, '{}th_overlap_median.png'.format(index+1)), sample_overlap_median)
            cv2.imwrite(os.path.join(save_dir, '{}th_overlap_trimmed.png'.format(index+1)), sample_overlap_trimmed)
            cv2.imwrite(os.path.join(save_dir, '{}th_noisier.png'.format(index + 1)),sample_noisier)  # Save noisier image

    # Total PSNR, SSIM
    print('{} Average PSNR | noisy:{:.3f}, prediction:{:.3f}, overlap_mean:{:.3f}, overlap_median:{:.3f}, overlap_trimmed_mean:{:.3f}'.format(
        args.dataset, psnr_averages['noisy'], psnr_averages['prediction'], psnr_averages['overlap_mean'], psnr_averages['overlap_median'], psnr_averages['noisier'], psnr_averages['overlap_trim']))
    print('{} Average SSIM | noisy:{:.3f},  prediction:{:.3f}, overlap_mean:{:.3f}, overlap_median:{:.3f}, overlap_trimmed_mean:{:.3f}'.format(
        args.dataset, ssim_averages['noisy'], ssim_averages['prediction'], ssim_averages['overlap_mean'], ssim_averages['overlap_median'], ssim_averages['noisier'], ssim_averages['overlap_trim']))
  
    # write average PSNR per k
    file_path = f'{csv_folder}PSNR_all_images_average.csv'  
    csv_data  = [args.aver_num, psnr_averages['noisy'], psnr_averages['prediction'], psnr_averages['overlap_mean'], psnr_averages['overlap_median'], psnr_averages['overlap_trim']]
    write_csv(file_path, csv_data, csv_header)

    # write average SSIM per k 
    file_path = f'{csv_folder}SSIM_all_images_average.csv'  
    csv_data  = [args.aver_num, ssim_averages['noisy'], ssim_averages['prediction'], ssim_averages['overlap_mean'], ssim_averages['overlap_median'], ssim_averages['overlap_trim']]
    write_csv(file_path, csv_data, csv_header)



    ###### UTILS FUNCTIONS #######
def create_next_experiment_folder(base_folder, exp_repeated = None):
    """
    Creates the next available experiment folder in a base folder.
    The folder is named with an incremental number (e.g., exp1, exp2, etc.).

    Parameters:
    base_folder (str): The base directory where experiment folders are created.

    Returns:
    str: The path to the newly created experiment folder.
    """
    if exp_repeated:
        return os.path.join(base_folder, f'{exp_repeated}/')
    
    # Ensure the base folder exists
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Find the next available folder number
    exp_num = 1
    while os.path.exists(f'{base_folder}/exp{exp_num}'):
        exp_num += 1
    
    # Create the new folder
    new_folder = f'{base_folder}/exp{exp_num}/'
    os.makedirs(new_folder)

    if opt.test_info:
        with open(new_folder + 'info.txt', 'w') as file:
            # Write test_info to the file
            file.write(opt.test_info + "\n")
            file.write(str(vars(opt)))
    

    return new_folder


def write_csv(file_path, data, header):
    """
    Appends data to a CSV file, creating the file with a header if it doesn't exist.

    Parameters:
    file_path (str): Path to the CSV file.
    data (list): A list of data to write as a new row.
    header (list): A list representing the header of the CSV file.
    """
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)


def calculate_metrics(clean_numpy, noisy_numpy, prediction_numpy, overlap_mean_numpy, overlap_median_numpy, noisier_numpy, overlap_trimmed_numpy):
    """
    Calculates PSNR and SSIM metrics for various stages of image processing.

    Parameters:
    clean_numpy (numpy.ndarray): The clean image array.
    noisy_numpy (numpy.ndarray): The noisy image array.
    prediction_numpy (numpy.ndarray): The predicted image array.
    overlap_mean_numpy (numpy.ndarray): The image after mean overlap.
    overlap_median_numpy (numpy.ndarray): The image after median overlap.
    noisier_numpy (numpy.ndarray): The noisier version of the image array.

    Returns:
    dict: A dictionary containing PSNR and SSIM metrics for each image stage.
    """
    metrics = {
        'psnr': {
            'noisy'         : psnr(clean_numpy, noisy_numpy, data_range=1),
            'prediction'    : psnr(clean_numpy, prediction_numpy, data_range=1),
            'overlap_mean'  : psnr(clean_numpy, overlap_mean_numpy, data_range=1),
            'overlap_median': psnr(clean_numpy, overlap_median_numpy, data_range=1),
            'overlap_trim'  : psnr(clean_numpy, overlap_trimmed_numpy, data_range=1),
            'noisier'       : psnr(clean_numpy, noisier_numpy, data_range=1)
        },
        'ssim': {
            'noisy'         : ssim(clean_numpy, noisy_numpy, data_range=1),
            'prediction'    : ssim(clean_numpy, prediction_numpy, data_range=1),
            'overlap_mean'  : ssim(clean_numpy, overlap_mean_numpy, data_range=1),
            'overlap_median': ssim(clean_numpy, overlap_median_numpy, data_range=1),
            'overlap_trim'  : ssim(clean_numpy, overlap_trimmed_numpy, data_range=1),
            'noisier'       : ssim(clean_numpy, noisier_numpy, data_range=1)
        }
    }
    return metrics


if __name__ == "__main__":
    generate(opt)