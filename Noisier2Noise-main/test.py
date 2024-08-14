import argparse
import random
import time
from glob import glob #comment new

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.DnCNN import DnCNN  # Import the DnCNN model (a neural network for image denoising)
from utils import *  # Import utility functions (like image transformations, etc.)

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Test Nr2N public')

# Define various command-line arguments
parser.add_argument('--gpu_num', default=0, type=int)  # Specifies which GPU to use (default: 0)
parser.add_argument('--seed', default=100, type=int)  # Seed for random number generation (for reproducibility)
parser.add_argument('--exp_num', default=10, type=int)  # Experiment number (used to save/load models)

# Model parameters
parser.add_argument('--n_epochs', default=180, type=int)  # Number of epochs for training the model

# Test parameters (used for evaluating the model)
parser.add_argument('--noise', default='gauss_25', type=str)  # Type and intensity of noise to add to images
parser.add_argument('--dataset', default='Set12', type=str)  # Dataset to use for testing (e.g., Set12, BSD100, etc.)
parser.add_argument('--aver_num', default=20, type=int)  # Number of noisy images to average in overlap prediction
parser.add_argument('--alpha', default=1.0, type=float)  # A parameter controlling the amount of noise added

# Image transformation parameters
parser.add_argument('--crop', type=bool, default=True)  # Whether to crop images
parser.add_argument('--patch_size', type=int, default=256)  # Size of the image patches
parser.add_argument('--normalize', type=bool, default=True)  # Whether to normalize images
parser.add_argument('--mean', type=float, default=0.4050)  # Mean value for normalization (ImageNet Gray: 0.4050)
parser.add_argument('--std', type=float, default=0.2927)  # Standard deviation for normalization (ImageNet Gray: 0.2927)
parser.add_argument('--use_overlap', type=bool, default=True)  # Whether to use the overlap prediction step

# Parse the command-line arguments into a variable called `opt`
opt = parser.parse_args()

# Optional function to perform overlap prediction using Method 1 or Method 2
def perform_overlap_prediction(model, noisier, transform, device, args, clean_numpy, method=2):
    if method == 1:
        # Method 1: Accumulate predictions from multiple noisy versions
        overlap = None

        if args.noise.split('_')[0] == 'gauss':
            noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * (float(args.noise.split('_')[1]) / 255.)
        elif args.noise.split('_')[0] == 'poisson':
            noisy_numpy = np.random.poisson(clean_numpy * 255. * (float(args.noise.split('_')[1]) / 255.)) / (float(args.noise.split('_')[1]) / 255.) / 255.

        for _ in range(args.aver_num):
            if args.noise.split('_')[0] == 'gauss':
                noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * (float(args.noise.split('_')[1]) / 255.) * args.alpha
            elif args.noise.split('_')[0] == 'poisson':
                noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * (float(args.noise.split('_')[1]) / 255.)) / (float(args.noise.split('_')[1]) / 255.) / 255. - clean_numpy)
            else:
                raise NotImplementedError('wrong type of noise')
            
            noisier_tensor = transform(noisier_numpy)
            noisier_tensor = torch.unsqueeze(noisier_tensor, dim=0).type(torch.FloatTensor).to(device)
            single_prediction = ((1 + args.alpha ** 2) * model(noisier_tensor) - noisier_tensor) / (args.alpha ** 2)

            if overlap is None:
                overlap = single_prediction.detach() / args.aver_num
            else:
                overlap += single_prediction.detach() / args.aver_num
        
        return overlap

    elif method == 2:
        # Method 2: Directly average predictions from stored noisy versions
        noisier = torch.zeros(size=(args.aver_num, 1, *clean_numpy.shape))

        if args.noise.split('_')[0] == 'gauss':
            noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * (float(args.noise.split('_')[1]) / 255.)
        elif args.noise.split('_')[0] == 'poisson':
            noisy_numpy = np.random.poisson(clean_numpy * 255. * (float(args.noise.split('_')[1]) / 255.)) / (float(args.noise.split('_')[1]) / 255.) / 255.

        for i in range(args.aver_num):
            if args.noise.split('_')[0] == 'gauss':
                noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * (float(args.noise.split('_')[1]) / 255.) * args.alpha
            elif args.noise.split('_')[0] == 'poisson':
                noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * (float(args.noise.split('_')[1]) / 255.)) / (float(args.noise.split('_')[1]) / 255.) / 255. - clean_numpy)
            else:
                raise NotImplementedError('wrong type of noise')
            
            noisier_tensor = transform(noisier_numpy)
            noisier_tensor = torch.unsqueeze(noisier_tensor, dim=0).type(torch.FloatTensor).to(device)
            noisier[i, :, :, :] = noisier_tensor.view(noisier[i, :, :, :].shape)
        
        noisier = noisier.type(torch.FloatTensor).to(device)
        overlap = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)
        overlap = torch.mean(overlap, dim=0)
        return overlap

    else:
        raise ValueError("Invalid method selected for overlap prediction.")

# Main function that performs image denoising and evaluation
def generate(args):
    #device = torch.device('cuda:{}'.format(args.gpu_num))
    # Set the device (CPU or GPU) where the computations will happen
    # Default to CPU for simplicity (change to GPU if available)
    # device = torch.device('cpu')

    gpu_num = args.gpu_num
    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')


    # Set random seeds to ensure reproducibility of results
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the pre-trained DnCNN model and set it to evaluation mode
    model = DnCNN().to(device)
    model.load_state_dict(torch.load('../experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Set up directories for input images and saving results
    img_dir = os.path.join('../all_datasets/', args.dataset)  # Directory containing test images
    save_dir = os.path.join('./results/', args.dataset)  # Directory to save the denoised images
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)  # Create the directory if it doesn't exist

    # Load all PNG and JPG images from the dataset directory
    img_paths = glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg'))
    imgs = []

    # Preprocess each image (convert to grayscale and pad to a consistent size)
    for p in img_paths:
        img = cv2.imread(p)
        # If the image is in color, convert it to grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pad the image to ensure it has a consistent size (256x256 pixels)
        desired_height = 256
        desired_width = 256
        padded_img = cv2.copyMakeBorder(
            img,
            0,  # Top padding
            max(0, desired_height - img.shape[0]),  # Bottom padding
            0,  # Left padding
            max(0, desired_width - img.shape[1]),  # Right padding
            cv2.BORDER_CONSTANT,  # Padding type
            value=0  # Padding color (black)
        )
        imgs.append(padded_img)  # Add the processed image to the list

    # Split the noise parameter into its type and intensity
    noise_type = args.noise.split('_')[0]  # 'gauss' or 'poisson'
    noise_intensity = float(args.noise.split('_')[1]) / 255.  # Normalize noise intensity

    # Set up image transformations (e.g., normalization)
    transform = transforms.Compose(get_transforms(args))

    # Initialize variables to store evaluation metrics
    noisy_psnr, output_psnr, prediction_psnr, overlap_psnr = 0, 0, 0, 0
    noisy_ssim, output_ssim, prediction_ssim, overlap_ssim = 0, 0, 0, 0

    avg_time1, avg_time2, avg_time3 = 0, 0, 0  # To store average processing times

    # Process each image in the dataset
    for index, clean255 in enumerate(imgs):
        if args.crop:
            clean255 = crop(clean255, patch_size=args.patch_size)  # Crop the image if specified

        clean_numpy = clean255 / 255.  # Normalize the image to have values between 0 and 1

        # Add noise to the image based on the specified noise type and intensity
        if noise_type == 'gauss':
            noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
            noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * args.alpha
        elif noise_type == 'poisson':
            noisy_numpy = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
            noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
        else:
            raise NotImplementedError('wrong type of noise')  # Error if an unsupported noise type is specified

        # Apply transformations and convert the noisy images to tensors
        noisy, noisier = transform(noisy_numpy), transform(noisier_numpy)
        noisy, noisier = torch.unsqueeze(noisy, dim=0), torch.unsqueeze(noisier, dim=0)
        noisy, noisier = noisy.type(torch.FloatTensor).to(device), noisier.type(torch.FloatTensor).to(device)

        # Process the noisy image through the model (currently, this step is bypassed, and the noisy image is used directly)
        start1 = time.time()
        output = noisy  # Uncomment and replace with 'output = model(noisy)' to actually denoise the image
        elapsed1 = time.time() - start1
        avg_time1 += elapsed1 / len(imgs)

        # Generate a prediction from the noisier image using the model
        start2 = time.time()
        prediction = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)
        elapsed2 = time.time() - start2
        avg_time2 += elapsed2 / len(imgs)

        # Optional: Perform overlap prediction if specified
        if args.use_overlap:
            start3 = time.time()
            overlap = perform_overlap_prediction(model, noisier, transform, device, args, clean_numpy, method=1)
            elapsed3 = time.time() - start3
            avg_time3 += elapsed3 / len(imgs)
        else:
            overlap = None  # Skip overlap calculation if not using

        # Convert the outputs back to numpy arrays
        if args.normalize:
            output = denorm(output, mean=args.mean, std=args.std)
            prediction = denorm(prediction, mean=args.mean, std=args.std)
            overlap = denorm(overlap, mean=args.mean, std=args.std) if overlap is not None else None

        output_numpy = output.detach().cpu().numpy()
        prediction_numpy = prediction.detach().cpu().numpy()
        overlap_numpy = overlap.detach().cpu().numpy() if overlap is not None else None

        output_numpy, prediction_numpy = np.squeeze(output_numpy), np.squeeze(prediction_numpy)
        overlap_numpy = np.squeeze(overlap_numpy) if overlap is not None else None

        # Calculate the PSNR (Peak Signal-to-Noise Ratio) for each stage
        n_psnr = psnr(clean_numpy, noisy_numpy, data_range=1)
        o_psnr = psnr(clean_numpy, output_numpy, data_range=1)
        p_psnr = psnr(clean_numpy, prediction_numpy, data_range=1)
        op_psnr = psnr(clean_numpy, overlap_numpy, data_range=1) if overlap is not None else 0

        # Accumulate the PSNR scores
        noisy_psnr += n_psnr / len(imgs)
        output_psnr += o_psnr / len(imgs)
        prediction_psnr += p_psnr / len(imgs)
        overlap_psnr += op_psnr / len(imgs)

        # Calculate the SSIM (Structural Similarity Index) for each stage
        n_ssim = ssim(clean_numpy, noisy_numpy, data_range=1)
        o_ssim = ssim(clean_numpy, output_numpy, data_range=1)
        p_ssim = ssim(clean_numpy, prediction_numpy, data_range=1)
        op_ssim = ssim(clean_numpy, overlap_numpy, data_range=1) if overlap is not None else 0

        # Accumulate the SSIM scores
        noisy_ssim += n_ssim / len(imgs)
        output_ssim += o_ssim / len(imgs)
        prediction_ssim += p_ssim / len(imgs)
        overlap_ssim += op_ssim / len(imgs)

        # Print the metrics for the current image
        print('{}th image | PSNR: noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f} | SSIM: noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
            index+1, n_psnr, o_psnr, p_psnr, op_psnr, n_ssim, o_ssim, p_ssim, op_ssim))

        # Save the first few images (up to 3) to the results directory for visual inspection
        if index <= 3:
            sample_clean, sample_noisy = 255. * np.clip(clean_numpy, 0., 1.), 255. * np.clip(noisy_numpy, 0., 1.)
            sample_output, sample_prediction = 255. * np.clip(output_numpy, 0., 1.), 255. * np.clip(prediction_numpy, 0., 1.)
            sample_overlap = 255. * np.clip(overlap_numpy, 0., 1.) if overlap_numpy is not None else None
            cv2.imwrite(os.path.join(save_dir, '{}th_clean.png'.format(index+1)), sample_clean)
            cv2.imwrite(os.path.join(save_dir, '{}th_noisy.png'.format(index+1)), sample_noisy)
            cv2.imwrite(os.path.join(save_dir, '{}th_output.png'.format(index+1)), sample_output)
            cv2.imwrite(os.path.join(save_dir, '{}th_prediction.png'.format(index+1)), sample_prediction)
            if sample_overlap is not None:
                cv2.imwrite(os.path.join(save_dir, '{}th_overlap.png'.format(index+1)), sample_overlap)

    # After processing all images, print the average PSNR and SSIM scores
    print('{} Average PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
        args.dataset, noisy_psnr, output_psnr, prediction_psnr, overlap_psnr))
    print('{} Average SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
        args.dataset, noisy_ssim, output_ssim, prediction_ssim, overlap_ssim))
    print('Average Time for Output | denoised:{}'.format(avg_time1))
    print('Average Time for Prediction | denoised:{}'.format(avg_time2))
    print('Average Time for Overlap | denoised:{}'.format(avg_time3))

# If the script is run directly, start the process
if __name__ == "__main__":
    generate(opt)
