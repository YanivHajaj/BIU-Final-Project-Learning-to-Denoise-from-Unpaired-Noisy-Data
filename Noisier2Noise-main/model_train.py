import sys
import argparse
from utils import convert_to_grayscale, calculate_mean_std

# Setup Python path
sys.path.append('Noisier2Noise-main')
from train import TrainNr2N

# Path configuration
source_folder = '../all_datasets/ImageNet_1000_Gray/train'
target_folder = '../all_datasets/ImageNet_1000_Gray/train'

# # Step 1: Convert images to grayscale (Run once)
# convert_to_grayscale(source_folder, target_folder)

# Step 2: Calculate mean and standard deviation of the grayscale images
mean, std = calculate_mean_std(target_folder)

# Define args using argparse.Namespace for attribute-like access
args = argparse.Namespace(
    exp_detail='Train Nr2N public',
    gpu_num=0,  # Use CPU, or set to 0 if you have GPU support
    seed=100,  # For reproducibility
    load_model=False,  # Start training from scratch
    load_exp_num=1,
    load_epoch=500,
    n_epochs=200,  # Reduce the number of epochs to prevent overfitting on a small dataset
    start_epoch=0,
    decay_epoch=50,  # Start decaying the learning rate earlier to prevent overfitting
    batch_size=4,  # Keep batch size small due to the small dataset size
    lr=1e-4,  # Reduce the learning rate to allow the model to learn more gradually
    noise='gauss_25',  # Type and intensity of noise
    crop=True,  # Enable cropping if your images are larger than the patch size
    patch_size=256,  # The size of patches to use
    normalize=True,  # Normalize the images
    mean=mean,  # Updated mean  // on 1000 cars is 0.40976835593231453
    std=std  # Updated standard deviation on 1000 cars is 0.2719549487997949
    #mean = 0.4050,  # Mean value for normalization
    #std = 0.2927  # Standard deviation for normalization
)

print(f'the mean is : {mean},the std is :{std}')
# # Initialize and train the model
# train_instance = TrainNr2N(args)
# train_instance.train()
