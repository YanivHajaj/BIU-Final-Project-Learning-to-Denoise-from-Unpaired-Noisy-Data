# Train the model
import sys
# Add the directory containing your module to the Python path
sys.path.append('Noisier2Noise-main')

from train import TrainNr2N

import argparse

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
    mean=0.4050,  # Mean value for normalization
    std=0.2927  # Standard deviation for normalization
)


train_instance = TrainNr2N(args)
train_instance.train()