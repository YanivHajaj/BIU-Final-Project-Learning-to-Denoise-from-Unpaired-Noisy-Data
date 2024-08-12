# Train the model
import sys
# Add the directory containing your module to the Python path
sys.path.append('Noisier2Noise-main')

from train import TrainNr2N

import argparse

# Define args using argparse.Namespace for attribute-like access
args = argparse.Namespace(
    exp_detail='Train Nr2N public',
    gpu_num=-1,
    seed=100,
    load_model=False,
    load_exp_num=1,
    load_epoch=500,
    n_epochs=500,
    start_epoch=0,
    decay_epoch=150,
    batch_size=4,
    lr=1e-3,
    noise='gauss_25',
    crop=True,
    patch_size=256,
    normalize=True,
    mean=0.4050,
    std=0.2927
)


train_instance = TrainNr2N(args)
train_instance.train()