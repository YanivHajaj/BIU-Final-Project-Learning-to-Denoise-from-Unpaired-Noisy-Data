# BIU-University-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data
Our final project

# Noisier2Noise
Self-Supervised Image Denoising
- Paper Summary
'https://www.notion.so/2020-Noisier2Noise-e87b5fb59368443ca4e203dfe4c7e9a5'

Article:
'https://arxiv.org/pdf/1910.11908'


Google Collab:
'https://colab.research.google.com/drive/1sJZCjDE0pUqGdUy00lxeBeDjobP2RBTf?hl=en#scrollTo=p-5QGYTVRN6Y'

1. Train Dataset: ImageNet 1000
2. Test Dataset: BSD100, Kodak, Set12


# Train:
Run the command 
   ```bash
   python Noisier2Noise-main/train.py \
    --exp_detail "Train Nr2N public" \
    --gpu_num 0 \
    --seed 100 \
    --load_model False \
    --load_exp_num 1 \
    --load_epoch 500 \
    --n_epochs 500 \
    --start_epoch 0 \
    --decay_epoch 150 \
    --batch_size 4 \
    --lr 0.001 \
    --noise "gauss_25" \
    --crop True \
    --patch_size 256 \
    --normalize True \
    --mean 0.4050 \
    --std 0.2927
   ```

# Test
Run the command 
   ```bash


   ```











Here's a comprehensive README file for the `Noisier2Noise` project, which includes theoretical information, usage examples, and instructions on how to run the code.

Complete it where needed!! Check all the information in here!


# Noisier2Noise: Learning to Denoise from Unpaired Noisy Data

## Introduction

**Noisier2Noise** is a novel approach for training neural networks to perform image denoising without requiring clean training examples or paired noisy examples. This method is particularly useful in situations where collecting clean images is difficult or impossible. The core idea behind Noisier2Noise is to add synthetic noise to an already noisy image and train the network to predict the original noisy image. By doing this, the model learns to differentiate between the noise present in the image and the additional synthetic noise, enabling it to better reconstruct the clean image.

### Key Features:
- **Unpaired Denoising**: The method only requires a single noisy realization of each training example.
- **Versatile Noise Models**: Applicable to a variety of noise models, including Gaussian and Poisson noise.
- **Competitive Performance**: The method achieves results close to those of methods requiring richer training data.

## Theoretical Background

### Motivation

In typical image denoising tasks, a neural network is trained using pairs of noisy and clean images. The network learns to map noisy inputs to their clean counterparts. However, in many real-world scenarios, obtaining clean images is infeasible. The Noisier2Noise method overcomes this limitation by leveraging the fact that given two noisy images of the same scene, the noise can be assumed to be independent. The network learns to denoise by predicting the original noise distribution from the combined noisy inputs.

### Mathematical Justification

Given a noisy image \( Y = X + N \) (where \( X \) is the clean image and \( N \) is noise), the Noisier2Noise method creates a doubly noisy version \( Z = Y + M \), where \( M \) is additional synthetic noise. The network is trained to predict \( Y \) from \( Z \), and the final clean image estimate \( \hat{X} \) is derived by:
\[ \hat{X} = 2E[Y|Z] - Z \]
This approach helps in recovering the clean image \( X \) by using the synthetic noise to guide the network in identifying the original noise.


## Improvements in the Noisier2Noise Method

The Noisier2Noise approach builds upon existing denoising techniques by introducing several key improvements. These improvements are designed to enhance the model's ability to denoise images effectively, even when only noisy data is available. Below is a summary of the main enhancements added to the method:

### 1. **Improved Handling of Gaussian Noise:**
   - **Synthetic Noise Addition:** The method involves adding synthetic Gaussian noise to already noisy images during training. This synthetic noise is carefully scaled using a parameter `α` (alpha), which helps the model to better distinguish between the original noise and the added synthetic noise.
   - **Correction Step:** After predicting the noisy image from the doubly noisy input, a correction step is applied to derive the clean image estimate. The formula used is:
     \[
     \hat{X} = \frac{(1 + \alpha^2) \cdot \text{model(noisier)} - \text{noisier}}{\alpha^2}
     \]
     This step effectively reduces the noise in the prediction and brings the output closer to the true clean image.

### 2. **Support for Non-Additive Noise Models:**
   - The method has been extended to handle non-additive noise models, such as multiplicative Bernoulli noise. This is achieved by modifying the training process and the correction step accordingly.
   - For example, in the case of Poisson noise, the method first adds Poisson noise to the image and then applies a Gaussian approximation to further refine the prediction.

### 3. **Adaptation of Noise Intensity:**
   - The noise intensity for the added synthetic noise can be adjusted during training using the parameter `α`. Lower values of `α` provide a clearer view of the noisy image but require careful correction to avoid magnifying errors.
   - The method allows for quick fine-tuning of the model with different values of `α`, enabling rapid exploration of the best settings for a given dataset and noise model.

### 4. **Optional Noise Model during Inference:**
   - During inference (testing), the method offers the option to add noise of a different intensity than what was used during training. This feature can improve the model's performance by providing a clearer input, which is closer to the singly noisy image rather than the doubly noisy one.
   - The approach can be beneficial when the primary focus is on optimizing metrics like PSNR, where reduced noise during inference helps achieve better scores.

### 5. **Structured Noise Handling:**
   - The method is also designed to handle structured noise, where noise is not independently and identically distributed (i.i.d.) across pixels but has some spatial correlation.
   - This enhancement makes the Noisier2Noise method applicable to a wider range of real-world scenarios where noise often exhibits spatial patterns.

### 6. **Fine-tuning for Various Noise Models:**
   - The Noisier2Noise method can be fine-tuned for different noise models, including Gaussian, Poisson, and Bernoulli noise. This flexibility allows the model to be adapted for specific applications, improving its robustness and effectiveness in diverse settings.

These improvements enable the Noisier2Noise method to perform well across a variety of noise conditions, making it a powerful tool for image denoising even in challenging scenarios where clean training data is unavailable.


## Installation

To use the Noisier2Noise code, you'll need to have Python and the necessary libraries installed.

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Scikit-image

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Noisier2Noise.git
   cd Noisier2Noise-main
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have the datasets in the correct directory structure as outlined in the code.

## Usage

### Training the Model

To train the model using the Noisier2Noise approach, you can use the `train.py` script. Below is an example command to start training:

```bash
python train.py \
    --exp_detail "Training Noisier2Noise Model" \
    --gpu_num 0 \
    --seed 100 \
    --load_model False \
    --n_epochs 500 \
    --batch_size 4 \
    --lr 0.001 \
    --noise "gauss_25" \
    --crop True \
    --patch_size 256 \
    --normalize True \
    --mean 0.4050 \
    --std 0.2927
```

### Testing the Model

Once the model is trained, you can test its performance on a dataset using the `test.py` script. Below is an example of how to run the test script:

```bash
python test.py \
    --gpu_num 0 \
    --seed 100 \
    --exp_num 10 \
    --n_epochs 180 \
    --noise "poisson_50" \
    --dataset "Set12" \
    --aver_num 10 \
    --alpha 1.0 \
    --crop True \
    --patch_size 256 \
    --normalize True \
    --mean 0.4050 \
    --std 0.2927
```

This command will load the trained model, apply it to the specified dataset, and output the denoised images along with performance metrics such as PSNR and SSIM.

### Example Code

Here is an example of how to set up and train the model programmatically within a Python script:

```python
import sys
sys.path.append('Noisier2Noise-main')

from train import TrainNr2N
import argparse

args = argparse.Namespace(
    exp_detail='Train Nr2N public',
    gpu_num=0,
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
```

### Explanation of Key Arguments

- **`exp_detail`**: Description of the experiment.
- **`gpu_num`**: The GPU index to use (set to `0` for the first GPU).
- **`seed`**: Seed for random number generation for reproducibility.
- **`load_model`**: Whether to load a pre-trained model.
- **`n_epochs`**: Number of epochs to train the model.
- **`batch_size`**: Number of samples per batch.
- **`lr`**: Learning rate for the optimizer.
- **`noise`**: The type and intensity of noise to apply (e.g., `gauss_25` for Gaussian noise with a standard deviation of 25).
- **`crop`**: Whether to crop the images during training.
- **`patch_size`**: Size of the image patches to use for training.
- **`normalize`**: Whether to normalize the image data.
- **`mean` and `std`**: Mean and standard deviation for normalization (based on the ImageNet dataset).

## Results and Evaluation

The Noisier2Noise method has been tested on various datasets, including the Kodak image set, and compared against other denoising methods such as Noise2Noise and BM3D. The results show that Noisier2Noise can achieve competitive performance, particularly in scenarios where obtaining clean images is impractical.

### Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the denoised image relative to the clean image.
- **SSIM (Structural Similarity Index)**: Evaluates the structural similarity between the denoised and clean images.

### Visual Results

The following images demonstrate the effectiveness of the Noisier2Noise approach:
- **Clean Image**: The original clean image (not used in training).
- **Noisy Image**: The input image with noise added.
- **Denoised Image**: The output from the Noisier2Noise model.

## References

- Moran, N., Schmidt, D., Zhong, Y., & Coady, P. (2019). Noisier2Noise: Learning to Denoise from Unpaired Noisy Data. *Algorithmic Systems Group, Analog Devices*.
- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). Noise2Noise: Learning Image Restoration without Clean Data. arXiv preprint arXiv:1803.04189.

## Contributing

Contributions to the Noisier2Noise project are welcome. If you have any suggestions, bug reports, or feature requests, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
