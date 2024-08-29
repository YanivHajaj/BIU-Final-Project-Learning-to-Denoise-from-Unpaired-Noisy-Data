# BIU-University-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data
Final project BIU - Computer Engineering
- Samy Nehmad (samy.nehmad1@gmail.com)
- Yaniv Hajaj (yaniv3456@gmail.com)

# Noisier2Noise: Learning to Denoise from Unpaired Noisy Data


## Useful Information (Internal use for training)
Article:
'https://arxiv.org/pdf/1910.11908'

Google Collab:
'https://colab.research.google.com/drive/1sJZCjDE0pUqGdUy00lxeBeDjobP2RBTf?hl=en#scrollTo=p-5QGYTVRN6Y'


### Running from Collab:
1. Clone the repo
```bash
!git clone https://github.com/YanivHajaj/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data.git
```

2. Navigate on the OS (if needed)
``` python
# navigate on the OS
import os

# Navigate to the project directory
project_path = 'BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data/Noisier2Noise-main'
os.chdir(project_path)

# List contents of the current directory
print("Contents of project directory:", os.listdir('.'))
```

3. pip install the needed libraries
```bash
!pip install -r requirements.txt
```

### Push from Collab:

1. Generate a Personal Access Token (PAT) on GitHub:
Log in to GitHub:

Go to GitHub and log in with your credentials.
Navigate to Settings:

Click on your profile picture in the top-right corner.
Select Settings from the dropdown menu.
Access Developer Settings:

Scroll down the left-hand menu and click on Developer settings.
Generate a New Token:

Click on Personal access tokens in the left sidebar.
Click on Tokens (classic) and then click on Generate new token.
Give your token a descriptive name, like "Google Colab Push".
Select Scopes:

Choose the scopes or permissions you want the token to have. To push to a repository, you’ll need to select repo (full control of private repositories). You can also include other scopes if necessary.
Click on Generate token.
Copy the Token:

GitHub will display the token once. Copy it immediately and store it securely. You won’t be able to view it again later.

2. Run push command on Collab (after cloning the project)
```bash
!git push https://<YOUR_PAT>@github.com/YanivHajaj/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data.git
```




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
   git clone https://github.com/YanivHajaj/BIU-Final-Project-Learning-to-Denoise-from-Unpaired-Noisy-Data.git
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
    --exp_detail "Training with Gaussian noise intensity 25 on ImageNetGray" \
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

Or edit the file `model_train.py` with the desired parameters and run it `python model_train.py`

Note that for the time being the dataset used for training is all_datasets/ImageNet_1000_Gray/train/

### Testing the Model

Once you have trained your Noisier2Noise model, you can evaluate its performance on a test dataset to see how well it can denoise new images. This section will guide you on how to prepare the test data and run the test script.

Preparing the Test Data
Create a Dataset Folder:

Inside the all_datasets directory, create a new folder for your test dataset. For example, if you want to name your dataset Set19, the path would be ./all_datasets/Set19.
Add Test Images:

Place your test images inside the Set19 folder. Ensure that all images are in PNG format.
These images should be clean (noise-free) if you want to evaluate the model's performance against a ground truth. Alternatively, you can use noisy images to simply observe the denoising capability of the model.
Running the Test Script
To test the model on the Set19 dataset, use the following command in your terminal or command prompt:

Run the command (for example)
   ```bash
  python test.py --exp_num 6 --n_epochs 200 --gpu_num 0 --dataset Set20
   ```

--exp_num: This should match the experiment number where your trained model is saved. For example, if your model is saved under exp2, set --exp_num to 2.
--n_epochs: This should be the number of epochs at which your model was saved. If your checkpoint is saved at 500 epochs, set --n_epochs to 500.
--gpu_num: Set this to 0 to use the first GPU. If you are running the test on a CPU, you can ignore this parameter.
--dataset: Set this to the name of the folder where your test images are stored (Set19).

An example containing all the parameters:

```bash
python test.py \
    --test_info "Running Set12 with Gaussian noise intensity 25" \
    --gpu_num 0 \
    --seed 90 \
    --exp_num 10 \
    --n_epochs 180 \
    --noise "gauss_25" \
    --dataset "Set12" \
    --exp_rep "rep1" \
    --aver_num 10 \
    --alpha 1.0 \
    --trim_op 0.05 \
    --crop True \
    --patch_size 256 \
    --normalize True \
    --mean 0.4097 \
    --std 0.2719
```

### Explanation of Key Arguments

#### For `train.py`:

- **`--exp_detail`**: A description of the experiment for logging and tracking purposes. Example: `"Train Nr2N public"`.
- **`--gpu_num`**: The GPU index to use. Default is `0`, which refers to the first GPU.
- **`--seed`**: Seed for random number generation to ensure reproducibility. Default is `100`.
- **`--load_model`**: Boolean flag indicating whether to load a pre-trained model. Default is `False`.
- **`--load_exp_num`**: The experiment number of the pre-trained model to load. Used in conjunction with `--load_model`. Default is `1`.
- **`--load_epoch`**: The specific epoch of the pre-trained model to load. Used in conjunction with `--load_model`. Default is `500`.
- **`--n_epochs`**: Number of epochs to train the model. Default is `500`.
- **`--start_epoch`**: The epoch to start training from. Useful when continuing training from a checkpoint. Default is `0`.
- **`--decay_epoch`**: The epoch after which the learning rate starts decaying. Default is `150`.
- **`--batch_size`**: Number of samples per batch. Default is `4`.
- **`--lr`**: Learning rate for the optimizer. Default is `0.001`.
- **`--noise`**: Specifies the type and intensity of noise applied during training. Format: `'noise_type_intensity'`. Example: `'gauss_25'` or `'poisson_50'`. Default is `'gauss_25'`.
- **`--crop`**: Boolean indicating whether to crop the images during training. Default is `True`.
- **`--patch_size`**: Size of the image patches to use during training. Default is `256`.
- **`--normalize`**: Boolean indicating whether to normalize the image data. Default is `True`.
- **`--mean`**: Mean value used for normalization. Default is `0.4050`.
- **`--std`**: Standard deviation used for normalization. Default is `0.2927`.

#### For `test.py`:

- **`--test_info`**: A string providing additional information about the test run. Default is `'None info given'`.
- **`--gpu_num`**: The GPU index to use. Default is `0`, which refers to the first GPU.
- **`--seed`**: Seed for random number generation to ensure reproducibility. Default is `90`.
- **`--exp_num`**: Experiment number to identify the specific configuration used for the test. Default is `10`.
- **`--n_epochs`**: Number of epochs for which the model was trained. Default is `180`.
- **`--noise`**: Specifies the type and intensity of noise applied during the test. Format: `'noise_type_intensity'`. Example: `'gauss_25'` or `'poisson_50'`. Default is `'gauss_25'`.
- **`--dataset`**: Name of the dataset used for testing. Examples include `Set12`, `BSD100`, and `Kodak`. Default is `Set12`.
- **`--exp_rep`**: Experiment repetition identifier. This is an optional string that can be used to distinguish different runs of the same experiment. Default is `None`.
- **`--aver_num`**: Number of averages to be used during testing to stabilize the performance evaluation. Default is `10`.
- **`--alpha`**: A scaling factor for the synthetic noise intensity added during the test. Default is `1.0`.
- **`--trim_op`**: A float value representing the trimming operation parameter. Used for trimming outliers in the overlap operations. Default is `0.05`.
- **`--crop`**: Boolean indicating whether to crop the images during testing. Default is `True`.
- **`--patch_size`**: Size of the image patches to use during testing. Default is `256`.
- **`--normalize`**: Boolean indicating whether to normalize the image data. Default is `True`.
- **`--mean`**: Mean value used for normalization. Default is `0.4097`.
- **`--std`**: Standard deviation used for normalization. Default is `0.2719`.


## Results and Evaluation

The Noisier2Noise method has been tested on various datasets, including the Kodak image set, and compared against other denoising methods such as Noise2Noise and BM3D. The results demonstrate that Noisier2Noise can achieve competitive performance, particularly in scenarios where obtaining clean images is impractical.

### How to Evaluate (run `test.py`)

#### Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the denoised image relative to the clean image. Higher values indicate better quality.
- **SSIM (Structural Similarity Index)**: Evaluates the structural similarity between the denoised and clean images. It considers changes in structural information, contrast, and luminance.

#### Results Structure

The results folder contains a variety of output images that illustrate the effectiveness of the Noisier2Noise approach. The structure is as follows:

```bash
project/
└── results/
    ├── results_summary.txt    # Summary of all results
    ├── Set20/                 # Dataset-specific results
    │   ├── imgs/                  # Directory containing result images
    │   │   ├── exp5/                  # Experiment 5 results
    │   │   │   ├── 1th_clean.png           # Clean image (reference)
    │   │   │   ├── 1th_noisy.png           # Noisy image used as input
    │   │   │   ├── 1th_prediction.png      # Denoised output from the model
    │   │   │   ├── 1th_overlap_mean.png    # Overlap of predictions (mean operation)
    │   │   │   ├── 1th_overlap_median.png  # Overlap of predictions (median operation)
    │   │   │   ├── 1th_overlap_trimmed.png # Overlap of predictions (trimmed mean)
    │   │   │   ├── ...                     # More images with similar naming
    │   ├── csvs/                 # Directory containing CSV files for metrics
    │   │   ├── exp5/                 # CSV files for experiment 5
    │   │   │   ├── PSNR_1_1th_clean.csv      # PSNR metrics
    │   │   │   ├── SSIM_1_1th_clean.csv      # SSIM metrics
    │   │   │   ├── PSNR_all_images_average.csv # Average PSNR across all images
    │   │   │   ├── SSIM_all_images_average.csv # Average SSIM across all images
    │   ├── graphs/               # Directory for storing generated graphs
    │   │   ├── exp5/                 # Graphs for experiment 5
    │   │   │   ├── psnr_graph.png            # Graph of PSNR across different images
    │   │   │   ├── ssim_graph.png            # Graph of SSIM across different images
    │   │   │   └── ...                       # Additional graphs as needed
```

#### Visual Results

The following images showcase the denoising capabilities of the Noisier2Noise model. Each step highlights the transformation from noisy input to the refined outputs using various aggregation methods.

- **Clean Image**: The original clean image, used as the reference for evaluation.
- **Noisy Image**: The input image with added noise.
- **Prediction Image**: The output from the Noisier2Noise model after denoising.
- **Overlap Mean Image**: The result of averaging multiple denoised outputs.
- **Overlap Median Image**: The result of taking the median of multiple denoised outputs.
- **Overlap Trimmed Mean Image**: The result of using a trimmed mean approach on multiple denoised outputs.

#### Graphs
The results also include graphs that provide a visual summary of the performance metrics:

- **PSNR Graph**: A graph showing the PSNR values for each image in the dataset.
- **SSIM Graph**: A graph showing the SSIM values for each image in the dataset.

## References

- Moran, N., Schmidt, D., Zhong, Y., & Coady, P. (2019). Noisier2Noise: Learning to Denoise from Unpaired Noisy Data. *Algorithmic Systems Group, Analog Devices*.
- Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). Noise2Noise: Learning Image Restoration without Clean Data. arXiv preprint arXiv:1803.04189.

## Contributing

Contributions to the Noisier2Noise project are welcome. If you have any suggestions, bug reports, or feature requests, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

