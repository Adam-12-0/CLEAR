# CLEAR - Comparative Learning and Evaluation of AI and Traditional Denoisers
# These directions are for running the VRT branch

## Overview
CLEAR (Comparative Learning and Evaluation of AI and Traditional Denoisers) is a project that compares various traditional and AI-based video denoising techniques using the UCF-101 dataset. The project aims to determine the optimal balance between denoising quality and computational efficiency across different types of video noise.

VRT is implemented in this repo: https://github.com/JingyunLiang/VRT

## Project Members
- **Adam Bawatneh**: Project Manager & Noise Injection Lead
- **Scott Spicer**: Fast Non-Local Means Denoising
- **Zengyan Wang**: Median Filtering and Bilateral Filtering
- **Jacob Braun**: VRT AI Denoiser
- **Martin Dinkov**: NVIDIA OptiX AI-Accelerated Denoiser

## Denoising Techniques
We are evaluating the following denoising techniques:
1. **VRT Denoising (Video Restoration Transformer)**
2. **Median Filtering (medianBlur)**
3. **Bilateral Filtering (bilateralFilter)**
4. **FFDNet (Fast and Flexible Denoising Convolutional Neural Network)**
5. **NVIDIA OptiX AI-Accelerated Denoiser**

## Noise Types Injected
The following types of noise are added to the videos:
- **Gaussian Noise**
- **Salt-and-Pepper Noise**
- **Poisson Noise**
- **Speckle Noise**

## Dataset
The UCF-101 dataset was used for all experiments. You can download the dataset from the following link:
[UCF-101 Dataset Download](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

Please download and extract the dataset to a suitable directory on your system.

## Setup Instructions

### Step 1: Clone the Repository
Clone the repository from GitHub and run the python file:

```bash
git clone https://github.com/Adam-12-0/CLEAR-Denoising-Project.gi

# Install dependencies
pip install -r requriements.txt

# Median filter and parameters
python Median_bilateral_denoiser.py -input_dir /path/to/noisy/videos/ -output_dir /path/to/save/ -filter_type 'median' --kernel_size 3

# Bilateral filter and parameters
python Median_bilateral_denoiser.py -input_dir /path/to/noisy/videos/ -output_dir /path/to/save/ -filter_type 'bilateral' --d 9 --sigma_color 75 --sigma_space 75
