# CLEAR - Comparative Learning and Evaluation of AI and Traditional Denoisers

## Overview
CLEAR (Comparative Learning and Evaluation of AI and Traditional Denoisers) is a project that compares various traditional and AI-based video denoising techniques using the UCF-101 dataset. The project aims to determine the optimal balance between denoising quality and computational efficiency across different types of video noise.

## Project Members
- **Adam Bawatneh**: Project Manager & Noise Injection Lead
- **Scott Spicer**: Fast Non-Local Means Denoising
- **Zengyan Wang**: Median Filtering and Bilateral Filtering
- **Jacob Braun**: VRT AI Denoiser
- **Martin Dinkov**: NVIDIA OptiX AI-Accelerated Denoiser

## Denoising Techniques
We are evaluating the following denoising techniques:
1. **Fast Non-Local Means Denoising (fastNlMeansDenoisingColored)**
2. **Median Filtering (medianBlur)**
3. **Bilateral Filtering (bilateralFilter)**
4. **VRT (Video Restoration Transformer)**
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
Clone the repository from GitHub:

```bash
git clone https://github.com/Adam-12-0/CLEAR.git
