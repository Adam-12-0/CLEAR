
# OptiX Denoising for UCF-101 Dataset

This branch contains the scripts necessary to denoise a noisy dataset. For instructions on injecting noise, refer to the `noise-injection` branch.

---

## Prerequisites

### 1. NVIDIA GPU (Pascal or Newer)
- A GPU supporting CUDA and OptiX is required.
- **Tested with:** GTX 1080Ti, RTX 2080.

### 2. Linux Environment
- This project must run on Linux due to missing DLL libraries on Windows.
- **Tested OS:** Ubuntu 24.04.

### 3. Latest NVIDIA Driver
- Ensure you have the latest NVIDIA driver for optimal performance.
- **Tested version:** 566.03.

### 4. CUDA
- Required for the OptiX SDK.
- **Tested version:** CUDA 12.6.2.

### 5. OptiX SDK (Version 7.6 or Newer)
- Required to build and run the PyOptiX library.
- **Tested version:** OptiX 8.0.

### 6. PyOptiX Library
- Must be built and installed locally.
- For detailed instructions, see the [PyOptiX GitHub repository](https://github.com/NVIDIA/otk-pyoptix).

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd CLEAR
git checkout optix-denoise
```

### 2. Download UCF-101 Dataset and Inject Noise
- Follow the instructions in the `noise-injection` branch README.
- A copy of the `noise.py` script is also included in this branch.

### 3. Install Dependencies
Install required Python libraries:
```bash
pip install opencv-python cupy numpy argparse tqdm piq scikit-image torchvision
```

### 4. Generate Flow Maps
- Pre-generate flow maps from the clear UCF-101 dataset before running `optix_denoise.py` with the `--flow 2` flag.
- Ensure the clean UCF-101 dataset is available.
- The flow maps will be saved in the `UCF-101_flow` directory.
```bash
python gen_flow.py
```

---

## Running Instructions

To denoise the noisy dataset (stored in the `output` directory):
```bash
python optix_denoise.py --flow 2
```
- **Indirect temporal denoising** will improve visual quality (enabled via --flow 2 option).
- **Output:** Denoised videos are saved in the `output_optix` directory.
- **Metrics:** Performance metrics are stored in `performance.csv`.

---

## Post-Processing

To compute the quality metrics of the denoised output and integrate performance metrics:
```bash
python calculate_metrics.py
```
- **Output:** Final metrics are saved in `metrics.csv`.

---

## Notes
- Metrics and performance directories store results generated with various temporal and tiling denoising options.
- For a complete list of options, run:
```bash
python optix_denoise.py --help
```
