# Fast Non-Local Means Denoising for UCF-101 Dataset

This branch contains the scripts necessary to denoise a noisy dataset. For instructions on injecting noise, refer to the `noise-injection` branch.

---

## Prerequisites
- OpenCV is required for the fastNlMeansDenoisingColored function
- Numpy
---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Adam-12-0/CLEAR.git
cd CLEAR
git checkout fastNLM
```

### 2. Download UCF-101 Dataset and Inject Noise
- Follow the instructions in the `noise-injection` branch README.
- A copy of the `noise.py` script is also included in this branch.

### 3. Install Dependencies
Install required Python libraries:
```bash
pip install opencv-python numpy argparse tqdm piq scikit-image torchvision
```

---
## Running Instructions

To denoise the noisy dataset (stored in the `output` directory):
```bash
python fastNLM.py output/ denoisedVideosFastNLM/ --h 10 --hColor 10 --templateWindowSize 7 --searchWindowSize 21
```
- **Noisy Path:** Path the noise injected videos (Required).
- **Denoised Path:** Path the the denoised videos are written to (Required).   
  - Make 'denoisedVideosFastNLM/' or edit gen_metrics.py script to reflect new path (lines 66 to 69).
- **h:** Regulates the filter strength for luminance (Optional flag, default is 10).
- **hColor:** Regulates the filter strength for color (Optional flag, default is 10).
- **templateWindowSize:** Size in pixels of the template patch (Optional flag, default is 7).
- **searchWindowSize:** Size in pixels of the window used to search for patches (Optional flag, default is 21).
- **Output:** Denoised videos are saved in the directory of the denoised path argument passed in.
---

## Post-Processing

To compute the quality metrics of the denoised output and integrate performance metrics:
```bash
python gen_metrics.py
```
- **Output:** Final metrics are saved in `./metrics/{noise}/metrics_summary.csv`.

To create the plots for the metrics:
```bash
python plot_metrics.py
```
- **Output:** Final plots are saved in `./metrics/graphs/{noise}/`.
---

## Notes
- Metrics and performance directories store results generated with various temporal and tiling denoising options.
- For a complete list of options, run:
```bash
python fastNLM.py --help
```
