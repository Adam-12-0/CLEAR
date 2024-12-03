# Noise Injection for UCF-101 Dataset

This branch contains the code to inject various types of noise into the UCF-101 video dataset for further denoising analysis.

## Setup Instructions

### 1. Clone the Repository
First, clone the repository and switch to the `noise` branch:

```bash
git clone https://github.com/Adam-12-0/CLEAR.git
cd CLEAR
git checkout noise
```

### 2. Setup the Conda Environment
Create and activate a new conda environment called `CLEAR`:

```bash
conda create -n CLEAR python=3.8
conda activate CLEAR
```

### 3. Install Dependencies
Install the required Python libraries, including OpenCV, NumPy, and others:

```bash
pip install opencv-python-headless numpy
```

### 4. Place Dataset in the Correct Directory
- Download the UCF-101 dataset from [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar).
- Extract the dataset and place it in the root folder of the project, so that the path is `./UCF-101/`.

### 5. Run the Noise Injection Script
To inject noise into the dataset, use the following command:

```bash
python noise.py
```

For a test run, use the following command to process only a single action category:

```bash
python noise.py --test
```

## Notes
- The dataset will be processed into four different noisy versions: Gaussian, Salt & Pepper, Poisson, and Speckle.
- Each type of noise will be saved in a separate folder named `UCF-101-(Noise Type)`.

## Command to Run the Script
- **To run the script and generate the noisy datasets**:

  ```bash
  python noise.py
  ```

- **If you are performing a test run**:

  ```bash
  python noise.py --test
  ```
