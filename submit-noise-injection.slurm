#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --error=Noise-Injection-%J.err
#SBATCH --output=Noise-Injection-%J.out
#SBATCH --job-name=pytorch-Noise-Injection

# load anaconda and cuda modules
module load anaconda
conda activate pytorch2.2.0+py3.11+cuda12.1
pip install opencv-python-headless numpy
module load cuda/cuda-12.2.2 

# list the modules 
module list

echo "---Start---"
time python noise.py
echo "----End----"
echo 