#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --error=FastNLM-%J.err
#SBATCH --output=FastNLM-%J.out
#SBATCH --job-name=FastNLM

# load anaconda and cuda modules
module load anaconda
conda activate pytorch2.2.0+py3.11+cuda12.1
conda install piq -c photosynthesis-team -c conda-forge -c pytorch
pip install opencv-python-headless numpy
module load cuda/cuda-12.2.2 

# list the modules 
module list

echo "---Start---"
time python fastNLM.py output/ denoisedVideos/
echo "----End----"
echo 
