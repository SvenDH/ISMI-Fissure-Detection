#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -p gpu
module load python/3.5.2
pip install tensorflow-gpu==1.4.1 --user
pip install keras --user
pip install tqdm --user
pip install requests --user
pip install h5py --user
pip install SimpleITK --upgrade --user

module load cuda/8.0.44
module load cudnn/8.0-v6.0
module load gcc/4.9.2

#$SLURM_SUBMIT_HOST

python3 main.py
