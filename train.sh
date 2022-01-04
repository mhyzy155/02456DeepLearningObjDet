#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=2"
#BSUB -J train_job
#BSUB -n 2
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# Load modules
module load python3/3.7.11
module load cuda/10.2

echo "Running script..."
python3 train.py

 
