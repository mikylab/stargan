#!/bin/bash
#SBATCH --job-name=stargan           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --tasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs required
#SBATCH --output=stargan.out         # Standard output log file
#SBATCH --error=stargan.err          # Standard error log file
#SBATCH --partition=short-gpu


# Load CUDA module
module load cuda12.2/toolkit/12.2.2

# Activate Conda environment
source activate ../miniconda3/envs/stargan_env

# CUDA program
python main.py --mode train --dataset RaFD --image_size 128 --c_dim 6 --rafd_image_dir /datasets/cyto_full/bone_marrow_cell_dataset/training/ --sample_dir experiments/samples  --log_dir experiments/logs --model_save_dir experiments/models --result_dir experiments/results --num_iters 300000 --lambda_id 0.001 --batch_size 16
