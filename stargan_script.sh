#!/bin/bash
#SBATCH --job-name=stargan           # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --tasks-per-node=1           # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs required
#SBATCH --output=stargan.out         # Standard output log file
#SBATCH --error=stargan.err          # Standard error log file
#SBATCH --partition=day-long-gpu 

# Load CUDA module
module load cuda12.2/toolkit/12.2.2

# Activate Conda environment
source activate ../miniconda3/envs/stargan_env

# CUDA program
# Assign the first command argument to a variable
NUMBER=$1

mkdir -p experiments/CYTO_${NUMBER} && \
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/train6/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
	       --num_iters 100000 --lambda_id .001 --batch_size 24 > experiments/CYTO_${NUMBER}/output_${NUMBER}.txt

python main.py --mode 'test' --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/train6/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --test_iters 100000 --lambda_id .001 --batch_size 1 \
	       --dist_file_name experiments/CYTO_${NUMBER}/CYTO_train_dist${NUMBER}.csv > experiments/CYTO_${NUMBER}/test_output_${NUMBER}.txt

python main.py --mode 'test' --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/test6/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --test_iters 100000 --lambda_id .001 --batch_size 1 \
               --dist_file_name experiments/CYTO_${NUMBER}/CYTO_test_dist${NUMBER}.csv >> experiments/CYTO_${NUMBER}/test_output_${NUMBER}.txt

python svmTest.py --train_dist_dir experiments/CYTO_${NUMBER}/CYTO_train_dist${NUMBER}.csv \
                   --test_dist_dir experiments/CYTO_${NUMBER}/CYTO_test_dist${NUMBER}.csv \
                   --svm linear --result_dir experiments/CYTO_${NUMBER}/svm100

#python main.py --mode train --dataset RaFD --image_size 128 --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/train6/ --sample_dir experiments/samples  --log_dir experiments/logs --model_save_dir experiments/models --result_dir experiments/results --resume_iters 30000 --num_iters 300000 --lambda_id .001 --batch_size 24 > output.txt


# Test program 
#python main.py --mode test --dataset RaFD --image_size 128 --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/test6/ --sample_dir experiments/samples  --log_dir experiments/logs --model_save_dir experiments/models --result_dir experiments/results --test_iters 300000 --lambda_id .001 --batch_size 1 --dist_file_name experiments/cyto_42424_test > test_output.txt


#python main.py --mode test --dataset RaFD --image_size 128 --c_dim 6 --rafd_image_dir /nfs/home/mbowen/datasets/bone_marrow_cell_dataset/train6/ --sample_dir experiments/samples  --log_dir experiments/logs --model_save_dir experiments/models --result_dir experiments/results --test_iters 300000 --lambda_id .001 --batch_size 1 --dist_file_name experiments/cyto_42424_train >> test_output.txt

#python svmTest.py --train_dist_dir experiments/cyto_42424_train.csv --test_dist_dir experiments/cyto_42424_test.csv svm linear --result_dir experiments/cyto_42424_svm300 > svm_output.txt


