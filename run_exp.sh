#!/bin/bash

# Assign the first command argument to a variable
NUMBER=$1

# First command
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /home/mikylab/datasets/cyto_full/bone_marrow_cell_dataset/classes6/training/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --num_iters 300000 --lambda_id 10 --batch_size 16 \


python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /home/mikylab/datasets/cyto_full/bone_marrow_cell_dataset/classes6/training/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --test_iters 300000 --lambda_id 10 --batch_size 1 \
               --dist_file_name experiments/CYTO_${NUMBER}/CYTO_train_dist${NUMBER}.csv

# Second command
python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir /home/mikylab/datasets/cyto_full/bone_marrow_cell_dataset/classes6/testing/ \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --test_iters 300000 --lambda_id 0 --batch_size 1 \
               --dist_file_name experiments/CYTO_${NUMBER}/CYTO_test_dist${NUMBER}.csv

# Third command
python3 svmTest.py --train_dist_dir experiments/CYTO_${NUMBER}/CYTO_train_dist${NUMBER}.csv \
                   --test_dist_dir experiments/CYTO_${NUMBER}/CYTO_test_dist${NUMBER}.csv \
                   --svm linear --result_dir experiments/CYTO_${NUMBER}/svm300
