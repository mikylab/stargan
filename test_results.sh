#!/bin/bash

# Assign the first command argument to a variable
NUMBER=$1

# First command
python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 3 --rafd_image_dir /home/mikylab/datasets/STAR_ISIC/train/ \
               --sample_dir experiments/ISIC_${NUMBER}/samples \
               --log_dir experiments/ISIC_${NUMBER}/logs \
               --model_save_dir experiments/ISIC_${NUMBER}/models \
               --result_dir experiments/ISIC_${NUMBER}/results \
               --test_iters 300000 --lambda_id 0 --batch_size 1 \
               --dist_file_name experiments/ISIC_${NUMBER}/ISIC_train_dist${NUMBER}.csv

# Second command
python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 3 --rafd_image_dir /home/mikylab/datasets/STAR_ISIC/test/ \
               --sample_dir experiments/ISIC_${NUMBER}/samples \
               --log_dir experiments/ISIC_${NUMBER}/logs \
               --model_save_dir experiments/ISIC_${NUMBER}/models \
               --result_dir experiments/ISIC_${NUMBER}/results \
               --test_iters 300000 --lambda_id 0 --batch_size 1 \
               --dist_file_name experiments/ISIC_${NUMBER}/ISIC_test_dist${NUMBER}.csv

# Third command
python3 svmTest.py --train_dist_dir experiments/ISIC_${NUMBER}/ISIC_train_dist${NUMBER}.csv \
                   --test_dist_dir experiments/ISIC_${NUMBER}/ISIC_test_dist${NUMBER}.csv \
                   --svm linear --result_dir experiments/ISIC_${NUMBER}/svm300
