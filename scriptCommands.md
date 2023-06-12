## Training Command Prompt
python main.py --mode train --dataset RaFD --image_size 128\
               --c_dim 7 --rafd_image_dir **data/facialExp256/train** \
               --**sample_dir** stargan_identity.00001/samples --log_dir stargan_identity.00001/logs \
               --model_save_dir stargan_identity.00001/models --result_dir stargan_identity.00001/results
               **--lambda_id .00001 --num_iters 400000**

Bolded commands are important pieces to change. The sample directory should always be changed for a new training run or else it overwrites the previous ones.

Lambda identity is 1 if the network doesn't change the orignal image at all. The smaller it gets the more influence the lambda identity should have on the results. 400,000 iterations seems to be a good starting place, although 200,000 can be run with results seen. After 400,000 results don't seem to change at least not from my sampled runs. 

## Testing Command Prompt
python main.py --mode test --dataset RaFD --image_size 128\
               --c_dim 7 --rafd_image_dir data/facialExp256/train \
               --**sample_dir** stargan_identity.00001/samples --log_dir stargan_identity.00001/logs \
               --model_save_dir stargan_identity.00001/models --result_dir stargan_identity.00001/results
