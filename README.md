## Image Class Translation Distance: A Novel Interpretable Feature for Image Classification
#### Using StarGAN - Official PyTorch Implementation

> **Image Class Translation Distance: A Novel Interpretable Feature for Image Classification**<br>
> Mikyla K. Bowen<sup>1</sup>, Jesse W. Wilson <sup>2,3</sup><br/>
> <sup>1</sup>College of Natural Sciences, Colorado State University, Colorado, United States of America, <sup>2</sup>Department of Electrical and Computer Engineering, Colorado State University, Colorado, United States of America<br>
> <sup>3</sup> School of Biomedical Engineering, Colorado State University, Colorado, United States of America<br/>
> https://arxiv.org/pdf/2408.08973 <br>
>
> **Abstract:** *We propose a novel application of image translation networks for image classification and demonstrate its
potential as a more interpretable alternative to conventional black box classification networks. We train a
network to translate images between possible classes, and then quantify translation distance, i.e. the degree
of alteration needed to conform an image to one class or another. These translation distances can then be
examined for clusters and trends, and can be fed directly to a simple classifier (e.g. a support vector machine,
SVM), providing comparable accuracy compared to a conventional end-to-end convolutional neural network
classifier. In addition, visual inspection of translated images can reveal class-specific characteristics and
biases in the training sets, such as visual artifacts that are more frequently observed in one class or another.
We demonstrate the approach on a toy 2-class scenario, apples versus oranges, and then apply it to two
medical imaging tasks: detecting melanoma from photographs of pigmented lesions and classifying 6 cell
types in a bone marrow biopsy smear. This novel application of image-to-image networks shows the potential
of the technology to go beyond imagining different stylistic changes and to provide greater insight into
image classification and medical imaging datasets.*


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


## Downloading datasets
[Bone Marrow Cytology Dataset](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
Dataset was can be downloaded from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification/data)

Data was split into equal sets of 5,000 using a seedfile.
```bash
find 'path'+ /BLA -type f -name "*.jpg" |
shuf --random-source=seedfile -n 5000 |
xargs -I {}
mv {} 'path'+ training/BLA
```

## Running the train, test, classification pipeline
The script run_exp.sh is a bash scrip that runs training, testing, and classification using translation distance. The script can be modified to train and test different parameters. 
Depending on machine requirements (might need to be adjsusted 
The resulting test and train translation distance csv files are used by the SVM in the svmTest script. However, the translation distances can be fed into other networks or visualized. See distClassifiers.ipynb. 


## Changes to the original PyTorch StarGAN script
#### main.py
- Add_argument: --lambda_id: weight for identity penalty.
- Add_argument: --dist_file_name: Store the translation distances in this file.
- Add_argument: --subset_dir: Default to using the entire dataset, or feed in a csv file with the listed subset samples

#### solver.py
- Add lambda_id, dist_file_name, and subset_dir
- Add identity loss:
``` bash
x_ident = self.G(x_real, c_org)
g_loss_ident=torch.sum(torch.abs(x_ident-x_real))
```
- Calculate Translation distance

### svmTest.py
- New script that uses the translation distances calculated by solver.py during testing and training.
- Normalize the translation distance, and then train and test and SVM classifier.

### EfficientNet_CNN_Cyto
Tensor Flow Implementation of an EfficientNet for prediction. It can be used for three classes or six classes, by updating the file path. 


### Attributes
- mode: train or test
- dataset: RaFD (if files are organized in a file directory structure)
- image_size: 128
- c_dim: number of classes (3 or 6 were used)
- rafd_image_dir: directory where the files are
- sample_dir, log_dir, model_save_dir, result_dir: directory to save, 'experiments' with a number were used in the script
- num_iters: Models were run with a minimum of 300,000 iterations. Additional iterations returned better image quality.
- lambda_id: default is 10, however after comparing results .0001 was the best value.
- batch_size: can be increased depending on the resources of the machine running it. 

```bash
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 6 --rafd_image_dir 'directory' \
               --sample_dir experiments/CYTO_${NUMBER}/samples \
               --log_dir experiments/CYTO_${NUMBER}/logs \
               --model_save_dir experiments/CYTO_${NUMBER}/models \
               --result_dir experiments/CYTO_${NUMBER}/results \
               --num_iters 500000 --lambda_id .0001 --batch_size 1 \
```

## Data Results
Translation Distance data in csv format can be found in the data folder. Additionally, the seed file for the datasets is also located in the data file.  


### Branches
	• Main branch: Currently overall functioning StarGAN code, 
		○ Is using the .001 lambda identity and loss calculations
		○ Saves pictures during training and after running testing
		○ Can use a subset of the data
		○ Also still using a slower version of the distance calculation
	• Manuscript branch: Similar to the main branch
		○ Key difference: saves images individually and can be used to generate graphics
		○ Use with the Manuscript_StarGAN.ipynb
		○ Also kept in a separate folder, but requires the full model saves
		○ Updated to a faster version of the translation distance calculation
	• Cluster:
		○ Cluster branch: adds additional finals to be used on a cluster
		○ Stargan_script.sh: run the training and testing of the network and pipe it to output files
		○ Still using the slower translation distance
		○ Computer has the full dataset of cyto: 5,000 images selected and 80/20 split
		○ Also the translation distance loss is also available for use
		○ Added weighted random sampling 


### Original PyTorch StarGAN Requirements 
This respository is a branch created from the official PyTorch implementation of the paper: 

> **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation**<br>
> [Yunjey Choi](https://github.com/yunjey)<sup>1,2</sup>, [Minje Choi](https://github.com/mjc92)<sup>1,2</sup>, [Munyoung Kim](https://www.facebook.com/munyoung.kim.1291)<sup>2,3</sup>, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup>2</sup>, [Sung Kim](https://www.cse.ust.hk/~hunkim/)<sup>2,4</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup>1,2</sup>    <br/>
> <sup>1</sup>Korea University, <sup>2</sup>Clova AI Research, NAVER Corp. <br>
> <sup>3</sup>The College of New Jersey, <sup>4</sup>Hong Kong University of Science and Technology <br/>
> https://arxiv.org/abs/1711.09020 <br>
>
> **Abstract:** *Recent studies have shown remarkable success in image-to-image translation for two domains. However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains. To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model. Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network. This leads to StarGAN's superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks.*


## Citation
Research based on the original StarGAN [paper](https://arxiv.org/abs/1711.09020):
```
@inproceedings{choi2018stargan,
author={Yunjey Choi and Minje Choi and Munyoung Kim and Jung-Woo Ha and Sunghun Kim and Jaegul Choo},
title={StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year={2018}
}
```

## Acknowledgements
We thank Priya Wolff (University of Colorado School of Medicine) and Yevgeniy Seminov (Mass. General Brigham) for helpful
conversations and suggestions, and Bill Carpenter (Colorado State University) for technical support with computational resources.
Experiments were carried out on the Riviera cluster (Data Science Research Institute, Colorado State University). This work was
funded by the Boettcher Foundation through a Boettcher Collaboration Grant and a Boettcher Educational Enrichment Grant.
