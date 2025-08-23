# Datasets
Our image-to-image classification network used benign and malignant images from the ISIC 2020 dataset. We selected a subset of the images. The name of the images are stored in CNN_trainset and CNN_valset. Download the ISIC 2020 dataset and store the images. Combine the path where those images are stored with the image_directory names in the CSV files to read in the files. These can be stored if you are going to use them often, or can just be read in initially. We load in the saved images in our example, but uncomment the code if you want to load the images in directly. Also, if the images are in the correct directory, image_list is not needed as the directory can just be used. 

The full ISIC dataset can be used in the CNN using the train.csv file. Using the train_test split with a seed of 316 results and a subset of 2,000 benign images with a 316 seed, results in the same dataset in the two CNN csv files. 

The histopathology and single image expert consensus images are subsetted from the ISIC 2020 dataset. They were collected using the tags on the ISIC database. Once you read in the 2020 dataset, you can then use the csv's to subset the images to only meet these diagnosis type. To preven overlap in the testing set, we recommend masking the images which can be found in the dataset preperation file. 

For the apples to oranges dataset, the images can be downloaded from Berkeley [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/). These images are in the correct folders and can be read in from the directory.
