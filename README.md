# Binary Classification of sclerotic and non-sclerotic glomeruli

This repository consists of research around and code for training and evaluation of various Neural Network models for 'Binary Classification of sclerotic and non-sclerotic glomeruli' using microscopic images.
For simplicity 'class_0' is non-sclerotic and 'class_1' is sclerotic glomeruli.

Below is a summary of the trained models and their accuracy on a small test dataset.

|          Model        | Test Set Accuracy |
|-----------------------|-------------------|
|   Logistic Regression | 88%               |
|   Simple CNN model    | 96%               |
|   Frozen VGG16 model  | 98%               |

## Dataset

The dateset collected given from the Computational Microscopy Imaging Lab at the University of Florida is distributed as follows:

|          Class Name              | Class Label      | Number of Samples  |
|----------------------------------|------------------|--------------------|
| globally_sclerotic_glomeruli     |     class_1      |      1054          |
| non_globally_sclerotic_glomeruli |     class_0      |      4704          |

As we can notice that this data is imbalanced but this might or might not affect the training of our neural network(s). To compensate for the imbalance we can use data augmentation techniques to increase the number of samples in the 'globally_sclerotic_glomeruli' class which hasn't been explored further yet. 

For the purposes of training, validation and evaluation (testing) I have divided the data as follows:

|       Stage           | Percentage| Number of Samples (2 classes)  |
|-----------------------|-----------|--------------------------------|
|    Training           |     70%   |            4121                |
|    Validation         |     20%   |            1154                |
|    Testing            |     10%   |            577                 |

To split the given data into above percentage by creating separate directories I have written a script that can be run as below:

```
python scripts\create_data_split.py
```

Note: Update the variables `data_dir` and `split_dir` in the script to the path to the original data directory and new directory respectively.


## Data Preprocessing

### Resizing and White Padding
The given image samples are of various aspect ratios and need to be standardized for the training of our neural networks. Initially, I tested my logistic regression model and Simple CNN model with simply resizing the images to 256x256 dimensions and trained the models. This resulted in low test accuracy (less than 90% for the CNN model). After having a closer look at the image samples, I realized that resizing directly might result in loss of information that would be essential to classify images. Therefore, I chose to first create white padded images to create square shaped images and then resize the images to 256x256. This resulted in significant improvement of the model's accuracy on the test set.

### Normalization
I made use of [Normalization layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization) by Tensorflow Keras API to normalize the given data using the mean and standard deviation. 
> This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1. It accomplishes this by precomputing the mean and variance of the data, and calling (input - mean) / sqrt(var) at runtime.

For preprocessing images for larger pre-trained networks I made use of already exisiting `preprocess_input()` methods for [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input) and [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input)
>The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

### Data Augmentation
After training the Simple CNN model with the above methods, the graphs comparing the training and validation accuracy and loss showed that the model performed poorly on validation dataset (validation loss increased and accuracy decreased with every epoch). This was a case of overfitting. To solve this, I decided to employ data augmention on the training dataset. I have currently only experimented with a random horizontal flip and a random rotation of 10% * 2pi which helped resolve the overfitting.

## Models

### Logistic Regression




