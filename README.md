# Binary Classification of sclerotic and non-sclerotic glomeruli

This repository consists of research and code for training and evaluation of various Neural Network models for 'Binary Classification of sclerotic and non-sclerotic glomeruli' using microscopic images.
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

To run white padding on the already split dataset (obtained by running the script above), run the given script as below:

```
python scripts\create_white_padded_data.py
```
Note: Update the variables `split_data_dir` and `padded_data_dir` in the script to the path to the original data directory and new directory respectively.

### Normalization
I made use of [Normalization layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization) by Tensorflow Keras API to normalize the given data using the mean and standard deviation. 
> This layer will shift and scale inputs into a distribution centered around 0 with standard deviation 1. It accomplishes this by precomputing the mean and variance of the data, and calling (input - mean) / sqrt(var) at runtime.

For preprocessing images for larger pre-trained networks I made use of already exisiting `preprocess_input()` methods for [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input) and [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input)
>The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.

### Data Augmentation
After training the Simple CNN model with the above methods, the graphs comparing the training and validation accuracy and loss showed that the model performed poorly on validation dataset (validation loss increased and accuracy decreased with every epoch). This was a case of overfitting. To solve this, I decided to employ data augmention on the training dataset. I have currently only experimented with a random horizontal flip and a random rotation of 10% * 2pi which helped resolve the overfitting.

Images after preprocessing:
![image](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/98502e10-ab5f-44c2-939f-b553fed9933c)


## Models

### Logistic Regression
I decided to first train and test on a simple single layer neural network with a sigmoid activation, which is basically a Logisitc Regression model. With 20 epochs of training, the model achieved an accuracy of 87% on the test dataset. From the loss graph we can observe that the model did not require many epochs to converge.
![<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/2c56932d-a946-42f0-abe7-6268b96b18bc" width="250"/>](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/2c56932d-a946-42f0-abe7-6268b96b18bc =300x200)

![image](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/a2c77d35-bd1c-4369-a8f5-e6179d144228 =300x200)

### Simple CNN Classifier
As the performance of the logistic regression model was not upto the mark, I decided to use Convolutional layers in the neural network model. I chose a general image classifier model provided in the Tensorflow documentaion for [image classification](https://www.tensorflow.org/tutorials/images/classification) to begin with. The model consisted of 3 convolutional layers, each followed by a Max Pooling layer and 2 fully connected layers at the end.
![image](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/b51f2c88-8819-4c6b-bc31-dca9a4243124 =300x200)

![image](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/81f853b6-9c22-4cf9-9dd9-5789b6d9c940 =300x200)

I further evaluated the model on 577 image samples set aside for testing and also computed the below confusion matrix:
![image](https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/634d6bcc-43ac-40c9-ade7-e84fb0a9a627)

### VGG-16


### ResNet-50


## Performance Metrics




