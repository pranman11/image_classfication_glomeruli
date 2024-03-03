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
The given image samples are of various aspect ratios and need to be standardized for the training of our neural networks. Initially, I tested my logistic regression model and Simple CNN model with simply resizing the images to 256x256 dimensions and trained the models. This resulted in low test accuracy (less than 90% for the CNN model). After having a closer look at the image samples, I realized that resizing directly might result in loss of information that would be essential to classify images. Therefore, I chose to first create white padded images to create square shaped images and then resize the images to 256x256 (for large pretrained models I user 224x224). This resulted in significant improvement of the model's accuracy on the test set.

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

#### Logistic Regression
I decided to first train and test on a simple single layer neural network with a sigmoid activation, which is basically a Logisitc Regression model. With 20 epochs of training, the model achieved an accuracy of 87% on the test dataset. From the loss graph we can observe that the model did not require many epochs to converge.

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/2c56932d-a946-42f0-abe7-6268b96b18bc" width="400" height="300"/>
<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/a2c77d35-bd1c-4369-a8f5-e6179d144228" width="400" height="300"/>

#### Simple CNN Classifier
The trained model can be downloaded from [here](https://drive.google.com/file/d/1iG-HhaZL1bl92Vjlrn0ozCs-sruuggOg/view?usp=drive_link) (42.4 MB)

As the performance of the logistic regression model was not upto the mark, I decided to use Convolutional layers in the neural network model. I chose a general image classifier model provided in the Tensorflow documentaion for [image classification](https://www.tensorflow.org/tutorials/images/classification) to begin with. The model consisted of 3 convolutional layers, each followed by a Max Pooling layer and 2 fully connected layers at the end. After training for 25 epochs, the model converged as follows:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/b51f2c88-8819-4c6b-bc31-dca9a4243124" width="400" height="300"/>
<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/81f853b6-9c22-4cf9-9dd9-5789b6d9c940" width="400" height="300"/>

I further evaluated the model on 577 image samples set aside for testing. The model achieved an accuracy of 97%, for which you can see the below confusion matrix:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/634d6bcc-43ac-40c9-ade7-e84fb0a9a627" width="500" height="400"/>

##### Future work:
As we have achieved a fairly decent accuracy above, tweaking the model by adding layers, using higher resolution images (I have used 256x256), trying out different preprocessing methods, or changing hyperparameters might help us improve the accuracy. However, higher accuracy could also be possibly attained by using already existing large pre-trained models like VGG16 or ResNet, which is what I decided to explore further.

### Large pretrained models
I first decided to train the VGG-16 model by normalizing the training data using the Normalization layer above, but as expected that did not not perform well (not documented here) with the VGG-16 model as it's weights have been computed using ImageNet data. Therefore, I use the `preprocess_input` methods mentioned above to preprocess images.

#### VGG-16
The trained model can be downloaded from [here](https://drive.google.com/file/d/1-2Q9IMdTt2wPdm3fdZPCCqUqkD6ubQtA/view?usp=drive_link) (1.25 GB)

I removed the top layer of the VGG-16 model and falttened the output of the last convolutional layer. Further I added 3 fully connected layers as shown [here](https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4). I have also added a Dropout layer to minimize overfitting. After training for 25 epochs, the model performed as below:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/78a9ec6d-9029-432f-a8e8-3fde03328be3" width="400" height="300"/>
<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/b34e81fc-6478-44b2-b084-9ee892830f7d" width="400" height="300"/>

I further evaluated the model on 577 image samples set aside for testing. The model achieved an accuracy of 98%, for which you can see the below confusion matrix:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/b3a32844-4bf9-468a-b1a3-f725d2c6909b" width="500" height="400"/>

#### ResNet-50
I used the same approach as used for VGG-16 to train this model. After training for 25 epochs, the model performed as below:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/bee3c7ee-6652-4a98-b007-02c9a9ebc916" width="400" height="300"/>
<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/44e43eca-39a8-4ee3-8317-1730e60917a9" width="400" height="300"/>

I further evaluated the model on 577 image samples set aside for testing. The model achieved an accuracy of 98%, for which you can see the below confusion matrix:

<img src="https://github.com/pranman11/image_classfication_glomeruli/assets/17320182/834dc53f-2fe9-415c-91bc-7d61ca1fc44c" width="500" height="400"/>

## Performance Metrics
I have chosen the Binary Cross Entropy loss function to train the model that uses a single probability value (0 to 1) for each image. I make use of the graph of loss vs. epoch and accuracy vs. epoch to observe whether there model overfits.

Since the last layer uses a sigmoid activation, I use threshold value of 0.5 to get the predicted values and compute the confusion matrix to visualize how the model performs on the dataset set aside for testing.

## Evaluation file

Before evalution the data needs to be white padded which can be done using the script mentioned above. You will have to edit the code to update the `split_data_dir` as the path of the data directory that contains the test data (with subfolders of the two classes).  The white padded data folder will be created with the name `split_data_white_padded` (provided it is not changed. The script is run as follows:

```
python scripts\create_white_padded_data.py
```

The evaluation.py takes a model - downloaded from [VGG-16](https://github.com/pranman11/image_classfication_glomeruli?tab=readme-ov-file#vgg-16) or [SimpleCNN](https://github.com/pranman11/image_classfication_glomeruli?tab=readme-ov-file#simple-cnn-classifier) and test data path as input and create the evaluation.csv file (alongwith the accuracy and confusion matrix). The <test_data_path> value will have the file created after the running the `create_white_padded_data.py` script. The evaluation.py file can be run as follows:

```
python evaluation.py <test_data_path> <model_path>
```

For example:
```
python evaluation.py split_white_padded_data\test vgg16_glomeruli_classifier
```

The evaluation.csv file created contains the file name and the corresponding predicted class as columns.

Note: Please do not change the name of the model file name as it uses the model name to preprocess images accordingly.

## Training the models:

The models can be trained using Google Colab or using Jupyter Lab on your local machine. In case of Google Colab, the `data` folder containing image samples needs to be uploaded to Google Drive (of the same user as Colab), in the same directory where the notebook is located. In case you're using Jupyer Lab, the same file structure needs to be maintained and the the requirements need to be installed as follows:

```
pip install -r requirements.txt
```

You may create a virtual environment and use that while running the notebook. This is shown [here](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/).

After installing the requirements, the data needs to be split into train, validation and test splits using `create_data_split.py` as mentioned above. Further, the data needs to be white padded as well using `create_white_padded_data.py`.

The DataLoader class has been written in such a way that it checks whether the code is being run on Google Colab or locally. The DataLoader mounts the root directory of your Google Drive. It takes the data directory path as input (which in the below case is the present in the root directory itself) and is used as follows:

```
data_loader = DataLoader('split_data_white_padded')

train_data_path = '/train'
test_data_path = '/test'
validation_data_path = '/validation'

IMAGE_SIZE = 224
BATCH_SIZE = 32
SHUFFLE = True

train_dataset = data_loader.load_image_data(train_data_path, IMAGE_SIZE, BATCH_SIZE, SHUFFLE)
validation_dataset = data_loader.load_image_data(validation_data_path, IMAGE_SIZE, BATCH_SIZE, not SHUFFLE)
test_dataset = data_loader.load_image_data(test_data_path, IMAGE_SIZE, BATCH_SIZE, not SHUFFLE)
```

All cells can be run after this without modification.
