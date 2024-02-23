# Binary Classification of sclerotic and non-sclerotic glomeruli

This repository consists of research around and code for training and evaluation of various Neural Network models for 'Binary Classification of sclerotic and non-sclerotic glomeruli' using microscopic images.
For simplicity 'class_0' is non-sclerotic and 'class_1' is sclerotic glomeruli.

Below is a summary of the trained models and their accuracy on a small test dataset.

|          Model        | Test Set Accuracy |
|-----------------------|-------------------|
|   Logistic Regression | 88%               |
|   Simple CNN model    | 96%               |
|   Frozen VGG16 model  | 98%               |

### Dataset

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

'''
python create_data_split.py
'''

Note: Update the variables 'data_dir' and 'split_dir' in the script to the path to the original data directory and new directory respectively.



