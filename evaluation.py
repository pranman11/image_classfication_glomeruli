import argparse
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from matplotlib import pyplot as plt


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained ML model.')
parser.add_argument('test_data_path', type=str, help='Path to the test dataset.')
parser.add_argument('model_path', type=str, help='Path to the trained model file.')
args = parser.parse_args()

if 'resnet' in args.model_path:
    IMAGE_SIZE = 224
elif 'vgg16' in args.model_path:
    IMAGE_SIZE = 224
else:
    IMAGE_SIZE = 256

test_dataset = tf.keras.utils.image_dataset_from_directory(
    args.test_data_path,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # Adjust image size as needed
    batch_size=32,
    shuffle=False,
)

# Preprocess the images with white padding and normalization
def preprocess_images(dataset):
    AUTOTUNE = tf.data.AUTOTUNE

    if 'resnet' in args.model_path:
        print('preprocessing for vgg16')
        dataset = dataset.map(lambda x, y: (tf.image.pad_to_bounding_box(x, 0, 0, IMAGE_SIZE, IMAGE_SIZE), y))
        dataset = dataset.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
            num_parallel_calls=AUTOTUNE)
    elif 'vgg16' in args.model_path:
        print('preprocessing for vgg16')
        dataset = dataset.map(lambda x, y: (tf.image.pad_to_bounding_box(x, 0, 0, IMAGE_SIZE, IMAGE_SIZE), y))
        dataset = dataset.map(lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y),
            num_parallel_calls=AUTOTUNE)
    else:
        # Create a normalization layer
        # mean and variance computed on training data provided
        normalization_layer = layers.Normalization(mean=181.12433, variance=3406.8462)
        dataset = dataset.map(lambda x, y: (tf.image.pad_to_bounding_box(x, 0, 0, IMAGE_SIZE, IMAGE_SIZE), y))
        
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=AUTOTUNE)
    return dataset

original_file_paths = test_dataset.file_paths
original_class_names = test_dataset.class_names
test_dataset = preprocess_images(test_dataset)

# Load the trained model
model = tf.keras.models.load_model(args.model_path)

loss, accuracy = model.evaluate(test_dataset)
print("Test accuracy:", accuracy)

# Get predictions and ground truth labels
y_pred = model.predict(test_dataset)
y_pred_classes = tf.where(y_pred > 0.5, 1, 0)
y_pred_list = [y[0] for y in y_pred_classes.numpy()]
y_true = tf.concat([y for _, y in test_dataset], axis=0)

# Calculate and plot confusion matrix
cm = tf.math.confusion_matrix(y_true, y_pred_classes)

# Annotate each cell with the corresponding value
plt.imshow(cm, cmap=plt.cm.Reds)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(original_class_names)), original_class_names, rotation=45)
plt.yticks(range(len(original_class_names)), original_class_names)
plt.colorbar()
for i in range(len(cm)):
    for j in range(len(cm)):
        value = cm[i, j].numpy()
        plt.text(i, j, value, ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.show()


# Create the evaluation CSV file
evaluation_data = {
    'name': original_file_paths,
    'predicted class': y_pred_list
}

df = pd.DataFrame(evaluation_data)
df.to_csv('evaluation.csv', index=False)

print('Evaluation results saved to evaluation.csv')
