import os
import shutil
from random import shuffle

data_dir = 'data'  # Path to the original data directory
split_dir = 'split_data'  # Path to the new split directory
train_dir = os.path.join(split_dir, 'train')
val_dir = os.path.join(split_dir, 'validation')
test_dir = os.path.join(split_dir, 'test')

classes = ['non_globally_sclerotic_glomeruli', 'globally_sclerotic_glomeruli']
class_labels = {'non_globally_sclerotic_glomeruli': 0, 'globally_sclerotic_glomeruli': 1}

# Create the split directory and subdirectories
os.makedirs(split_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
   class_dir = os.path.join(data_dir, class_name)
   images = os.listdir(class_dir)
   shuffle(images)  # Randomize the order of images

   num_images = len(images)
   train_end = int(0.7 * num_images)
   val_end = int(0.9 * num_images)

   for i, image in enumerate(images):
       image_path = os.path.join(class_dir, image)
       target_dir = train_dir if i < train_end else val_dir if i < val_end else test_dir
       target_class_dir = os.path.join(target_dir, f'class_{class_labels[class_name]}')
       os.makedirs(target_class_dir, exist_ok=True)
       shutil.copy(image_path, target_class_dir)

print("Data split successfully!")
