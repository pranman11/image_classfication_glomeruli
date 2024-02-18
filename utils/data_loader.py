from tensorflow.keras.utils import image_dataset_from_directory

try:
  from google.colab import drive
  IN_COLAB = True
except:
  IN_COLAB = False
    
class DataLoader:
    def __init__(self, root_path):
        if IN_COLAB:
        	# Mount Google Drive
            drive.mount('/content/drive')
            self.root = '/content/drive/MyDrive/' + root_path
        else:
            self.root = root_path
        
    def load_image_data(self, path, image_size, batch_size, shuffle):
        data_directory = self.root + path
        
        dataset = image_dataset_from_directory(
            data_directory,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=shuffle
		)
        
        return dataset