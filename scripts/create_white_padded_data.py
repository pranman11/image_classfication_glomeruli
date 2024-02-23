import os
from PIL import Image

split_data_dir = 'split_data'
padded_data_dir = 'split_data_white_padded'

os.makedirs(padded_data_dir, exist_ok=True)  # Create the padded directory
count = 0
for root, directories, files in os.walk(split_data_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, file)
            padded_path = os.path.join(padded_data_dir + root.replace(split_data_dir, ''), file)  # Preserve path structure
            os.makedirs(os.path.dirname(padded_path), exist_ok=True)  # Ensure subdirectories exist
            count += 1
            print(f"{count} {padded_path}")
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    new_size = max(width, height)
                    padded_img = Image.new('RGB', (new_size, new_size), color='white')  # Create white background
                    padded_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))  # Center the original image
                    padded_img.save(padded_path)  # Save to the correct path
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

print("Image padding completed!")
