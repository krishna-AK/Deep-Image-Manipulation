import os
from PIL import Image

def resize_images_in_folder(folder_path, target_size):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)

            with Image.open(file_path) as img:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Overwrite the original image
                img.save(file_path)
                print(f"Resized {file_name}")

# Usage
folder = "D:\\datasets\\Flickr-faces-512"  # Replace with your folder path
target_size = (64, 64)  # Replace with your desired size (width, height)
resize_images_in_folder(folder, target_size)
