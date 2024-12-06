import os
import shutil
import zipfile
import requests
from prepare_dataset_object import prepare_dataset_obj

def download_dataset(url, download_path):
    """
    Download a dataset from the given URL and save it to the specified path.
    """
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open(download_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"Dataset downloaded to {download_path}.")

def extract_zip(zip_path, extract_to):
    """
    Extract the contents of a zip file to the specified directory.
    """
    print(f"Extracting zip file: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete. Files are in {extract_to}.")

def gather_images(src_dir, dest_dir):
    """
    Recursively gather all images from the source directory into a single destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print("Gathering all images into one folder...")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)

                # Rename file if there's a name conflict
                base_name, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_file_path):
                    dest_file_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                    counter += 1

                shutil.move(src_file_path, dest_file_path)
    print(f"All images moved to {dest_dir}.")

def cleanup(zip_path, extract_dir):
    """
    Delete the original zip file and the extracted directory.
    """
    print("Cleaning up...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted zip file: {zip_path}")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
        print(f"Deleted extracted directory: {extract_dir}")

if __name__ == "__main__":
    # Define dataset URL and paths
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/greatgamedota/ffhq-face-data-set"
    project_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    datasets_dir = os.path.join(project_dir, "datasets")
    download_path = os.path.join(datasets_dir, "ffhq-face-data-set.zip")
    extract_to = os.path.join(datasets_dir, "ffhq-face-data-set")
    car_images_dir = os.path.join(datasets_dir, "all_extracted_faces")

    # Ensure the datasets folder exists
    os.makedirs(datasets_dir, exist_ok=True)

    # try:
    #     # Download the dataset
    #     download_dataset(dataset_url, download_path)

    #     # Extract the dataset
    #     extract_zip(download_path, extract_to)

    #     # Gather all images into a single folder
    #     gather_images(extract_to, car_images_dir)

    #     # Cleanup: remove the zip file and extracted directory
    #     cleanup(download_path, extract_to)

    #     print(f"All tasks completed successfully. face images are in {car_images_dir}.")

    # except Exception as e:
    #     print(f"An error occurred: {e}")

    prepare_dataset_obj(source_dataset_folder=car_images_dir, dataset_name="ffhq-face-data-set")