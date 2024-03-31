import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle

import torchvision.transforms as transforms


class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs=None, transform=None, image_paths=None):
        # image_dirs is a list of directories
        self.image_paths = []

        if image_dirs is not None:
            for dir in image_dirs:
                for root, _, files in os.walk(dir):
                    for img_file in files:
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other formats if needed
                            self.image_paths.append(os.path.join(root, img_file))

        elif image_paths is not None:
            self.image_paths = image_paths

        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    @staticmethod
    def load(file_path):

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])

        with open(file_path, 'rb') as f:
            image_paths = pickle.load(f)
        return AnimeDataset(image_paths=image_paths, transform=transform)

class ConditionalResize(transforms.Resize):
    def __call__(self, img):
        if img.size != self.size:
            img = super(ConditionalResize, self).__call__(img)
        return img


if __name__ == '__main__':


    transform = transforms.Compose([
        ConditionalResize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # image_dirs = ['D:\\datasets\\anime_faces\\data',"D:\\datasets\\human_faces\\thumbnails128x128"]
    image_dirs = ["D:\\datasets\\Flickr-faces-512"]
    # Assuming you've already created an instance of the dataset
    dataset = AnimeDataset(image_dirs=image_dirs, transform=transform)


    # Save the dataset information
    with open('datasets/flickr_faces_dataset_paths.pkl', 'wb') as f:
        pickle.dump(dataset.image_paths, f)


    # with open('valorant_dataset_paths.pkl', 'rb') as f:
    #     image_paths = pickle.load(f)

    # Now you can use image_paths to recreate the dataset or for other purposes



    # view transformed image

    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    # Assuming the ValorantDataset and the transform have been defined as before

    # # Create an instance of the dataset
    # dataset = ValorantDataset(image_dir='/screenshots/valorant', transform=transform)

    # Load an image (for example, the first image in the dataset)
    img = dataset[7]  # 'img' is now a transformed tensor

    # Function to convert a tensor to a PIL Image
    def tensor_to_pil(tensor):
        tensor = tensor.clone()  # Clone the tensor so we don't make changes to the original
        tensor = tensor.clamp(0, 1)  # Ensure values are in the range [0, 1]
        return transforms.ToPILImage()(tensor)


    # Convert the tensor to a PIL Image and display it
    pil_img = tensor_to_pil(img)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


