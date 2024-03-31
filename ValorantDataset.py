import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle

import torchvision.transforms as transforms


class ValorantDataset(Dataset):
    def __init__(self, image_dir=None, transform=None, image_paths=None):
        self.transform = transform

        if image_paths is not None:
            # Initialize dataset from a list of image paths
            self.image_paths = image_paths
        elif image_dir is not None:
            # Initialize dataset from a directory
            self.image_dir = image_dir
            self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        else:
            raise ValueError("Either image_dir or image_paths must be provided")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image

    @staticmethod
    def load(file_path):

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])

        with open(file_path, 'rb') as f:
            image_paths = pickle.load(f)
        return ValorantDataset(image_paths=image_paths, transform=transform)


if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Assuming you've already created an instance of the dataset
    dataset = ValorantDataset(image_dir="D:\\PycharmProjects\\CAE\\screenshots\\Valorant", transform=transform)


    # Save the dataset information
    with open('datasets/valorant_dataset_paths.pkl', 'wb') as f:
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
    img, _ = dataset[121]  # 'img' is now a transformed tensor

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


