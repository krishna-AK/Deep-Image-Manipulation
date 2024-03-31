import os

import torch
from PIL import Image
from torchvision.transforms import transforms

from models.swapping_autoencoder_model import SwappingAutoencoderModel
from options.Options import Options

import matplotlib.pyplot as plt


def load_model(model, path,epoch):
    """ Load the trained model components from saved states. """
    g_path = path + 'G_epoch_'+str(epoch)+'.pth'
    e_path = path + 'E_epoch_' + str(epoch) + '.pth'
    d_path = path + 'D_epoch_' + str(epoch) + '.pth'
    model.G.load_state_dict(torch.load(g_path, map_location=model.opt.device))
    model.D.load_state_dict(torch.load(d_path, map_location=model.opt.device))
    model.E.load_state_dict(torch.load(e_path, map_location=model.opt.device))
    model.eval()
    return model

def lerp(a, b, r):
    if type(a) == list or type(a) == tuple:
        return [lerp(aa, bb, r) for aa, bb in zip(a, b)]
    return a * (1 - r) + b * r


def load_image(path):
    # Identify the image using the file name
    filename = os.path.basename(path)
    print(f"Loading image: {filename}")

    # Convert the image to PNG format
    png_path = os.path.splitext(path)[0] + '.png'
    if not os.path.exists(png_path) or os.path.getmtime(path) > os.path.getmtime(png_path):
        img = Image.open(path).convert('RGB')
        img.save(png_path)
        print(f"Converted to PNG and saved as: {png_path}")
    else:
        print(f"PNG image already exists: {png_path}")

    # Load the PNG image
    img = Image.open(png_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # Apply other necessary transformations like normalization
    ])

    tensor = transform(img).unsqueeze(0)  # Add a batch dimension
    return tensor


def evaluate(model):
    structure_image = load_image(input_structure_image).to('cuda')
    texture_image = load_image(input_texture_image).to('cuda')
    os.makedirs(output_dir, exist_ok=True)

    structure_code, source_texture_code = model.encode(structure_image)
    _, target_texture_code = model.encode(texture_image)

    alphas = [0, 0.25, 0.5, 0.75, 1]  # Include 0 and 1 to represent original images
    fig, axes = plt.subplots(1, len(alphas) + 2, figsize=(20, 5))  # Adjust the size as needed

    # Display original structure and texture images
    axes[0].imshow(transforms.ToPILImage()(structure_image[0].cpu()))
    axes[0].set_title("Original Structure")
    axes[0].axis('off')

    axes[1].imshow(transforms.ToPILImage()(texture_image[0].cpu()))
    axes[1].set_title("Target Texture")
    axes[1].axis('off')

    # Display interpolated images
    for i, alpha in enumerate(alphas, start=2):
        texture_code = lerp(source_texture_code, target_texture_code, alpha)
        output_image = model.decode(structure_code, texture_code)
        output_image = transforms.ToPILImage()((output_image[0].clamp(0, 1)))

        axes[i].imshow(output_image)
        axes[i].set_title(f"Alpha: {alpha}")
        axes[i].axis('off')

    plt.show()
    return {}



options = Options()  # Update this line based on how you handle options
sae_model = SwappingAutoencoderModel(options)

# Paths to model components
epoch = 639200
path = "train_models/Folder_SC8_GC2048_Res2_DSsp4_DSgl2_Ups4_PS32/"

# Load model components
sae_model = load_model(sae_model,path,epoch)


input_structure_image = 'interpolation_input/structure.png'
input_texture_image = 'interpolation_input/texture.png'
output_dir = 'interpolation_output'
texture_mix_alphas = [0,0.25,0.5,0.75,1]

evaluate(sae_model)