import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.swapping_autoencoder_model import SwappingAutoencoderModel
from options.Options import Options  # Ensure this imports your options correctly
from prepare_dataset_object import DatasetObj
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def load_model(model, path, steps):
    """Load the trained model components from saved states."""
    g_path = f"{path}G_steps_{steps}.pth"
    e_path = f"{path}E_steps_{steps}.pth"
    d_path = f"{path}D_steps_{steps}.pth"
    model.G.load_state_dict(torch.load(g_path, map_location=model.opt.device))
    model.D.load_state_dict(torch.load(d_path, map_location=model.opt.device))
    model.E.load_state_dict(torch.load(e_path, map_location=model.opt.device))
    model.eval()
    return model



def test_discriminator_with_generated(model, test_loader, num_images=10):
    """
    Test the discriminator on both real and generated images, and plot the results.
    """
    model.D.eval()  # Ensure the discriminator is in evaluation mode
    with torch.no_grad():
        real_images, generated_images, real_outputs, fake_outputs = [], [], [], []

        for i, data in enumerate(test_loader):
            images = data[0].to(model.opt.device)

            # Get discriminator outputs for real images
            real_out = model.D(images)
            real_images.append(images)
            real_outputs.append(real_out)

            # Generate fake images using the generator
            sp, gl = model.encode(images)
            fake_imgs = model.decode(sp, gl)
            fake_out = model.D(fake_imgs)
            generated_images.append(fake_imgs)
            fake_outputs.append(fake_out)

            if len(real_images) * images.size(0) >= num_images:
                break

        # Concatenate results
        real_images = torch.cat(real_images, dim=0)[:num_images]
        generated_images = torch.cat(generated_images, dim=0)[:num_images]
        real_outputs = torch.cat(real_outputs, dim=0)[:num_images]
        fake_outputs = torch.cat(fake_outputs, dim=0)[:num_images]

        # Plot the images and discriminator outputs
        fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))

        for idx in range(num_images):
            # Real image
            axes[idx, 0].imshow(real_images[idx].permute(1, 2, 0).cpu().numpy())
            axes[idx, 0].set_title("Real Image", fontsize=14)
            axes[idx, 0].axis("off")

            # Generated image
            axes[idx, 1].imshow(generated_images[idx].permute(1, 2, 0).cpu().numpy())
            axes[idx, 1].set_title("Generated Image", fontsize=14)
            axes[idx, 1].axis("off")

            # Discriminator outputs
            axes[idx, 2].text(
                0.5,
                0.5,
                f"Real: {real_outputs[idx].item():.4f}\nFake: {fake_outputs[idx].item():.4f}",
                fontsize=14,
                ha="center",
                va="center",
            )
            axes[idx, 2].set_title("Discriminator Output", fontsize=14)
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.show()





def test_model(model, test_loader):
    """Generate output using the model and return it."""
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(model.opt.device)
            batch_size = images.size(0)

            # Encode the images to get structure and style codes
            sp, gl = model.encode(images)

            # Create a random permutation for style codes
            perm = torch.randperm(batch_size)
            gl_style = gl[perm]
            style_images = images[perm]

            # Generate images using structure from original and permuted style codes
            generated_images = model.decode(sp, gl_style)

            return images, style_images, generated_images


def show_images_grid(structure_images, style_images, generated_images):
    """Display structure images, style images, and generated images in a grid."""
    num_images = min(len(structure_images), len(style_images), len(generated_images))

    # Set up the figure with a grid of subplots
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    for i in range(num_images):
        # Structure image
        axes[i, 0].imshow(structure_images[i].permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 0].set_title('Structure Image', fontsize=14)
        axes[i, 0].axis('off')

        # Style image
        axes[i, 1].imshow(style_images[i].permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 1].set_title('Style Image', fontsize=14)
        axes[i, 1].axis('off')

        # Generated image
        axes[i, 2].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 2].set_title('Generated Image', fontsize=14)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Load options and model
options = Options()  # Make sure this loads your options correctly
model = SwappingAutoencoderModel(options)

# Paths to model components
steps = 28032
path = "train_models/Folder_SC8_GC512_Res3_DSsp3_DSgl1_Ups3_PS32/"

# Load model components
model = load_model(model, path, steps)

# Prepare test data
# Ensure that your Dataset class returns images in the correct format
test_data = DatasetObj.load('dataset-objects/human_faces_paths.pkl')

  # Limit the dataset to 1000 samples
subset_size\
= 1000
subset_indices = torch.arange(subset_size)
limited_data = Subset(test_data, subset_indices)

test_loader = DataLoader(limited_data, batch_size=4, shuffle=True)

# Test the model
structure_images, style_images, generated_images = test_model(model, test_loader)

# Show the images
show_images_grid(structure_images, style_images, generated_images)

# Test the discriminator
test_discriminator_with_generated(model, test_loader, num_images=5)
