import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.swapping_autoencoder_model import SwappingAutoencoderModel
from options.Options import Options  # Ensure this imports your options correctly
from prepare_dataset_object import DatasetObj
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import argparse
import random

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
        fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))

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
            # axes[idx, 2].text(
            #     0.5,
            #     0.5,
            #     f"Real: {real_outputs[idx].item():.4f}\nFake: {fake_outputs[idx].item():.4f}",
            #     fontsize=14,
            #     ha="center",
            #     va="center",
            # )
            # axes[idx, 2].set_title("Discriminator Output", fontsize=14)
            # axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.show()




def generate_and_display_combinations(model, dataset, num_structure=4, num_style=4):
    """
    Generate images for combinations of selected structure and style codes, and display them in a grid.
    
    Args:
        model: The trained swapping autoencoder model.
        dataset: The dataset to sample images from.
        num_structure: Number of images to select for structure codes (rows).
        num_style: Number of images to select for style codes (columns).
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Select random indices for structure and style codes
    structure_indices = random.sample(range(len(dataset)), num_structure)
    style_indices = random.sample(range(len(dataset)), num_style)
    
    # Load the structure and style images
    structure_images = torch.stack([dataset[i][0] for i in structure_indices]).to(model.opt.device)
    style_images = torch.stack([dataset[i][0] for i in style_indices]).to(model.opt.device)
    
    # Encode the images to get structure and style codes
    sp_structure, _ = model.encode(structure_images)  # Only structure codes needed
    _, gl_style = model.encode(style_images)  # Only style codes needed
    
    # Generate images for all combinations
    generated_images = []
    for sp in sp_structure:
        row_images = []
        for gl in gl_style:
            generated_image = model.decode(sp.unsqueeze(0), gl.unsqueeze(0))
            row_images.append(generated_image.squeeze(0))  # Remove batch dimension
        generated_images.append(torch.stack(row_images))
    
    generated_images = torch.stack(generated_images)  # Shape: (num_structure, num_style, C, H, W)
    
    # Convert tensors to numpy for visualization
    structure_images_np = structure_images.permute(0, 2, 3, 1).cpu().numpy()
    style_images_np = style_images.permute(0, 2, 3, 1).cpu().numpy()
    generated_images_np = generated_images.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    
    # Plot the grid
    fig, axes = plt.subplots(num_structure + 1, num_style + 1, figsize=(15, 15))
    
    # Fill in the structure and style images along the grid borders
    for i in range(num_structure):
        axes[i + 1, 0].imshow(structure_images_np[i])
        axes[i + 1, 0].axis('off')
        if i == 0:
            axes[i + 1, 0].set_title('Structure', fontsize=12)
    
    for j in range(num_style):
        axes[0, j + 1].imshow(style_images_np[j])
        axes[0, j + 1].axis('off')
        if j == 0:
            axes[0, j + 1].set_title('Style', fontsize=12)
    
    # Fill in the generated images in the grid
    for i in range(num_structure):
        for j in range(num_style):
            axes[i + 1, j + 1].imshow(generated_images_np[i, j])
            axes[i + 1, j + 1].axis('off')
    
    # Hide the top-left corner of the grid
    axes[0, 0].axis('off')
    
    # Add horizontal and vertical lines
    for ax in axes[1:, 0]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(2)

    for ax in axes[0, 1:]:
        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Adjust spacing
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
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

            # Shift the style codes by 1
            gl_style = torch.roll(gl, shifts=1, dims=0)  # Circularly shift the style codes by 1
            style_images = torch.roll(images, shifts=1, dims=0)  # Corresponding style images

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



def main(args):
    # Load options and model
    options = Options()  # Ensure this loads your options correctly
    model = SwappingAutoencoderModel(options)

    # Paths to model components
    path = "train_models/Folder_SC8_GC2048_Res3_DSsp2_DSgl1_Ups2_PS32-ffhq-128/"

    # Load model components
    model = load_model(model, path, args.steps)

    # Prepare test data
    test_data = DatasetObj.load('dataset-objects/ffhq-face-data-set.pkl')

    # Limit the dataset to 1000 samples
    subset_size = 1000
    subset_indices = torch.arange(subset_size)
    limited_data = Subset(test_data, subset_indices)

    test_loader = DataLoader(limited_data, batch_size=4, shuffle=True)

    # Test the model
    structure_images, style_images, generated_images = test_model(model, test_loader)

    # Show the images
    # show_images_grid(structure_images, style_images, generated_images)

    # generate_and_display_combinations(model, test_data, num_structure=3, num_style=3)


    # Test the discriminator
    test_discriminator_with_generated(model, test_loader, num_images=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Swapping Autoencoder Model")

    # Add arguments for the script
    parser.add_argument("--steps", type=int, required=True, help="Number of steps for the model checkpoint to load")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
