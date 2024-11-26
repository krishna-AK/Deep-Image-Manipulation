import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.swapping_autoencoder_model import SwappingAutoencoderModel
from options import TrainOptions  # Update this as per your options implementation
from HumanFaces import HumanFaces
from AnimeDataset import AnimeDataset
from torch.utils.data import DataLoader

from options.Options import Options



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

def test_model(model, test_loader):
    """ Generate output using the model and return it. """
    with torch.no_grad():
        for data in test_loader:
            # images = data.to(model.opt.device)
            images = torch.cat(data, dim=0).to(model.opt.device)
            sp, gl = model.encode(images)
            print(gl)
            generated_images = model.decode(sp, gl)
            mix_generated_images = model.decode(sp,model.swap(gl))
            return images, generated_images, mix_generated_images


import torch
torch.device('cuda')

def generate_samples(model, test_loader, no_of_samples=10, noise_scale_factor=1):
    with torch.no_grad():
        for data in test_loader:
            # images = data.to(model.opt.device)
            images = torch.cat(data, dim=0).to(model.opt.device)
            sp, gl = model.encode(images)

            # Calculate the standard deviation of gl
            std_gl = torch.std(gl)
            std_sp = torch.std(sp)



            generated_images = []
            for _ in range(no_of_samples):
                noise = noise_scale_factor * std_gl * torch.randn_like(gl)
                modified_gl = gl + noise
                print(modified_gl)
                generated_image = model.decode(sp,modified_gl)
                generated_images.append(generated_image)

            # Convert list of images to a tensor and squeeze the extra dimension
            generated_images_tensor = torch.stack(generated_images).squeeze(1)

            return images, generated_images_tensor

    # If no_of_samples is larger than the number of images in a single batch,
    # additional logic is needed to accumulate the correct number of samples.



def show_images(real_images, generated_images, mix_generated_images = None):
    """ Display the real and generated images for comparison. """
    real_images = make_grid(real_images, nrow=4)
    generated_images = make_grid(generated_images, nrow=4)
    if mix_generated_images is not None:
        mix_generated_images = make_grid(mix_generated_images,nrow = 4)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Real Images")
    plt.imshow(real_images.permute(1, 2, 0).cpu())
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Generated Images")
    plt.imshow(generated_images.permute(1, 2, 0).cpu())
    plt.axis('off')

    if mix_generated_images is not None:
        plt.subplot(1, 2, 1)
        plt.title("Mix Generated Images")
        plt.imshow(mix_generated_images.permute(1, 2, 0).cpu())
        plt.axis('off')

    plt.show()

def show_images_mix(real_images, mix_generated_images):
    """ Display the real and generated images for comparison. """
    real_images = make_grid(real_images, nrow=4)
    mix_generated_images = make_grid(mix_generated_images, nrow=4)

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title("Real Images")
    plt.imshow(real_images.permute(1, 2, 0).cpu())
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("MIX Generated Images")
    plt.imshow(mix_generated_images.permute(1, 2, 0).cpu())
    plt.axis('off')

    plt.show()

# Load options and model
options = Options()  # Update this line based on how you handle options
model = SwappingAutoencoderModel(options)

# Paths to model components
epoch = 62400
path = "train_models/Folder_SC8_GC2048_Res2_DSsp4_DSgl2_Ups4_PS32/"

# Load model components
model = load_model(model,path,epoch)

# Prepare test data
test_data = HumanFaces.load('datasets/human_faces_paths.pkl')  # Update with your dataset path
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

# # Test the model
real_images, generated_images, mix_generated_images = test_model(model, test_loader)

# real_images,generated_images = generate_samples(model, test_loader)

# Show the images
show_images(real_images, generated_images, mix_generated_images)
# show_images_mix(real_images, mix_generated_images)

# show_images(real_images,generated_images)
