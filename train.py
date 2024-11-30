import itertools
import sys
import os
import torch
from torch.utils.data import DataLoader
import gc
from datetime import datetime  # For timestamps

from prepare_dataset_object import DatasetObj
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
from options.Options import Options
from util import IterationCounter
from torch.utils.data import Subset


def createAndSetModelFolder(opt):
    # Create a folder name using the given options
    folder_name = (f"Folder_SC{opt.spatial_code_ch}_GC{opt.global_code_ch}_Res{opt.netG_num_base_resnet_layers}"
                   f"_DSsp{opt.netE_num_downsampling_sp}"
                   f"_DSgl{opt.netE_num_downsampling_gl}_Ups{opt.netG_no_of_upsamplings}_PS{opt.patch_size}"
                   f"-{opt.dataset_name}")

    # Use os.path.join for proper path handling
    model_dir = os.path.join(opt.save_training_models_dir, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model folder at: {model_dir}")
    else:
        print(f"Model folder already exists at: {model_dir}")

    # Set the model_path attribute in opt
    opt.model_dir = model_dir

    return model_dir


def logLosses(loss_log, dir):
    log_file_name = os.path.join(dir, 'training_log.txt')

    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        with open(log_file_name, 'a') as f:
            f.write('\n'.join(loss_log) + '\n')
            f.write('-' * 80 + '\n')
        print("Training log updated!")
    except IOError as e:
        print(f"Error writing to log file: {e}")


def get_gradient_flow(model, name="Model"):
    """
    Compute and log the gradient flow for a model.
    """
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return f"{name} Avg Gradient Norm: {avg_grad:.6f}"


class Prefetcher:
    """
    Prefetcher to preload data batches onto GPU.
    """
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_data = [item.to(self.device, non_blocking=True) for item in self.next_data]
        except StopIteration:
            self.next_data = None

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
        return self.next()


def train(opt, dataset_object_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Aborting training.")
    
    device = torch.device('cuda')

    createAndSetModelFolder(opt)

    # Load dataset
    data = DatasetObj.load(dataset_object_path)

    # Limit the dataset to 7000 samples
    subset_size = 7000
    subset_indices = torch.arange(subset_size)
    limited_data = Subset(data, subset_indices)

    # Optimized DataLoader setup
    dataset = DataLoader(
        limited_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,  # Adjust this based on your CPU cores
        pin_memory=True,
        prefetch_factor=4,  # Tune this based on available memory
        persistent_workers=True  # Workers remain active across epochs
    )
    opt.dataset = dataset
    iter_counter = IterationCounter(opt)

    # Initialize model and optimizer
    model = SwappingAutoencoderModel(opt).to(device)
    optimizer = SwappingAutoencoderOptimizer(model)

    loss_log = []
    total_epochs = opt.total_epochs

    try:
        while not iter_counter.completed_training():
            print(f"Starting epoch {iter_counter.epochs_completed + 1}/{total_epochs}")
            dataset_iterator = Prefetcher(dataset, device)

            for batch_idx, cur_data in enumerate(dataset_iterator):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with iter_counter.time_measurement("data"):
                    # Data is already on the GPU due to Prefetcher
                    pass

                with iter_counter.time_measurement("train"):
                    losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)

                    if iter_counter.steps_so_far%1000 == 0:
                        # Log losses to console
                        print(f"Iteration {iter_counter.steps_so_far}: ", end="")
                        for key in sorted(losses.keys()):
                            print(f"{key}: {losses[key]:.4f}", end=", ")
                        print()

                    # Compute gradient flow (if needed)
                    gen_grad_flow = get_gradient_flow(model.G, "Generator")
                    disc_grad_flow = get_gradient_flow(model.D, "Discriminator")
                    patch_disc_grad_flow = get_gradient_flow(model.Dpatch, "Patch Discriminator")

                    # Log all information
                    log_entry = (
                        f"Timestamp: {timestamp}, "
                        f"Epoch {iter_counter.epochs_completed + 1}, "
                        f"Iteration {iter_counter.steps_so_far}, "
                        f"Losses: {', '.join(f'{key}: {losses[key]:.4f}' for key in sorted(losses.keys()))}, "
                        f"{gen_grad_flow}, {disc_grad_flow}, {patch_disc_grad_flow}"
                    )
                    loss_log.append(log_entry)

                iter_counter.record_one_iteration()

            iter_counter.record_one_epoch()
            print(f"Epoch {iter_counter.epochs_completed} completed.")

            # Save the model after every epoch
            optimizer.save(iter_counter.steps_so_far)

            # End of an epoch, log everything
            logLosses(loss_log, opt.model_dir)
            loss_log.clear()

        print("Training completed.")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        print("Training finished.")


if __name__ == '__main__':
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    opt = Options()
    train(opt, 'dataset-objects/human_faces_paths.pkl')
