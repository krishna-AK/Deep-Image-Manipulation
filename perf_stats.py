import time
import gc
import torch
from datetime import datetime
from prepare_dataset_object import DatasetObj
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
from options.Options import Options
from util import IterationCounter
from torch.utils.data import Subset, DataLoader

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

def run_training_for_one_minute(opt, dataset_object_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Aborting training.")
    
    device = torch.device('cuda')
    data = DatasetObj.load(dataset_object_path)

    # Limit the dataset to 70,000 samples
    subset_size = 70000
    subset_indices = torch.arange(subset_size)
    limited_data = Subset(data, subset_indices)

    dataset = DataLoader(
        limited_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=16,
        persistent_workers=True
    )

    model = SwappingAutoencoderModel(opt).to(device)
    optimizer = SwappingAutoencoderOptimizer(model)
    iter_counter = IterationCounter(opt)

    # Metrics
    total_data_time = 0
    total_train_time = 0
    total_batches = 0

    start_time = time.time()
    dataset_iterator = Prefetcher(dataset, device)

    try:
        while time.time() - start_time < 60:  # Run for 1 minute
            for batch_idx, cur_data in enumerate(dataset_iterator):
                # Measure data loading time
                data_start_time = time.time()

                # Data is already preloaded to GPU
                total_data_time += time.time() - data_start_time

                # Measure training time
                train_start_time = time.time()
                optimizer.train_one_step(cur_data, iter_counter.steps_so_far)
                total_train_time += time.time() - train_start_time

                total_batches += 1

                if time.time() - start_time >= 60:
                    break

    except Exception as e:
        print(f"Error during profiling: {e}")

    finally:
        # Collect metrics
        avg_data_time = total_data_time / total_batches if total_batches else 0
        avg_train_time = total_train_time / total_batches if total_batches else 0

        print("\nProfiling Results:")
        print(f"Total Batches Processed: {total_batches}")
        print(f"Average Data Loading Time per Batch: {avg_data_time:.4f} seconds")
        print(f"Average Training Time per Batch: {avg_train_time:.4f} seconds")
        print(f"Total Time Spent on Data Loading: {total_data_time:.2f} seconds")
        print(f"Total Time Spent on Training: {total_train_time:.2f} seconds")

if __name__ == '__main__':
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    opt = Options()
    run_training_for_one_minute(opt, 'dataset-objects/human_faces_paths.pkl')
