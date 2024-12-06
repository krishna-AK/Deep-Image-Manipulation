import os
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from datetime import datetime  # For timestamps

from prepare_dataset_object import DatasetObj
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
from options.Options import Options
from util import IterationCounter

def createAndSetModelFolder(opt):
    folder_name = (f"Folder_SC{opt.spatial_code_ch}_GC{opt.global_code_ch}_Res{opt.netG_num_base_resnet_layers}"
                   f"_DSsp{opt.netE_num_downsampling_sp}_DSgl{opt.netE_num_downsampling_gl}_Ups{opt.netG_no_of_upsamplings}"
                   f"_PS{opt.patch_size}-{opt.dataset_name}")
    model_dir = os.path.join(opt.save_training_models_dir, folder_name)
    os.makedirs(model_dir, exist_ok=True)
    opt.model_dir = model_dir
    return model_dir


def logLosses(loss_log, dir):
    log_file_name = os.path.join(dir, 'training_log.txt')
    os.makedirs(dir, exist_ok=True)
    with open(log_file_name, 'a') as f:
        f.write('\n'.join(loss_log) + '\n')
        f.write('-' * 80 + '\n')



def train(rank, world_size, opt, dataset_object_path):
    # Initialize process group if running on multiple GPUs
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set the device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Create model folder on rank 0 only
    if rank == 0:
        createAndSetModelFolder(opt)

    # Load dataset
    data = DatasetObj.load(dataset_object_path)

    # Limit the dataset
    subset_size = 70000
    subset_indices = torch.arange(subset_size)
    limited_data = Subset(data, subset_indices)

    # DistributedSampler ensures the dataset is split across GPUs
    sampler = DistributedSampler(
        limited_data, num_replicas=world_size, rank=rank, shuffle=True
    )

    dataset = DataLoader(
        limited_data,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )
    iter_counter = IterationCounter(opt)

    # Initialize model and optimizer
    model = SwappingAutoencoderModel(opt).to(device)
    optimizer = SwappingAutoencoderOptimizer(model)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])


    loss_log = []
    total_epochs = opt.total_epochs

    try:
        while not iter_counter.completed_training():
            
            print(f"Rank {rank}: Starting epoch {iter_counter.epochs_completed + 1}/{total_epochs}")

            # Synchronize across all GPUs before starting an epoch
            if world_size > 1:
                dist.barrier()

            sampler.set_epoch(iter_counter.epochs_completed)

            for batch_idx, cur_data in enumerate(dataset):
                cur_data = [item.to(device, non_blocking=True) for item in cur_data]

                # Forward and backward passes
                losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)

                # # Aggregate losses across all GPUs
                # aggregated_losses = {}
                # for key, value in losses.items():
                #     tensor = torch.tensor(value, device=device)
                #     if world_size > 1:
                #         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                #     aggregated_losses[key] = tensor.item() / world_size

                # Log losses on rank 0
                if iter_counter.steps_so_far %(opt.batch_size) == 0 and rank == 0:
                    print(f"Iteration {iter_counter.steps_so_far}: ", end="")
                    for key, value in losses.items():
                        print(f"{key}: {value:.4f}", end=", ")
                    print()

                # Log all information
                log_entry = (
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Epoch {iter_counter.epochs_completed + 1}, "
                    f"Iteration {iter_counter.steps_so_far}, "
                    f"Losses: {', '.join(f'{key}: {losses[key]:.4f}' for key in sorted(losses.keys()))}, "
                )
                loss_log.append(log_entry)

                iter_counter.record_one_iteration()

            iter_counter.record_one_epoch()
            if rank == 0:
                print(f"Epoch {iter_counter.epochs_completed} completed.")

                # Save model and log losses
                optimizer.save(iter_counter.steps_so_far)
                logLosses(loss_log, opt.model_dir)
                loss_log.clear()

        if rank == 0:
            print("Training completed.")

    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
    finally:
        if world_size > 1:
            dist.destroy_process_group()
        print(f"Rank {rank}: Training finished.")



def train_parallel(opt, dataset_obj_path):
    # Constants for GPU usage
    MAX_GPUS = 2  # Adjust this to limit the number of GPUs used
    total_gpus = torch.cuda.device_count()
    world_size = min(total_gpus, MAX_GPUS)

    if world_size == 0:
        raise RuntimeError("No GPUs available!")

    print(f"Using {world_size} out of {torch.cuda.device_count()} available GPUs.")

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Use localhost for single-node training
    os.environ["MASTER_PORT"] = "29500"     # Default port; you can change this if needed

    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            args=(world_size, opt, dataset_obj_path),
            nprocs=world_size
        )
    else:
        # For single GPU, set these variables explicitly
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["RANK"] = str(0)
        train(0, 1, opt, dataset_obj_path)


if __name__ == '__main__':
     # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    train_parallel(opt=Options(), dataset_obj_path='dataset-objects/ffhq-face-data-set.pkl')
    
    gc.collect()
    torch.cuda.empty_cache()
