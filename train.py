import itertools
import sys
import os
import data
import torch
from torch.utils.data import DataLoader

import models
import optimizers

from ValorantDataset import ValorantDataset
from AnimeDataset import AnimeDataset
from models.swapping_autoencoder_model import SwappingAutoencoderModel
from optimizers.swapping_autoencoder_optimizer import SwappingAutoencoderOptimizer
from options import TrainOptions
from options.Options import Options
from util import IterationCounter
# from util import Visualizer
from util import MetricTracker
# from evaluation import GroupEvaluator



def createAndSetModelFolder(opt):
    # Create a folder name using the given values
    folder_name = (f"Folder_SC{opt.spatial_code_ch}_GC{opt.global_code_ch}_Res{opt.netG_num_base_resnet_layers}"
                   f"_DSsp{opt.netE_num_downsampling_sp}"
                   f"_DSgl{opt.netE_num_downsampling_gl}_Ups{opt.netG_no_of_upsamplings}_PS{opt.patch_size}")

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        opt.save_training_models_dir += folder_name+"/"
        opt.model_config_str = folder_name
        return folder_name
    else:
        opt.save_training_models_dir += folder_name + "/"
        opt.model_config_str = folder_name
        return folder_name


def logLosses(loss_log, dir):
    log_file_name = os.path.join(dir, 'loss_log.txt')

    # Check if the directory exists, create if not
    if not os.path.exists(dir):
        os.makedirs(dir)

    try:
        with open(log_file_name, 'a') as f:
            f.write('\n'.join(loss_log) + '\n')
        print("losses logged!")
        # Flush the log
    except IOError as e:
        print(f"Error writing to log file: {e}")


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt = Options()
createAndSetModelFolder(opt)

data = AnimeDataset.load('datasets/flickr_faces_dataset_paths.pkl')
dataset = DataLoader(data, batch_size=opt.batch_size, shuffle=True)  # Adjust batch size as needed
dataset_iterator = itertools.cycle(dataset)

opt.dataset = dataset
iter_counter = IterationCounter(opt)

metric_tracker = MetricTracker(opt)

# evaluators = GroupEvaluator(opt)

model = SwappingAutoencoderModel(opt)
optimizer = SwappingAutoencoderOptimizer(model)

loss_log = []
log_losses_every_in_batches = opt.batch_size*100
try:
    while not iter_counter.completed_training():
        with iter_counter.time_measurement("data"):
            cur_data = next(dataset_iterator)
            if len(cur_data) != opt.batch_size:
                continue

        with iter_counter.time_measurement("train"):
            losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)
            # print(losses)
            print("iter : ",iter_counter.steps_so_far)
            for key in sorted(losses.keys()):
                print(key+" : "+str(losses[key]),end = ',')
            print(end = '\n')
            metric_tracker.update_metrics(losses, smoothe=True)

            # Log losses
            loss_str = f"Iteration :{iter_counter.steps_so_far}, " + ", ".join(
                f"{key}: {losses[key]}" for key in sorted(losses.keys()))
            loss_log.append(loss_str)

            # Every 1000 iterations, write losses to a file and reset the log
            # if iter_counter.steps_so_far % log_losses_every_in_batches == 0:
            #     logLosses(loss_log,opt.save_training_models_dir)
            #     loss_log.clear()

        # with iter_counter.time_measurement("maintenance"):
            # if iter_counter.needs_printing():
            #     visualizer.print_current_losses(iter_counter.steps_so_far,
            #                                     iter_counter.time_measurements,
            #                                     metric_tracker.current_metrics())

            # if iter_counter.needs_displaying():
            #     visuals = optimizer.get_visuals_for_snapshot(cur_data)
            #     visualizer.display_current_results(visuals,
            #                                        iter_counter.steps_so_far)

            # if iter_counter.needs_evaluation():
            #     metrics = evaluators.evaluate(
            #         model, dataset, iter_counter.steps_so_far)
            #     metric_tracker.update_metrics(metrics, smoothe=False)

            if iter_counter.needs_saving() and iter_counter.steps_so_far != int(opt.resume_iter):
                optimizer.save(iter_counter.steps_so_far)
                logLosses(loss_log, opt.save_training_models_dir)
                loss_log.clear()

            if iter_counter.completed_training():
                print("Training ended by iter")
                sys.exit(0)
                break

            iter_counter.record_one_iteration()
except Exception as e:
    print(e)
finally:
    # optimizer.save(iter_counter.steps_so_far)
    print('Training finished.')

