from train import train
from options.Options import Options
import torch
import gc




if __name__ == '__main__':
    opt = Options()
    opt.batch_size = 8
    opt.total_epochs = 1
    opt.dataset_name = 'ffhq-128'


    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    train(opt=opt, dataset_object_path='dataset-objects/ffhq-face-data-set.pkl')