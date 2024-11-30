class Options:
    def __init__(self):
        self.name = "train-default"  # Requires explicit setting
        self.easy_label = ""
        self.num_gpus = 1
        self.checkpoints_dir = './checkpoints/'
        self.model = 'swapping_autoencoder'
        self.optimizer = 'swapping_autoencoder'
        self.phase = 'train'
        self.num_classes = 0

        # input/output sizes
        self.preprocess = 'scale_width_and_crop'
        self.load_size = 128
        self.crop_size = 128
        self.preprocess_crop_padding = None
        self.no_flip = False
        self.shuffle_dataset = None

        # for setting inputs
        self.dataroot = "."
        self.dataset_mode = 'lmdb'
        self.nThreads = 8
        self.device = 'cuda'
        # networks
        self.netG = "StyleGAN2Resnet"
        self.netD = "StyleGAN2"
        self.netE = "StyleGAN2Resnet"
        self.netPatchD = "StyleGAN2"
        self.use_antialias = True

        # Additional parameters
        self.spatial_code_ch = 8
        self.global_code_ch = 512
        self.netE_num_downsampling_sp = 3
        self.netE_num_downsampling_gl = 1
        self.netG_num_base_resnet_layers = 3
        self.netG_no_of_upsamplings = 3

        #     train params
        self.isTrain = True
        self.batch_size = 16
        self.continue_train = False
        self.resume_iter = "0"
        self.resume_epochs_completed = 0

        # iter attributes
        # self.total_nimgs = 1000000000
        self.total_epochs = 10
        self.save_freq = 1
        # self.evaluation_freq = 500000000000
        self.print_freq = 10
        # self.display_freq = 1000000000000000

        # Additional parameters
        self.netG_scale_capacity = 1
        self.netE_scale_capacity = 1
        self.netG_use_noise = True
        self.netE_nc_steepness = 2
        self.lambda_R1 = 10.0
        self.lambda_patch_R1 = 1.0
        self.lambda_L1 = 1.0
        self.lambda_GAN = 1.0
        self.lambda_PatchGAN = 1.0
        self.patch_size = self.load_size // 4
        self.patch_min_scale = 1 / 8
        self.patch_max_scale = 1 / 4
        self.patch_num_crops = 8
        self.patch_use_aggregation = True
        self.netPatchD_scale_capacity = 2.0
        self.netPatchD_max_nc = 256 + 128

        self.max_num_tiles = 8
        self.patch_random_transformation = False

        # optimizer attributes
        self.lr = 0.002
        self.beta1 = 0.0
        self.beta2 = 0.99
        self.R1_once_every = 16

        self.save_training_models_dir = 'train_models/'
        self.model_dir = 'default'
        self.dataset_name = 'dummy'


    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                if hasattr(self, key):
                    setattr(self, key, self.parse_value(key, value))

    @staticmethod
    def parse_value(key, value):
        if key in ['num_gpus', 'num_classes', 'batch_size', 'load_size', 'crop_size', 'preprocess_crop_padding', 'nThreads',
                   'spatial_code_ch', 'global_code_ch', 'netG_num_base_resnet_layers', 'netG_scale_capacity', 'netE_num_downsampling_sp']:
            return int(value)
        elif key in ['no_flip', 'use_antialias', 'netG_use_noise']:
            return value.lower() in ['true', 'yes', '1']
        elif key in ['shuffle_dataset']:
            return None if value == 'None' else value.lower() in ['true', 'yes', '1']
        else:
            return value

if __name__ == '__main__':
    # Example usage
    options = Options()
    options.load_from_file('config.txt')
    print(options.name, options.num_gpus)  # Just to test
