import glob
import os

import torch
import util
# from models import MultiGPUModelWrapper
# from optimizers.base_optimizer import BaseOptimizer


class SwappingAutoencoderOptimizer():
    """ Class for running the optimization of the model parameters.
    Implements Generator / Discriminator training, R1 gradient penalty,
    decaying learning rates, and reporting training progress.
    """
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument("--lr", default=0.002, type=float)
    #     parser.add_argument("--beta1", default=0.0, type=float)
    #     parser.add_argument("--beta2", default=0.99, type=float)
    #     parser.add_argument(
    #         "--R1_once_every", default=16, type=int,
    #         help="lazy R1 regularization. R1 loss is computed "
    #              "once in 1/R1_freq times",
    #     )
    #     return parser

    def __init__(self, model):
        self.opt = model.opt
        opt = self.opt
        self.model = model
        self.train_mode_counter = 0
        self.discriminator_iter_counter = 0

        self.Gparams = self.model.get_parameters_for_mode("generator")
        self.Dparams = self.model.get_parameters_for_mode("discriminator")

        # self.optimizer_G = torch.optim.Adam(
        #     self.Gparams, lr=opt.lr_G, betas=(opt.beta1, opt.beta2)
        # )

        self.optimizer_G = torch.optim.RMSprop(self.Gparams, lr=opt.lr_G)

        # c.f. StyleGAN2 (https://arxiv.org/abs/1912.04958) Appendix B
        # c = opt.R1_once_every / (1 + opt.R1_once_every)
        # self.optimizer_D = torch.optim.Adam(
        #     self.Dparams, lr=opt.lr_D*c, betas=(opt.beta1**c, opt.beta2**c)
        # )

        self.optimizer_D = torch.optim.RMSprop(self.Dparams, lr=opt.lr_D)

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load(int(self.opt.resume_iter.replace("k", "")))

    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)

    def prepare_images(self, data_i):
        return data_i[0]

    # def toggle_training_mode(self):
    #     modes = ["discriminator","discriminator","discriminator", "generator"]
    #     self.train_mode_counter = (self.train_mode_counter + 1) % len(modes)
    #     return modes[self.train_mode_counter]

    def toggle_training_mode(self):
        self.train_mode_counter = self.train_mode_counter + 1
        if (self.train_mode_counter)%2 == 1:
            return "generator"
        return "discriminator"

    def train_one_step(self, data_i, total_steps_so_far):
        images_minibatch = self.prepare_images(data_i)
        if self.toggle_training_mode() == "discriminator":
            losses = self.train_discriminator_one_step(images_minibatch)
        else:
            losses = self.train_generator_one_step(images_minibatch)
        return util.to_numpy(losses)

    # def train_one_step(self, data_i, total_steps_so_far):
    #     device = torch.device("cuda:0")
        
    #     # Move tensors to GPU individually to reduce memory spikes
    #     images_minibatch = [img.to(device) for img in data_i]
        
    #     # Combine tensors after moving to GPU
    #     images_minibatch = torch.cat(images_minibatch, dim=0)

    #     # Toggle training mode between generator and discriminator
    #     if self.toggle_training_mode() == "generator":
    #         losses = self.train_discriminator_one_step(images_minibatch)
    #     else:
    #         losses = self.train_generator_one_step(images_minibatch)

    #     # Convert losses to numpy for easier processing
    #     return util.to_numpy(losses)

    def train_generator_one_step(self, images):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)
        sp_ma, gl_ma = None, None
        self.optimizer_G.zero_grad()
        # g_losses, g_metrics = self.model(
        #     images, sp_ma, gl_ma, command="compute_generator_losses"
        # )
        g_losses, g_metrics = self.model.compute_generator_losses(images,sp_ma,gl_ma)
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer_G.step()
        g_losses.update(g_metrics)
        return g_losses

    def train_discriminator_one_step(self, images):
        if self.opt.lambda_GAN == 0.0 and self.opt.lambda_PatchGAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        # d_losses, d_metrics, sp, gl = self.model(
        #     images, command="compute_discriminator_losses"
        # )
        d_losses, d_metrics, sp, gl = self.model.compute_discriminator_losses(images)
        self.previous_sp = sp.detach()
        self.previous_gl = gl.detach()
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()

        needs_R1 = self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.opt.R1_once_every == 0
        if needs_R1_at_current_iter:
            self.optimizer_D.zero_grad()
            # r1_losses = self.model(images, command="compute_R1_loss")
            r1_losses = self.model.compute_R1_loss(images)
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])
        d_losses.update(d_metrics)
        return d_losses

    def get_visuals_for_snapshot(self, data_i):
        images = self.prepare_images(data_i)
        with torch.no_grad():
            return self.model(images, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far, save_optimizer_state = True):
        # self.model.save(total_steps_so_far)

        """
               Save the model's state and optionally the optimizer's state. Old model files are deleted
               only after a successful save of the new model.

               :param total_steps_so_far:
               :param epoch: Current epoch number, used for naming the saved files.
               :param save_optimizer_state: Flag to indicate if optimizer state should be saved.
               """

        steps = total_steps_so_far
        save_dir = self.model.opt.model_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        new_model_filenames = [
            f'E_steps_{steps}.pth',
            f'G_steps_{steps}.pth',
            f'D_steps_{steps}.pth',
            f'Dpatch_steps_{steps}.pth',
            f'optimizer_G_steps_{steps}.pth' if save_optimizer_state else '',
            f'optimizer_D_steps_{steps}.pth' if save_optimizer_state else ''
        ]
        new_model_paths = [os.path.join(save_dir, filename) for filename in new_model_filenames if filename]

        try:
            # Save state dictionaries of the models
            torch.save(self.model.E.state_dict(), new_model_paths[0])
            torch.save(self.model.G.state_dict(), new_model_paths[1])
            torch.save(self.model.D.state_dict(), new_model_paths[2])
            if self.opt.lambda_PatchGAN > 0:
                torch.save(self.model.Dpatch.state_dict(), new_model_paths[3])
            if save_optimizer_state:
                torch.save(self.optimizer_G.state_dict(), new_model_paths[4])
                torch.save(self.optimizer_D.state_dict(), new_model_paths[5])

            # Delete old model files after successful save
            existing_pth_files = glob.glob(os.path.join(save_dir, '*.pth'))
            for file in existing_pth_files:
                if 'steps_' in os.path.basename(file) and os.path.basename(file) not in new_model_filenames:
                    os.remove(file)

            print(f"Model and optimizer states saved at steps {steps}")

        except IOError as e:
            print(f"Error saving model at steps {steps}: {e}")

    def load(self, steps):
        """
        Load the model's state from saved files.

        :param load_dir: Directory where the model's state was saved.
        :param epoch: Epoch number from which to load the saved states.
        """

        load_dir = self.opt.model_dir
        # File paths for the saved states
        optimizer_G_path = os.path.join(load_dir, f'optimizer_G_steps_{steps}.pth')
        optimizer_D_path = os.path.join(load_dir, f'optimizer_D_steps_{steps}.pth')

        # Check if the files exist
        if not os.path.exists(optimizer_G_path) or not os.path.exists(optimizer_D_path):
            print("Saved model states not found in the specified directory.")
            return

        # Load the states into the model
        self.optimizer_G.load_state_dict(torch.load(optimizer_G_path,map_location=self.opt.device))
        self.optimizer_D.load_state_dict(torch.load(optimizer_D_path, map_location=self.opt.device))

        print(f"Optimizer loaded from epoch {steps}")
