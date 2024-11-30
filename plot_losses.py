import matplotlib.pyplot as plt


def plot_losses_from_log(log_file_path):
    # Initialize lists to store iteration numbers and loss values
    iterations = []
    loss_values = {}
    iter_val = -1
    # Read and process the log file
    with open(log_file_path, 'r') as file:
        try:
            for line in file:
                parts = line.strip().split(', ')
                # if len(parts) == 2:
                epoch, iteration, losses = parts[0], parts[1], parts[2:]
                z = int(iteration.split(' ')[1].strip())
                # epoch = int(epoch.split(' ')[1].strip())
                # print(epoch)
                # if epoch < 0:
                #     continue
                if z > iter_val:
                    iter_val = z
                else:
                    continue
                # iteration = int(iteration.split()[1])  # Extract iteration number

                # Process individual losses
                for loss in losses:
                    key, value = loss.split(':')
                    key = key.strip()
                    value = float(value.strip())

                    if key not in loss_values:
                        loss_values[key] = []
                    loss_values[key].append(value)
        except Exception:
            pass

    # Plotting
    plt.figure(figsize=(10, 6))
    # t_losses = []
    # for i in range(100):
    #     t_loss = 0
    #     for key in loss_values.keys():
    #         t_loss += loss_values[key][i]
    #     t_losses.append(t_loss)
    #
    # plt.plot([i * 16 for i in range(len(t_losses))], t_losses, label='total_loss')
    all_keys = loss_values.keys()
    plot_keys = ['D_total','G_L1','G_GAN_rec']
    # G_losses = ['G_L1','G_GAN_mix', 'G_GAN_rec','G_mix']
    # G_losses = ['G_GAN_rec','G_GAN_mix']
    D_losses = ['D_real','D_mix','D_rec']
    # D_losses = ['D_rec']
    patch_losses = ['PatchD_mix','PatchD_real']
    for key in D_losses:
        values = [loss_values[key][i] for i in range(0,len(loss_values[key]),1)]
        plt.plot([i*16 for i in range(0,len(values))], values, label=key)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
log_file_path = 'train_models\Folder_SC8_GC512_Res3_DSsp3_DSgl1_Ups3_PS32\loss_log.txt'  # Replace with your actual log file path
plot_losses_from_log(log_file_path)
