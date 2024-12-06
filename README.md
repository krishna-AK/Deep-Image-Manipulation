Project Setup Guide
Follow these steps to set up, train, and test the model effectively:

1. Install Required Packages
Ensure you have Python installed.
Run the following command to install all necessary packages using the requirements.txt file:

pip install -r requirements.txt

2. Set up the Datasets Folder
Create a folder named datasets in your project directory.
Inside the datasets folder, add a subfolder containing all your training images.
Refer to the sample script setup_dataset.py for guidance on organizing the dataset folder.

3. Generate a Dataset Object File
Use the path of your datasets folder to create a dataset object file (.pkl) that will be used for training.
This file encapsulates your dataset for efficient training.
Refer to setup_dataset.py for assistance in creating the dataset object file.

4. Configure the Training Script
Open the train.py script.
Provide the path to the generated dataset object file (.pkl) in the script.
Once configured, you can start training the model.

5. Training Outputs
During training, a unique folder is created inside the train_models directory for each architecture configuration.
This folder contains:
The saved model checkpoints.
The corresponding training logs.

6. Testing the Model
Use the test.py script to test the model.
Provide the path to the folder where the trained models are saved.

7. Plotting Training Losses
Use the plot_losses.py script to visualize the training losses.
This helps in analyzing the model's performance over the training period.

7. Multi-GPU Support
The training script (train.py) is designed to work seamlessly in multi-GPU environments.
Configure the maximum number of GPUs to use in the script.
At the start of training, the script will display the number of GPUs being utilized.
