# Image Segmentation with UNet

## Description
This project implements an image segmentation model using the UNet architecture. 
The code is designed to load a dataset of images and masks, train the UNet model, and evaluate its performance.

## Requirements
The project requires the following libraries:
- Python 3.9+
- PyTorch
- WandB (Weights & Biases) for logging
- OpenCV
- NumPy
- Matplotlib
- ...

To install the required packages, run:
pip install -r requirements.txt

## File Structure
The main files and directories in the project are organized as follows:

- `src/`: Contains the main source code files.
  - `train.py`: Main script to train the model.
  - `model.py`: Defines the UNet model architecture.
  - `dataloader.py`: Loads the image and mask datasets.

- `data/`: Directory for the dataset (not included in this repo; place your dataset here).
  - `cells/`: Subdirectory containing image and mask files.

- `wandb/`: Directory for W&B.
- `run.sh`: Run the 'train.py' code.
- `README.txt`: Instructions on how to run the code.
- `requirements.txt`: List of required libraries for the project.

## Usage
### 1. Data Preparation
Ensure that your dataset is in the `data/cells/` directory. Each image should have a corresponding mask file with the same file name format.

### 2. Training the Model
To train the UNet model, navigate to the `src` directory and run `run.sh`:

### 3. Hyperparameter Configuration
You can adjust the hyperparameters directly in the `train.py` file, such as `learning_rate`, `batch_size`, and `epochs`.

### 4. Logging with Weights & Biases
Ensure you have your Weights & Biases API key configured (see wandb documentation or the code comments for setup instructions).

## Important Details
- **Model Saving**: The model's state is saved as `checkpoint.pth` in the current directory.
- **Training Details**: Training and validation loss and accuracy are logged for each epoch.

For more information, please refer to the source code and comments in each script.