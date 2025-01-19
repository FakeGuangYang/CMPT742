# Object Detection with SSD

## Description
This project implements an object detection model using the Single Shot MultiBox Detector (SSD) architecture. 
The code is designed to load a dataset of images and annotations, train the SSD model, and evaluate its performance.

## Requirements
The project requires the following libraries:
- Python 3.9+
- PyTorch
- Albumentations
- OpenCV
- NumPy
- Matplotlib
- ...

To install the required packages, run:
pip install -r requirements.txt

## File Structure
The main files and directories in the project are organized as follows:

- `dataset.py`: Dataset preparation, including data augmentation and preprocessing.
- `model.py`: Defines the SSD model architecture.
- `main.py`: Main script for training, validation, and testing.
- `utils.py`: Utility functions such as Non-Maximum Suppression (NMS) and mAP computation.
- `requirements.txt`: List of required libraries for the project.
- `run.sh`: Shell script to execute the training or testing pipeline.
- `README.md`: Instructions on how to set up and run the project.
- `network.pth`: Trained model

## Usage
### 1. Data Preparation
Ensure that your dataset is in the `data/train/` and `data/test/` directories. Each image should have a corresponding annotation file with the same file name but .txt extension. The annotation files should follow the format specified in the project documentation.

### 2. Training the Model
To train the SSD model, navigate to the `materials` directory and run `run.sh`:

### 3. Hyperparameter Configuration
You can adjust the hyperparameters directly in the `main.py` file, such as `learning_rate`, `batch_size`, and `num_epochs`.

### 4. Logging and Visualization
Intermediate predictions, ground truths, and evaluation results are visualized during training and testing. Ensure that the display is supported or save the visualizations to files for review.

## Important Details
- **Model Saving**: The model’s state is saved as `network.pth` in the current directory during training.
- **Evaluation**: The `utils.py` script includes methods for evaluating model performance, such as generate_mAP for mAP computation and NMS for bounding box post-processing.
- **Error Handling**: Make sure your dataset annotations and image formats align with the input expectations of the model.

For more information, please refer to the source code and comments in each script.