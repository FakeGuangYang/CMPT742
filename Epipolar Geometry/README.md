# Eight-Point Algorithm and Fundamental Matrix Estimation

## Overview

This project implements the Eight-Point Algorithm for estimating the fundamental matrix between two images. The code includes:
- A custom implementation of the algorithm.
- A comparison with OpenCV’s built-in function. 
- Visualization of epipolar geometry. 
- Optional support for RANSAC to handle outliers in keypoint matches.

## Project Structure

- `/data`: Example images
- `eight_point_fw.py`: Defines the SSD model architecture.
- `main.py`: Main implementation of the Eight-Point Algorithm
- `requirements.txt`: List of required libraries for the project.
- `run.sh`: Shell script to execute `eight_point_fw.py`.
- `README.md`: Instructions on how to set up and run the project.

## Requirements

To install the required packages, run:
`pip install -r requirements.txt`

## Usage
To run the code, run `bash run.sh` on the command line.

For more information, please refer to the source code and comments in each script.