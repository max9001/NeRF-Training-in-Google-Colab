# Project Outline: NeRF Training in Google Colab

## Project Idea
This project aims to collect code from various sources to create an accessible pipeline for training a NeRF model in Google Colab. This will make NeRF training feasible for users without a Linux machine or NVIDIA GPU.

## Photo Collection
### Photos of Checkerboard
  - Purpose: Required to calculate the intrinsic parameters of the user's camera.
  - Approach: Follow Zhang's method for camera calibration [[1]](https://doi.org/10.1109/34.888718}{10.1109/34.888718).

### Photos of Scene
  - Ambiguity: Requires research on best practices for capturing photos suitable for NeRF. 

## Camera Calibration
### Getting Intrinsic Parameters
  - Reuse code from HW3 to leverage OpenCV functions \texttt{findChessboardCorners()} and \texttt{calibrateCamera()}.

### Getting Poses
  - Use structure-from-motion algorithms, such as COLMAP, to obtain extrinsic parameters (rotation $R$ and translation $t$) as required by NeRF [[2]](https://github.com/Fyusion/LLFF/blob/master/llff/poses/colmap_wrapper.py#L23).

## Data Preparation
### Organize Calculations into JSON
  - Store the intrinsic and extrinsic parameters calculated in n a JSON file, associating each image path with its corresponding parameters.

### Train, Validation, and Test Split
  - Split the data into training, validation, and test sets for model evaluation.

## NeRF Model Training
### Leverage PyTorch Lightning
  - Use PyTorch Lightning to simplify the training process.

### Implement NeRF Model
  - Reference the PyTorch Lightning implementation of NeRF [[3]](https://github.com/yenchenlin/nerf-pytorch/tree/master).

### Save Best Model Checkpoint
  - Save the checkpoint of the best-performing model.

## Model Rendering
### Load Last Model Checkpoint
  - Reload the final model checkpoint for rendering.

### Render Video
  - Use the trained NeRF model to render a video.

### Save Results
  - Store the rendered video for user access.

