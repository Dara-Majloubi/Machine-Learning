# Vision Networks & Fast Training

Implementation and evaluation of vision-network architectures, transfer learning, hardware acceleration, and residual connections using PyTorch.

## Overview

This project explores practical techniques for training and adapting convolutional neural networks efficiently.

The notebook combines hardware-aware training with established vision architectures and transfer-learning techniques. It also introduces skip connections and pre-residual network blocks for deeper architectures.

## Topics Covered

- Vision network architectures
- LeNet-5
- AlexNet
- GPU acceleration with PyTorch
- Training and evaluation modes
- VGG architectures
- CIFAR-10
- Transfer learning
- Pretrained model weights
- Feature extraction
- Freezing network parameters
- Efficient classifier training
- Parallel data loading
- Skip connections
- Pre-residual networks

## Hardware Acceleration

The project demonstrates how PyTorch models and tensors can be moved between CPU and GPU devices.

The training and evaluation workflow is adapted so that:

- Input tensors are placed on the same device as model parameters.
- The network is switched between training and evaluation modes.
- Modules such as Dropout and Batch Normalisation behave correctly depending on the mode.

## VGG and CIFAR-10

The project adapts the feature-extraction architecture of VGG for the CIFAR-10 classification task.

The implementation uses:

- VGG-style convolutional feature extraction
- Global average pooling
- A classifier adapted to CIFAR-10 classes
- Pretrained VGG feature representations

## Transfer Learning

Pretrained VGG-11 weights are used to initialise the convolutional feature extractor.

The classifier is then adapted and trained for the CIFAR-10 task.

This demonstrates a common transfer-learning workflow:

1. Load a pretrained vision model.
2. Reuse its learned feature extractor.
3. Freeze the pretrained parameters.
4. Train a new classifier for the target dataset.

## Efficient Training

The training pipeline is designed to reduce computational overhead by:

- Training only the classifier while keeping the feature extractor frozen.
- Upscaling CIFAR-10 images to the input resolution expected by VGG.
- Using GPU acceleration.
- Using a parallel data loader.
- Separating training and validation data.
- Evaluating the classifier during training.

## Skip Connections

The project introduces residual learning and skip connections.

A basic residual mapping can be expressed as:

`x + f(x)`

Skip connections provide a direct path for information and gradients through deeper networks.

The notebook then explores pre-residual connections, where the activation is applied before the residual transformation.

## Pre-Residual Networks

A `PreResBlock` is implemented as a building block for pre-residual convolutional networks.

The implementation considers:

- Two-layer convolutional residual paths
- Configurable kernel size
- Spatial downsampling through stride
- Different input and output channel dimensions
- Efficient skip connections
- Preservation of spatial dimensions when appropriate

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Jupyter Notebook
- CIFAR-10
- CUDA / GPU acceleration

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`06-vision-networks-and-fast-training.ipynb`
