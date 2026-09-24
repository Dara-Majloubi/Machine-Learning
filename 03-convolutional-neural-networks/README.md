# Convolutional Neural Networks

Implementation of core components of Convolutional Neural Networks using NumPy.

## Overview

This project focuses on understanding and implementing the fundamental operations and building blocks used in Convolutional Neural Networks (CNNs).

The implementation starts from basic one-dimensional convolution operations and progresses to multi-channel two-dimensional convolutions and a complete convolutional layer.

## Topics Covered

- Cross-correlation
- One-dimensional convolution
- Multi-channel 2D convolution
- `im2col` / `sig2col` representation
- Convolutional layers
- Forward propagation
- Backward propagation
- Weight and bias gradients
- Gradient checking
- ReLU activation
- ELU activation

## Convolution Operations

The project implements one-dimensional cross-correlation and convolution without relying on high-level convolution functions such as `np.convolve` or `np.correlate`.

It then extends the implementation to multi-channel two-dimensional convolutions.

## Convolutional Layer

A modular `Conv2d` layer is implemented with support for:

- Trainable convolution kernels
- Optional bias parameters
- Forward computation
- Backward propagation
- Weight gradients
- Bias gradients
- Input gradients

Gradient checking is used to verify the correctness of the analytical gradients.

## Activation Functions

The project also implements two commonly used activation functions:

- ReLU (Rectified Linear Unit)
- ELU (Exponential Linear Unit)

Both forward and backward computations are implemented.

## Implementation

The core operations are implemented using **NumPy** and integrated into a modular neural-network framework.

The implementation emphasizes the underlying mathematical operations rather than relying on high-level deep learning libraries.

## Technologies

- Python
- NumPy
- Jupyter Notebook
- NNumpy

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`03-convolutional-neural-networks.ipynb`
