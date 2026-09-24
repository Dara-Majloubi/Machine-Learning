# Regularisation, Initialisation & Normalisation

Implementation of fundamental techniques for improving the training and generalisation of neural networks.

## Overview

This project explores three important aspects of neural-network training:

- Regularisation to reduce overfitting
- Parameter initialisation to support effective learning
- Normalisation to improve information and gradient flow during training

The implementations are developed within a modular neural-network framework using NumPy.

## Topics Covered

### Regularisation

The project implements **inverted dropout**, a commonly used regularisation technique for neural networks.

The dropout module includes:

- Training-mode behaviour
- Prediction/inference-mode behaviour
- Random neuron masking
- Scaling of activations during training
- Forward propagation
- Backward propagation

### Parameter Initialisation

The project implements **Xavier (Glorot) uniform initialisation**.

The implementation considers:

- Fan-in
- Fan-out
- Parameter variance
- Gain factors
- Reproducible random initialisation
- Convolutional layer parameter shapes

Xavier initialisation is designed to maintain suitable variance of signals through a network at the beginning of training.

### Batch Normalisation

The project implements **Batch Normalisation** with:

- Batch mean and variance
- Normalisation of activations
- Learnable scale parameter `gamma`
- Learnable shift parameter `beta`
- Moving-average statistics
- Training-mode behaviour
- Inference-mode behaviour
- Forward propagation
- Backward propagation

## Implementation

The techniques are implemented as modular components and integrated into the neural-network framework.

The notebook focuses on the underlying mathematical operations and gradient computations rather than relying on high-level deep-learning libraries.

## Technologies

- Python
- NumPy
- Jupyter Notebook
- NNumpy

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`05-regularisation-initialisation-normalisation.ipynb`
