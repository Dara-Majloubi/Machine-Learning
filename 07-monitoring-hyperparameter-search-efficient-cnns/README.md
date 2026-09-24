# Monitoring, Hyperparameter Search & Advanced CNNs

Implementation and evaluation of training monitoring, hyperparameter search, efficient convolutional neural networks, and alternative vision architectures using PyTorch.

## Overview

This project explores practical techniques for monitoring and improving neural-network training.

The notebook starts with training monitoring using Weights & Biases and then investigates different hyperparameter search strategies. It also implements an efficient convolutional neural network with a constrained parameter budget and compares CNN-based architectures with an MLP-Mixer.

## Topics Covered

- Training monitoring
- Weights & Biases (W&B)
- PyTorch training loops
- Hyperparameter search
- Manual search
- Grid search
- Random search
- Bayesian optimisation
- Neural Architecture Search
- Efficient CNNs
- Depthwise separable convolutions
- Grouped convolutions
- CIFAR-10 classification
- MLP-Mixer
- Token mixing
- Channel mixing
- Residual connections
- Layer normalisation
- Inductive biases in vision architectures

## Training Monitoring

A reusable `Trainer` class is implemented to organise the neural-network training process.

The monitoring functionality records:

- Per-batch training loss
- Average training loss
- Validation loss
- Training progress across epochs

Weights & Biases is used to organise and compare training runs.

## Hyperparameter Search

The project introduces several approaches for finding suitable neural-network hyperparameters:

### Manual Search

Reasonable hyperparameter combinations are selected and evaluated manually.

### Grid Search

A predefined set of values for different hyperparameters is combined to evaluate multiple configurations systematically.

### Random Search

Hyperparameter configurations are sampled randomly instead of evaluating every possible combination.

### Bayesian Optimisation

Previous evaluations are used to guide the selection of promising configurations through a surrogate model and acquisition function.

### Neural Architecture Search

The project also introduces Neural Architecture Search as a broader approach to automatically exploring neural-network architectures.

## Efficient CNN

An efficient CNN is implemented for CIFAR-10 under a strict parameter budget.

The architecture uses depthwise separable convolutions consisting of:

1. Depthwise convolution
2. Pointwise 1×1 convolution

Batch normalisation, ReLU activations, pooling, and global average pooling are used to construct the network.

The model is designed to use fewer than 30,000 trainable parameters.

## CIFAR-10 Training

The efficient CNN is trained on the CIFAR-10 dataset.

Hyperparameter search is used to investigate:

- Learning rate
- Optimizer choice
- Model configuration

The experiments use a fixed batch size and evaluate the resulting cross-entropy loss.

## MLP-Mixer

The project implements an MLP-Mixer architecture for CIFAR-10 without convolutions or attention.

The model consists of:

- Patch embedding
- Token-mixing MLPs
- Channel-mixing MLPs
- Residual connections
- Layer normalisation
- Global average pooling
- Linear classification head

The implementation is constrained to fewer than 30,000 parameters.

## CNN vs MLP-Mixer

The project discusses the different inductive biases of CNNs and MLP-Mixers.

CNNs introduce strong spatial inductive biases such as:

- Locality
- Translation-related structure
- Weight sharing

MLP-Mixers use fewer assumptions about spatial structure and instead learn interactions between image patches and feature channels.

The experiments provide a practical comparison of their training and generalisation behaviour on CIFAR-10.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Weights & Biases
- CIFAR-10
- Jupyter Notebook

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`07-monitoring-hyperparameter-search-efficient-cnns.ipynb`
