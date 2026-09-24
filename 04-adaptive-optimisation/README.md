# Adaptive Optimisation

Implementation and evaluation of optimization methods for training neural networks using NumPy.

## Overview

This project focuses on optimization algorithms used to train neural networks and on integrating these optimizers into a modular deep learning framework.

The notebook starts with stochastic gradient-based optimization and progressively introduces momentum and adaptive learning-rate methods.

The implemented optimizers are then used to train neural networks on the MNIST handwritten digit classification task.

## Topics Covered

- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch training
- Momentum
- Adaptive learning rates
- Adamax optimizer
- Training and evaluation loops
- Logistic regression
- Multi-layer neural networks
- Convolutional Neural Networks
- MNIST classification

## Optimization Methods

### Stochastic Gradient Descent

Instead of calculating gradients over the complete dataset, SGD uses mini-batches to reduce computational cost and memory requirements.

The notebook implements the mini-batch training procedure and integrates it with the neural-network framework.

### Momentum

Momentum accumulates information from previous gradient directions to improve optimization.

The implementation maintains an optimizer state and uses this state to determine the parameter update direction.

### Adamax

Adamax is an adaptive optimization algorithm related to Adam.

Instead of normalizing the gradient using the L2 norm, Adamax uses the L-infinity norm to control the parameter updates.

## Training and Evaluation

The project implements training and evaluation procedures for neural networks.

The training process includes:

- Forward propagation
- Loss computation
- Backpropagation
- Gradient computation
- Parameter updates
- Evaluation on unseen data
- Tracking training and evaluation loss

## MNIST Classification

The implemented optimization methods are evaluated on the MNIST handwritten digit classification dataset.

### Logistic Regression

A single-layer neural network is trained to classify MNIST digits using stochastic gradients and an adaptive optimizer.

### Convolutional Neural Network

A multi-layer convolutional neural network is trained on MNIST using the implemented optimization framework.

The experiment demonstrates how the optimization algorithms can be applied to deeper neural-network architectures.

## Implementation

The optimization algorithms are implemented and integrated into a modular neural-network framework using NumPy.

The project emphasizes the underlying optimization mechanisms rather than relying on high-level training APIs.

## Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook
- NNumpy
- MNIST

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`04-adaptive-optimisation.ipynb`
