# Machine Learning & Deep Learning Portfolio

A collection of machine learning and deep learning projects developed during my Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

The portfolio covers fundamental machine-learning concepts, neural-network implementation from first principles, convolutional neural networks, optimisation, regularisation, modern vision architectures, generative models, and adversarial learning.

## Overview

The projects demonstrate both theoretical understanding and practical implementation using Python, NumPy, and PyTorch.

The portfolio progresses from fundamental neural-network concepts to more advanced deep-learning architectures and generative models.

## Projects

### 01 — Regression Basics

Introduction to regression and fundamental machine-learning concepts.

Topics include:

- Regression
- Model training
- Loss functions
- Prediction
- Model evaluation
- Fundamental machine-learning concepts

### 02 — Multilayer Perceptrons

Implementation of core components of multilayer perceptrons and a small neural-network framework using NumPy.

Topics include:

- Modular neural-network design
- Forward propagation
- Backpropagation
- Activation functions
- Fully connected layers
- Trainable parameters
- Parameter gradients
- Numerical gradient checking
- One-hot encoding

### 03 — Convolutional Neural Networks

Implementation of fundamental CNN components using NumPy.

Topics include:

- Cross-correlation
- 1D convolution
- Multi-channel 2D convolution
- `im2col` / `sig2col`
- Convolutional layers
- Forward and backward propagation
- Weight and bias gradients
- Gradient checking
- ReLU
- ELU

### 04 — Adaptive Optimisation

Implementation and evaluation of optimisation methods for training neural networks.

Topics include:

- Gradient Descent
- Stochastic Gradient Descent
- Mini-batch training
- Momentum
- Adaptive learning rates
- Adamax
- Training and evaluation loops
- Logistic regression
- CNN training
- MNIST classification

### 05 — Regularisation, Initialisation & Normalisation

Implementation of techniques that improve neural-network training and generalisation.

Topics include:

- Inverted Dropout
- Xavier / Glorot initialisation
- Fan-in and fan-out
- Batch Normalisation
- Moving-average statistics
- Training and inference modes
- Forward and backward propagation

### 06 — Vision Networks & Fast Training

Exploration of vision architectures, transfer learning, and efficient training using PyTorch.

Topics include:

- LeNet-5
- AlexNet
- VGG
- CIFAR-10
- GPU acceleration
- Transfer learning
- Pretrained model weights
- Feature extraction
- Parameter freezing
- Parallel data loading
- Skip connections
- Pre-residual networks

### 07 — Monitoring, Hyperparameter Search & Efficient CNNs

Exploration of practical methods for monitoring and improving neural-network training.

Topics include:

- Weights & Biases
- Training monitoring
- Manual hyperparameter search
- Grid search
- Random search
- Bayesian optimisation
- Neural Architecture Search
- Efficient CNNs
- Depthwise separable convolutions
- Grouped convolutions
- CIFAR-10
- MLP-Mixer
- Token mixing
- Channel mixing
- Layer Normalisation
- Residual connections
- Inductive bias

### 08 — Auto-Encoders & Variational Auto-Encoders

Exploration of representation learning and generative modelling with auto-encoder architectures.

Topics include:

- Auto-Encoders
- Transposed convolutions
- Invertible / undoable convolutional operations
- Feature visualisation
- Convolutional auto-encoders
- Image reconstruction
- Variational Auto-Encoders
- Reparameterization trick
- Latent representations
- Gaussian latent variables
- Reconstruction loss
- Latent-space sampling
- Image generation
- MNIST

### 09 — Adversarial Training & Generative Adversarial Networks

Exploration of adversarial examples, adversarial training, and GANs.

Topics include:

- Adversarial examples
- Gradient-based adversarial attacks
- Projected Gradient Ascent
- Fooling neural networks
- Adversarial training
- Real/fake classification
- Discriminators
- Synthetic data generation
- Generative Adversarial Networks
- Generator networks
- Min-max optimisation
- Binary cross-entropy
- MNIST

## Technologies

- Python
- NumPy
- pandas
- scikit-learn
- PyTorch
- Torchvision
- Matplotlib
- Jupyter Notebook
- Weights & Biases
- MNIST
- CIFAR-10

## Learning Focus

The portfolio demonstrates progression across several areas of machine learning and deep learning:

**Foundations → Neural Networks → CNNs → Optimisation → Regularisation → Modern Vision → Hyperparameter Search → Generative Models → Adversarial Learning**

Particular emphasis is placed on understanding the underlying mechanisms of neural networks by implementing important components with NumPy before applying higher-level PyTorch workflows.

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Repository Structure

Each project is maintained as an individual Jupyter Notebook with a dedicated README where appropriate.

```text
machine-learning-projects/
│
├── 01-regression-basics.ipynb
├── 02-multilayer-perceptrons.ipynb
├── 03-convolutional-neural-networks.ipynb
├── 04-adaptive-optimisation.ipynb
├── 05-regularisation-initialisation-normalisation.ipynb
├── 06-vision-networks-and-fast-training.ipynb
├── 07-monitoring-hyperparameter-search-efficient-cnns.ipynb
├── 08-from-reversing-convolutions-to-vaes.ipynb
└── 09-adversarial-training-and-gans.ipynb
