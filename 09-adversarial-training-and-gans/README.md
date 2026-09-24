# Adversarial Training & Generative Adversarial Networks

Implementation and exploration of adversarial examples, adversarial training, and Generative Adversarial Networks (GANs) using PyTorch.

## Overview

This project explores how neural networks can be fooled by carefully constructed adversarial inputs and how similar ideas can be used to generate synthetic data.

The notebook first investigates adversarial examples using gradient-based attacks. It then develops a discriminator that distinguishes real and generated samples and finally implements the main components of a Generative Adversarial Network.

## Topics Covered

- Adversarial Examples
- Adversarial Attacks
- Projected Gradient Ascent
- Gradient-based Input Perturbations
- Adversarial Training
- Real/Fake Classification
- Discriminators
- Synthetic Data Generation
- Generative Adversarial Networks
- Generator Networks
- Min-Max Optimisation
- Binary Cross-Entropy
- MNIST
- PyTorch

## Adversarial Examples

The project demonstrates how a trained neural network can be fooled by modifying its input rather than its parameters.

The adversarial attack maximises the classification loss with respect to the input while constraining the perturbation to remain within an epsilon-ball around the original image.

This results in an adversarial example that is visually similar to the original input but can lead to a different model prediction.

## Projected Gradient Ascent

A gradient-based adversarial attack is implemented using iterative projected gradient ascent.

At each step:

1. The gradient of the loss with respect to the input is computed.
2. The input is moved in the direction that increases the loss.
3. The perturbation is projected back into the allowed epsilon range.
4. The process is repeated for a fixed number of steps.

The implementation is evaluated on the MNIST classification task.

## Fooling a Discriminator

The project then investigates a discriminator that classifies MNIST inputs as either real or fake.

Instead of using only random noise as fake data, adversarial examples are generated to make fake samples progressively more similar to real data from the perspective of the discriminator.

This creates an iterative training process in which the fake examples are improved using adversarial optimisation.

## Real/Fake Dataset

A custom dataset combines:

- Real MNIST samples
- Generated fake samples

The fake samples are initially generated from random values with statistics related to the real dataset.

They are subsequently updated using adversarial attacks against the discriminator.

## Adversarial Training

The discriminator is trained while its fake examples are periodically updated.

The process alternates between:

1. Training the discriminator.
2. Generating improved fake examples.
3. Using the new fake examples for subsequent discriminator training.

This demonstrates how adversarial optimisation can be used to generate increasingly realistic synthetic inputs.

## Generative Adversarial Networks

The project then introduces Generative Adversarial Networks (GANs).

A GAN consists of two neural networks:

### Generator

The generator maps samples from a latent distribution to the image domain.

The latent vector provides a compact representation from which the generator creates synthetic MNIST images.

### Discriminator

The discriminator receives both real and generated images and predicts whether each input is real or fake.

The generator therefore attempts to create samples that the discriminator classifies as real.

## Min-Max Game

GAN training can be formulated as a min-max optimisation problem:

$$
\max_D \min_G
\mathbb{E}[\log D(X)] +
\mathbb{E}[\log(1-D(G(Z)))].
$$

The discriminator attempts to distinguish real samples from generated samples, while the generator attempts to produce samples that fool the discriminator.

The training procedure alternates between updating the discriminator and the generator.

## Generator and Discriminator Training

The implementation separates the two optimisation objectives.

During discriminator updates:

- Real samples should be classified as real.
- Generated samples should be classified as fake.
- Gradients are prevented from updating the generator.

During generator updates:

- Generated samples are passed through the discriminator.
- The generator is optimised so that generated samples are classified as real.

Binary cross-entropy with logits is used as the classification objective.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- MNIST
- Jupyter Notebook

## Academic Context

Developed as part of the Master's program in Artificial Intelligence at Johannes Kepler University Linz (JKU).

## Notebook

`09-adversarial-training-and-gans.ipynb`
