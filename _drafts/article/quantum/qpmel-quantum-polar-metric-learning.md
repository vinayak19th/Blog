---
title: "QPMeL: Quantum Polar Metric Learning for NISQ Devices"
short_title: Quantum Polar Metric Learning
description: An exploration of Quantum Polar Metric Learning (QPMeL), a novel approach to metric learning on near-term quantum computers that leverages polar coordinate representations
author: vinayak
date: 2026-01-29 11:33:00 +0800
categories: [Article, Quantum Computing]
tags: [quantum computing, quantum machine learning, metric learning, NISQ, pennylane, research]
pin: false
math: true
comments: true
---
{% assign image_path = "/Blog/PostMedia/article/quantum/qpmel/" %}

<center markdown="block">

[![arXiv](https://img.shields.io/badge/Paper-Paper.svg?&logo=googledocs&style=for-the-badge&color=B31B1B&logoColor=white)](https://arxiv.org/abs/2312.01655){: .light}
[![arXiv](https://img.shields.io/badge/Paper-Paper.svg?&logo=googledocs&style=for-the-badge&color=ff6b6b)](https://arxiv.org/abs/2312.01655){: .dark}

</center>

> This work represents a significant step forward in applying quantum computing to practical machine learning tasks, specifically addressing the challenges of metric learning on NISQ devices.

# Introduction

As quantum computing continues to evolve, one of the most exciting frontiers lies in **Quantum Machine Learning (QML)** - the intersection of quantum computing and machine learning. However, practical solutions are severely constrained by the current limitations of quantum hardware, known as the **Noisy Intermediate-Scale Quantum (NISQ)** era. NISQ devices have limited qubit counts, short coherence times, and lack robust error correction.

My recent work on **Quantum Polar Metric Learning (QPMeL)** explores how we can leverage the unique properties of quantum systems to tackle a fundamental machine learning problem: **metric learning**, while sidestepping the pitfalls of training on NISQ devices.

## I. What is Metric Learning?

Metric learning is the task of learning a distance function or similarity measure between data points. Ideally, we want to learn an embedding space where similar items are close together and dissimilar items are far apart. This is crucial for applications like:

1. **Similarity search**: Finding similar images, documents, or products.
2. **Clustering**: Grouping similar data points.
3. **Face recognition**: Determining if two images show the same person.
4. **Few-Shot Learning**: Learning to classify new concepts with very few examples.

Traditional methods use neural networks to learn these embeddings. **QPMeL** brings this concept into the quantum realm, aiming to create encodings that are not just "quantum-ready" but "quantum-native"—designed to exploit the geometry of Hilbert space.

![Metric Learning Concept]({{image_path}}/Figure2.png){: width="800" }
_**Fig 1.** Overview of the QPMeL Framework_

## II. The Challenge with QML

To use quantum models on classical data (like images), we first need to **encode** the data into a quantum state.
- **Fixed Encodings** (like Angle Encoding) are simple but often lead to poor separability.
- **Trainable Encodings** (using Parameterized Quantum Circuits) can learn better representations but suffer from **Barren Plateaus**—where gradients vanish, making training impossible as the system scales.

QPMeL solves this by being **Quantum-Aware** but **Classically-Trained**.

## III. Quantum Polar Metric Learning (QPMeL)

The core idea of QPMeL is to define a **Unified Feature Space** composed of the surfaces of independent unit spheres in $\mathbb{R}^3$. This natural geometry matches the state space of single qubits (the Bloch Sphere!).

### The Unified Feature Space

Instead of mapping data to arbitrary vectors, QPMeL maps standard data points to **Polar Coordinates** $(\theta, \gamma)$.
- **$\theta$ (Polar Angle)**: Maps to $[0, \pi]$
- **$\gamma$ (Azimuthal Angle)**: Maps to $[-\pi, \pi]$

These coordinates directly correspond to the angles needed to prepare a qubit state using $R_y$ and $R_z$ gates.

$$
\ket{\psi} = R_z(\gamma) R_y(\theta) \ket{0}
$$

By training a classical neural network to output these specific angles, we create an embedding that is instantly translatable to a quantum state **without needing a quantum computer during training.**

![QPMeL Encoder]({{image_path}}/Figure3.png){: width="600" }
_**Fig 2.** The QPMeL Encoder architecture showing the projection to angular coordinates._

### Projective Metric Function (PMeF)

Standard distance metrics like Euclidean distance don't capture the true "distance" between quantum states (Fidelity). To train the classical encoder effectively, we introduced the **Projective Metric Function (PMeF)**.

PMeF allows us to calculate the **fidelity** (similarity) between two quantum states entirely classically, using only their angular coordinates $(\theta_1, \gamma_1)$ and $(\theta_2, \gamma_2)$.

$$
F(\ket{\psi_1}, \ket{\psi_2}) = |\braket{\psi_1}{\psi_2}|^2
$$

This function is differentiable, meaning we can use standard backpropagation to train our encoder to maximize the fidelity between similar class examples and minimize it for different classes.

![Training Pipeline]({{image_path}}/Figure5.png){: width="800" }
_**Fig 3.** The training pipeline using Prototypical Loss and PMeF._

## IV. Results

We tested QPMeL on standard benchmarks like **MNIST** and **Fashion-MNIST**, as well as complex **Few-Shot Learning** tasks.

### Classification Performance
QPMeL achieves state-of-the-art results compared to other QML approaches, often using significantly fewer qubits and gates.

![Results Table]({{image_path}}/Figure6.png){: width="800" }
_**Fig 4.** Comparison of QPMeL against other quantum and classical methods._

Key Highlights:
- **3X better separation** in multi-class settings compared to standard QMeL.
- **No Barren Plateaus**: Since training is purely classical, convergence is stable and fast.
- **Scalability**: We successfully scaled up to 10-class classification and 15-way few-shot learning, a first for this class of QML models.

## V. Conclusion

QPMeL demonstrates that we don't always need a quantum computer in the loop to learn good quantum representations. By respecting the **geometry of quantum states** (the Bloch sphere) within a classical training loop, we can generate powerful, efficient, and scalable quantum embeddings.

This "quantum-aware" approach could be the key to unlocking practical QML applications in the NISQ era.

<br>
<hr>

## Acknowledgments

This work was conducted at the **MPS-Lab at Arizona State University**. I'm grateful to my advisor and collaborators for their guidance and support throughout this research project.

## Further Reading

- [Read the full paper on arXiv](https://arxiv.org/abs/2312.01655)
- [Project Website](https://mpslab-asu.github.io/QPMeL/)
- [PennyLane QML](https://pennylane.ai/qml/)
