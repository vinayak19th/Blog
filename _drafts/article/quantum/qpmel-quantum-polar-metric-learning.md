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

> This work represents a significant step forward in applying quantum computing to practical machine learning tasks, specifically addressing the challenges of metric learning on NISQ devices.

<center markdown="block">

[![arXiv](https://img.shields.io/badge/Paper-Paper.svg?&logo=googledocs&style=for-the-badge&color=B31B1B&logoColor=white)](https://arxiv.org/abs/2312.01655){: .light}
[![arXiv](https://img.shields.io/badge/Paper-Paper.svg?&logo=googledocs&style=for-the-badge&color=ff6b6b)](https://arxiv.org/abs/2312.01655){: .dark}

</center>

## Abstract
Quantum Machine Learning (QML) promises richer data representations and improved learning by leveraging the unique properties of quantum computation. A necessary first step in QML is to encode classical data into quantum states. Static encoding mechanisms offer limited expressivity, while training quantum model suffers from barren plateaus, making optimization unstable and computationally expensive. We propose Quantum Projective Metric Learning (QPMeL) - a quantum-aware, classically-trained framework to learn dense and high-quality quantum encodings. QPMeL maps classical data to the surface of independent unit spheres in $\mathbb{R}^3$, which naturally aligns with the state of multiple unentangled qubits. QPMeL also introduces a novel Projective Metric Function (PMeF) to approximate Hilbert space similarity in $\mathbb{R}^3$, along with a gradient stabilization trick that further enhances training efficiency. QPMeL achieves state-of-the-art performance on MNIST, Fashion-MNIST, and Omniglot, scaling up to 10-class classification and 15-way few-shot learning with high accuracy using significantly fewer qubits. It is the first QML approach to support multi-modal (image+text) learning, achieving over 90\% accuracy in the 15-way-1-shot setting with only 20 qubits.
# Introduction

Quantum Machine Learning (QML) offers the potential for richer feature representations and faster learning, but practical applications are currently hindered by hardware limitations such as low qubit counts and short coherence times. A fundamental step in overcoming these barriers is effectively encoding classical data into quantum states, as the quality of this encoding directly impacts model performance. Previous approaches, including static encoding and trainable quantum circuits, have faced significant challenges: static methods often lack expressivity, while trainable circuits frequently suffer from "barren plateaus," making optimization unstable and computationally expensive.

To solve these issues, the authors propose Quantum Projective Metric Learning (QPMeL), a novel framework that produces quantum-aware, classically-trained data encodings. The core innovation of QPMeL is the creation of a unified feature space where classical data is mapped to the surfaces of independent unit spheres, naturally aligning with the geometry of unentangled qubits. By employing a unique Projective Metric Function (PMeF) and a gradient stabilization trick, the framework learns dense, separable embeddings purely through classical training, thereby avoiding the instability associated with optimizing quantum circuits directly.

The main contributions introduced by QPMeL can be summarized into:
1. A unified feature space consisting of independent spherical surfaces common to the classical and quantum domains created via a classical encoder which outputs angular encodings ($\theta,\gamma$).
2. A novel Projective Metric Function (PMeF) to approximate Hilbert space similarity in $\mathbb{R}^3$.
3. A gradient trick for PMeF leading to more stable gradients during training, allowing the models to converge more consistently.

QPMeL demonstrates superior efficiency and accuracy compared to existing methods, achieving state-of-the-art results on benchmarks like MNIST and Fashion-MNIST while using significantly fewer qubits. It scales effectively to 10-class classification tasks and 15-way few-shot learning . notably, it is the first QML approach to support multi-modal learning (integrating image and text), achieving over 90% accuracy in 15-way-1-shot settings with only 20 qubits.

## I. What is Metric Learning?

![Metric Learning Concept]({{image_path}}/metric_learning.webp){: width="450" }
_**Fig 1.** Simplified View of Metric Learning ([Source: GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/metric-learning/))_

Metric learning is the task of learning a distance function or similarity measure between data points. Ideally, we want to learn an embedding space where similar items are close together and dissimilar items are far apart. This is shown in Fig 1 where after training **data from similar classes** are **close together in feature space**.  *This is crucial for applications like:*

1. **Similarity search**: Finding similar images, documents, or products.
3. **Face recognition**: Determining if two images show the same person.
4. **Few-Shot Learning**: Learning to classify new concepts with very few examples.

Metric learning has been a promising field within **Quantum Machine Learning (QML)** to address the '*input-side bottleneck*' in quantum circuits. Previous papers[^QFSL] have also applied few shot learning in the QML context to demonstrate the capabilities of **Quantum Metric Learning (QMeL)**. However, there remain major challenges with previous implementations.

## II. The Challenge with QML

To use quantum models on classical data (like images), we first need to **encode** the data into a quantum state.
- **Fixed Encodings** (like Angle Encoding) are simple but often lead to poor separability.
- **Trainable Encodings** (using Parameterized Quantum Circuits) can learn better representations but suffer from **Barren Plateaus**—where gradients vanish, making training impossible as the system scales.

QPMeL solves this by being **Quantum-Aware** but **Classically-Trained**.

![Metric Learning Concept]({{image_path}}/Figure2.png){: width="800" }
_**Fig 1.** Overview of the QPMeL Framework_

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

> **Definition 1. [Quantum Polar Metric Learning for NISQ]** : A framework that maps classical data to the polar coordinates of a quantum state (Bloch sphere surface), enabling the learning of quantum-native embeddings using purely classical training.
{: .prompt-definition }

> **Theorem 1. [Faithfulness of PMeF]** : The Projective Metric Function (PMeF) computed classically using angular coordinates is mathematically equivalent to the Fidelity metric between the corresponding quantum states.
{: .prompt-theorem }

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

# References
[^QFSL]: [Z. Huang, J. Shi and X. Li, 'Quantum Few-Shot Image Classification,' in IEEE Transactions on Cybernetics, vol. 55, no. 1, pp. 194-206, Jan. 2025, doi: 10.1109/TCYB.2024.3476339.](https://ieeexplore.ieee.org/document/10735395)

## Acknowledgments

This work was conducted at the **MPS-Lab at Arizona State University**. I'm grateful to my advisor and collaborators for their guidance and support throughout this research project.

## Further Reading

- [Read the full paper on arXiv](https://arxiv.org/abs/2312.01655)
- [Project Website](https://mpslab-asu.github.io/QPMeL/)
- [PennyLane QML](https://pennylane.ai/qml/)

