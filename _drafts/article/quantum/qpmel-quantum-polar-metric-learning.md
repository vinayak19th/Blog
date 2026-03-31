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

[![arXiv](https://img.shields.io/badge/Paper-Paper.svg?&logo=googledocs&style=for-the-badge&color=B31B1B&logoColor=white)](https://arxiv.org/abs/2312.01655)

## Abstract
*Quantum Machine Learning (QML) promises richer data representations and improved learning by leveraging the unique properties of quantum computation. A necessary first step in QML is to encode classical data into quantum states. Static encoding mechanisms offer limited expressivity, while training quantum model suffers from barren plateaus, making optimization unstable and computationally expensive. We propose Quantum Projective Metric Learning (QPMeL) - a quantum-aware, classically-trained framework to learn dense and high-quality quantum encodings. QPMeL maps classical data to the surface of independent unit spheres in $\mathbb{R}^3$, which naturally aligns with the state of multiple unentangled qubits. QPMeL also introduces a novel Projective Metric Function (PMeF) to approximate Hilbert space similarity in $\mathbb{R}^3$, along with a gradient stabilization trick that further enhances training efficiency. QPMeL achieves state-of-the-art performance on MNIST, Fashion-MNIST, and Omniglot, scaling up to 10-class classification and 15-way few-shot learning with high accuracy using significantly fewer qubits. It is the first QML approach to support multi-modal (image+text) learning, achieving over 90\% accuracy in the 15-way-1-shot setting with only 20 qubits.*


# Introduction
Quantum Machine Learning (QML) offers the potential for richer feature representations and faster learning, but practical applications are currently hindered by hardware limitations such as low qubit counts and short coherence times. A fundamental step in overcoming these barriers is effectively encoding classical data into quantum states, as the quality of this encoding directly impacts model performance. Previous approaches, including static encoding and trainable quantum circuits, have faced significant challenges: static methods often lack expressivity, while trainable circuits frequently suffer from "barren plateaus," making optimization unstable and computationally expensive.

To solve these issues, we propose Quantum Projective Metric Learning (QPMeL), a novel framework that produces quantum-aware, classically-trained data encodings. The core innovation of QPMeL is the creation of a unified feature space where classical data is mapped to the surfaces of independent unit spheres, naturally aligning with the geometry of unentangled qubits. By employing a unique Projective Metric Function (PMeF) and a gradient stabilization trick, the framework learns dense, separable embeddings purely through classical training, thereby avoiding the instability associated with optimizing quantum circuits directly.

The main contributions introduced by QPMeL can be summarized into:
1. A unified feature space consisting of independent spherical surfaces common to the classical and quantum domains created via a classical encoder which outputs angular encodings ($\theta,\gamma$).
2. A novel Projective Metric Function (PMeF), which computes the similarity between points in quantum state space using only the $\R^3$ coordinates derived from the angular encodings.
3. A gradient trick for PMeF leading to more stable gradients during training, allowing the models to converge more consistently.

<!-- QPMeL demonstrates superior efficiency and accuracy compared to existing methods, achieving state-of-the-art results on benchmarks like MNIST and Fashion-MNIST while using significantly fewer qubits. It scales effectively to 10-class classification tasks and 15-way few-shot learning . notably, it is the first QML approach to support multi-modal learning (integrating image and text), achieving over 90% accuracy in 15-way-1-shot settings with only 20 qubits. -->

## I. What is Metric Learning?

![Metric Learning Concept]({{image_path}}/metric_learning.webp){: width="450" }
_**Fig 1.** Simplified View of Metric Learning ([Source: GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/metric-learning/))_

Metric learning is the task of learning a distance function or similarity measure between data points. Ideally, we want to learn an embedding space where similar items are close together and dissimilar items are far apart. This is shown in Fig 1 where after training **data from similar classes** are **close together in feature space**.  *This is crucial for applications like:*

1. **Similarity search**: Finding similar images, documents, or products.
3. **Face recognition**: Determining if two images show the same person.
4. **Few-Shot Learning**: Learning to classify new concepts with very few examples.

Metric learning has been a promising field within **Quantum Machine Learning (QML)** to address the '*input-side bottleneck*' in quantum circuits. Previous papers[^QFSL] have also applied few shot learning in the QML context to demonstrate the capabilities of **Quantum Metric Learning (QMeL)**. However, there remain major challenges with previous implementations.

> Here are some additional resources on metric learning -  [Good Medium Article[1]](https://medium.com/aimonks/understanding-metric-learning-and-contrastive-learning-a-beginners-guide-680115baf3a4), [Lecture PDF from Columbia[2]](https://www.cs.columbia.edu/~verma/talks/metric_learning_tutorial_verma.pdf), [Towards Data Science post[3]](https://towardsdatascience.com/metric-learning-tips-n-tricks-2e4cfee6b75b/)
{: .prompt-info}

## II. The Landscape of Quantum Encodings

The single biggest obstacle in Quantum Machine Learning (QML) isn't the algorithms—it's how we get classical data (like images) into a quantum state. This is the **Encoding Challenge**.

As highlighted by [Havlíček et al. (2019)](https://www.nature.com/articles/s41586-019-0980-2), the choice of encoding defines the "vision" of your quantum model. On near-term NISQ hardware, we need encodings that are **qubit-efficient** yet **highly expressive**.

There are four major ways researchers tackle this today:

**1. Classical Compression + Static Encoding:**
Initially, researchers used techniques like PCA or Autoencoders to compress data and "stuffed" it into a quantum state using fixed gates (like $R_Y$). 
- **The Good:** Extremely simple and hardware-friendly.
- **The Bad:** Limited expressivity. It only captures a tiny sliver of the Hilbert space and fails to achieve class separability in the quantum state.
- **_Reference Papers_**: [Mari et al., 2020](https://quantum-journal.org/papers/q-2020-10-09-340/), [Havlíček et al. (2019)](https://www.nature.com/articles/s41586-019-0980-2)

**2. Classical Compression + Trainable Encoding:**
This evolved into **Quantum Metric Learning (QMeL)**, where compressed data is passed through a trainable quantum circuit.
- **The Good:** Specifically learns task-relevant features.
- **The Bad:** Requires deep circuits (vulnerable to noise) and suffers from training instability due to the non-Euclidean geometry of quantum states.
- **_Reference Papers_**: [Llyod et al. (2020)](https://arxiv.org/abs/2001.03622), [Thumwanit et al. (2021)](https://ieeexplore.ieee.org/document/9586190)

**3. The "Hybrid" Approach: Joint Training:**
The modern state-of-the-art involves training both the classical encoder and the quantum circuit together.
- **The Good:** High representation quality.
- **The Bad:** It's haunted by the **Barren Plateau**. As you add qubits, gradients vanish, making training painfully slow and convergence difficult.
- **_Reference Papers_**: [Hou et al., 2023](https://link.springer.com/article/10.1140/epjqt/s40507-023-00182-1), [Liu et al., 2021](https://ieeexplore.ieee.org/document/9951229), etc.

> **What is a Barren Plateau?** It's like trying to find a needle in an infinitely large, flat haystack. The gradient (the direction to go) becomes so small that the optimizer can't find its way.
{: .prompt-warning }

**4. Quantum-Aware Classical Training (QPMeL):**
This is where **QPMeL** changes the conversation. Instead of fighting quantum circuits during training, we train a classical model that is "aware" of quantum geometry.
- **The Innovation:** The model learns to output angular parameters $(\theta, \gamma)$ that are inherently separable in Hilbert space.
- **The Result:** Dense, expressive embeddings with **zero quantum training overhead** and only 2 gates per qubit.
- **_Reference Papers_**: [Sharma et al., 2025](https://arxiv.org/abs/2312.01655)

![Encoding Strategy Comparison]({{image_path}}/Figure2.png){: width="800" }
_**Fig 2.** Mapping the different approaches to Quantum Data Encoding._

## III. Quantum Polar Metric Learning (QPMeL)

The **Quantum Projective Metric Learning (QPMeL)** framework is built on a simple but powerful premise: we can learn quantum-native representations without actually needing a quantum computer during the training phase.

By creating a **Unified Feature Space**—a shared geometry that both classical and quantum systems understand—we can bridge the gap between classical data and quantum states.

### 1. The Unified Feature Space (The Bloch Sphere)

Instead of mapping data to arbitrary high-dimensional vectors, QPMeL maps every data point to the surface of independent unit spheres in $\mathbb{R}^3$. Why spheres? Because the state of a single qubit is perfectly represented as a point on a unit sphere, famously known as the **Bloch Sphere**.

While many QML models try to jump straight into complex multi-qubit entanglement, QPMeL focuses on making the most of individual qubits first. The encoder outputs two specific angles for each "feature" or qubit:
- **$\theta$ (Polar Angle)**: Ranges from $[0, \pi]$
- **$\gamma$ (Azimuthal Angle)**: Ranges from $[-\pi, \pi]$

These **Angular Coordinates** are all we need to perfectly describe a quantum state. This coordinate system is extremely lightweight. In a quantum circuit, these angles directly parameterize two simple rotation gates:

$$
\ket{\psi} = R_z(\gamma) R_y(\theta) \ket{0}
$$

> **Why this matters for NISQ:** Most current quantum computers (NISQ devices) are limited by "noise" and "depth." By only requiring 2 gates per qubit ($R_Y$ and $R_Z$), QPMeL is incredibly hardware-efficient. It produces "Quantum-Native" embeddings that are ready to run on any device with zero extra processing.
{: .prompt-tip }

### 2. The QPMeL Encoder (Angle Projection)

To achieve this, we don't start from scratch. We take a standard, powerful classical neural network (like a ResNet or a Transformer) and add a specialized **Angle Projection layer** at the end.

This layer consists of two parallel dense branches. Instead of outputting a generic feature vector, these branches output the $\vec{\theta}$ and $\vec{\gamma}$ values. To ensure they stay within the valid bounds of the Bloch sphere, we use a custom scaling function (or specific activations like Tanh/Sigmoid followed by scaling) to map the real-valued outputs into our rotation ranges.

![QPMeL Encoder]({{image_path}}/Figure3.png){: width="600" }
_**Fig 3.** The QPMeL Encoder architecture showing how high-level classical features are projected into angular coordinates._

### 3. Training with Awareness (PMeF)

If we're training classically, how does the model "know" it’s learning a quantum state? This is where the **Projective Metric Function (PMeF)** comes in. 

Standard distance metrics like Euclidean distance are "flat"—they don't understand the curved, periodic geometry of quantum fidelity. In the quantum world, similarity is measured by **Fidelity** ($F = |\braket{\psi_1}{\psi_2}|^2$). 

PMeF allows us to calculate this exact quantum fidelity **entirely in the classical domain**, using only the angular coordinates $(\theta_1, \gamma_1)$ and $(\theta_2, \gamma_2)$ produced by our encoder. This has two massive benefits:
1.  **Zero Quantum Overhead**: We can train the entire model on a standard GPU using traditional backpropagation.
2.  **No Barren Plateaus**: Since we aren't optimizing through a stochastic quantum circuit, we avoid the "gradient vanishing" problem that kills most QML training.

Once trained, the encoder has effectively learned a "Quantum-Aware" representation. It knows exactly how to place data points on the Bloch sphere so that they are perfectly separable when viewed through a quantum lens.

![Training Pipeline]({{image_path}}/Figure5.png){: width="800" }
_**Fig 4.** High-level overview of the training pipeline using PMeF to simulate quantum similarity classically._

## IV. Results & Benchmarks

The real proof of **QPMeL** lies in its performance across diverse and challenging datasets. We benchmarked the framework against several state-of-the-art quantum and classical models, focusing on both standard classification and the much harder world of **Few-Shot Learning**.

### 1. Classification Performance
On standard benchmarks like **MNIST** and **Fashion-MNIST**, QPMeL achieves state-of-the-art results compared to other QML architectures. What’s truly remarkable is that it does so while using significantly fewer resources:
- **Qubit Efficiency**: High accuracy with up to 10-class classification using only 10-20 qubits.
- **Circuit Depth**: Only 2 gates per qubit ($R_Y, R_Z$), minimizing noise and errors on hardware.
- **Fast Convergence**: Since training is purely classical, QPMeL avoids the "Barren Plateau" entirely, reaching its peaks in minutes rather than hours.

![Results Table]({{image_path}}/Figure6.png){: width="800" }
_**Fig 5.** Quantitative comparison showing QPMeL's accuracy across different datasets._

### 2. Pushing the Boundaries: Few-Shot & Multi-Modal Learning

Most QML models struggle as you increase the number of classes. QPMeL, however, scales exceptionally well. One of the most exciting breakthroughs was its performance in **15-way few-shot learning**—a setting where the model sees only 1 example of a new class and must correctly identify it among 15 candidates.

- **Omniglot Challenge**: QPMeL successfully learned to classify complex handwritten characters from the Omniglot dataset with high precision.
- **15-Way 1-Shot**: It achieved over **90% accuracy** in this challenging setting, a first for this class of QML models.
- **Multi-Modal Learning**: For the first time in a QML framework, we demonstrated **image+text learning**. By mapping both modalities to our "Unified Feature Space," we proved that quantum embeddings can handle more than just simple pixels.

> **Research Spotlight:** QPMeL is the first quantum metric learning approach to support multi-modal learning, achieving over **90% accuracy** in complex 15-way-1-shot settings with just **20 qubits**.
{: .prompt-info }

---

## V. Looking Ahead: The Future of Quantum-Aware Learning

The success of **QPMeL** demonstrates a shift in how we should think about near-term QML. We don't always need to fight with noisy, unstable quantum circuits during the optimization process. By respecting the **geometry of quantum states** (like the Bloch sphere) within a classical training loop, we can pre-train powerful quantum-native representations.

### Key Takeaways:
- **Stable Training**: Purely classical optimization means no barren plateaus and faster experimentation.
- **Native Hardware Support**: Your trained models are "deployment-ready" for NISQ devices with zero overhead.
- **Scalable QML**: From multi-class classification to multi-modal data, QPMeL bridges the gap between today's hardware and tomorrow's complex tasks.

This "quantum-aware" philosophy could be the key to unlocking the power of the first generation of quantum computers. As we move closer to a hybrid classical-quantum future, methods like QPMeL will be essential tools in any machine learning researcher’s toolkit.

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

