---
title: "Barren plateaus in Quantum Machine Learning"
short_title: Barren plateaus
description: Exploring the phenomenon of barren plateaus in quantum neural network training landscapes
author: vinayak
date: 2026-05-13 03:00:00 +0800
categories: [Article, Quantum Computing]
tags: [quantum computing, quantum machine learning, barren plateaus, paper breakdown]
pin: false
math: true
comments: true
plotly: true
---
{% assign image_path = "/Blog/PostMedia/article/quantum/barren-plateaus/" %}

> This blog post tries to provide a simpler, less mathematically complex but still accurate intuition behind the occourance of barren plateaus in quantum neural networks. The ideas here are from the set of papers listed in the [references](#references).

# Introduction (TL;DR)
At the heart of Quantum Machine Learning (QML) lies a contradictory relationship between the expressive power of quantum neural networks and their trainability. While there is more nuance to it, the main idea is that as we increase the expressivity of a Quantum Circuit via the number of qubits and/or entangling gates, the training landscape flattens out exponentially, or we get barren plateaus which makes the circuit impossible to train. 

To better understand this phenomenon, we first need to define 2 things - 
1. How do we measure the expressivity of a Quantum Circuit?
2. How do we identify if a given circuit will lead to barren plateaus?

My goal with this blog post is to simply both ideas to make them a little more approachable and intuitive. 

# Background
## The anatomy of a QML circuit
While most modern QML algorithms have adopted Data ReUploading (DR)[^dr_models] as a core part of their architecture, for the purposes of this post we will stick to the simpler structure of encoding data once followed by a set of trainable parameters. However, everything outlined in this post also applied to DR models since we treat the data encoding gates as essential constant functions.

![VQC Circuit]({{image_path}}/VQC_light.svg){: .light }{: width="550" }
![VQC Circuit]({{image_path}}/VQC_dark.svg){: .dark }{: width="550" }
_**Fig 1.**  A general overview of a Variational Quantum Circuit (VQC)_ 

What Fig. 1 shows is that generally, a VQC can be thought of as being made up of 3 components-
1. Data Encoding Layer - The quantum embedding (more details in this [pennylane article](https://pennylane.ai/qml/glossary/quantum_embedding)[^embeddings]) which encodes classical data into the quantum state amplitudes. 
2. Trainable Parameterized Layer - The part of the circuit that holds all the trainable parameters, usually composed of single and multi qubit rotations. 
3. Measurement Layer - The measurement layer, where the final measurement is performed to obtain the output classical data. This is represented by the $\mathcal{M}$ in Fig 1.

## Statistical Measurements of Quantum State
Recall that every quantum state is essentially a discrete probability distribution over the basis states (ex. $\ket{000},\ket{100}$ in the case of 3 qubits).  This means 

In most practical models the measured quantity is the expectation value of the state w.r.t to some quantum observable $\hat{O}$. This is given by the following formula-

$$ \bra{\psi} \hat{O} \ket{\psi} $$

where $\ket{\psi}$ is the final quantum state before measurement and $\hat{O}$ is the measurement operator represented by a [Hermitian Matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)[^hermitian_matrix] (Any matrix $H$ such that $H=H^{\dagger}$). **This simplified structure forms the basis of our analysis**.

# Expressivity of Quantum Circuits


# References
[^dr_models]: [Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085)
[^embeddings]: [Pennylane Quantum Embeddings](https://pennylane.ai/qml/glossary/quantum_embedding)
[^barren_plateaus_paper]: [McClean et al](https://arxiv.org/abs/1803.11173)
[^pennylane_haar_measure]: [Pennylane Tutorial on the Haar Measure](https://pennylane.ai/qml/demos/tutorial_haar_measure)
[^hermitian_matrix]: [Hermitian Matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)
