---
title: "Analyzing barren plateaus in Quantum Machine Learning"
short_title: Analyzing barren plateaus
description: A simplified introduction to the analysis of barren plateaus in QML.
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

> This blog post tries to provide a simpler, less mathematically complex but still accurate intuition behind the occurrence of barren plateaus in quantum neural networks. The ideas here are from the set of papers listed in the [references](#references).

# 1. Introduction (TL;DR)
At the heart of Quantum Machine Learning (QML) lies a contradictory relationship between the expressive power of quantum neural networks and their trainability. While there is more nuance to it, the main idea is that as we increase the expressivity of a Quantum Circuit via the number of qubits and/or entangling gates, the training landscape flattens out exponentially, or we get barren plateaus which makes the circuit impossible to train. 

To better understand this phenomenon, we first need to define 2 things - 
1. How do we measure the expressivity of a Quantum Circuit?
2. How do we identify if a given circuit will lead to barren plateaus?

My goal with this blog post is to simply both ideas to make them a little more approachable and intuitive. 

# 2. Background
## 2.1. The anatomy of a QML circuit
While most modern QML algorithms have adopted Data ReUploading (DR)[^dr_models] as a core part of their architecture, for the purposes of this post we will stick to the simpler structure of encoding data once followed by a set of trainable parameters. However, everything outlined in this post also applied to DR models since we treat the data encoding gates as essential constant functions.

![VQC Circuit]({{image_path}}/VQC_light.svg){: .light }{: width="550" }
![VQC Circuit]({{image_path}}/VQC_dark.svg){: .dark }{: width="550" }
_**Fig 1.**  A general overview of a Variational Quantum Circuit (VQC)_ 

What Fig 1 shows is that generally, a VQC can be thought of as being made up of 3 components-
1. Data Encoding Layer - The quantum embedding (more details in this [pennylane article](https://pennylane.ai/qml/glossary/quantum_embedding)[^embeddings]) which encodes classical data into the quantum state amplitudes. 
2. Trainable Parameterized Layer - The part of the circuit that holds all the trainable parameters, usually composed of single and multi qubit rotations. 
3. Measurement Layer - The measurement layer, where the final measurement is performed to obtain the output classical data. This is represented by the $\mathcal{M}$ in Fig 1.

## 2.2. Statistical Measurements of Quantum State
Recall that every quantum state is essentially a discrete probability distribution over the basis states (ex. $\ket{000},\ket{100}$ in the case of 3 qubits).  This means 

In most practical models the measured quantity is the expectation value of the state w.r.t to some quantum observable $\hat{O}$. This is given by the following formula-

$$ \bra{\psi} \hat{O} \ket{\psi} $$

where $\ket{\psi}$ is the final quantum state before measurement and $\hat{O}$ is the measurement operator represented by a [Hermitian Matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)[^hermitian_matrix] (Any matrix $H$ such that $H=H^{\dagger}$). **This simplified structure forms the basis of our analysis**.

# 3. Expressivity of Quantum Circuits
This section outlines the formal definitions of expressivity and how we can quantify this otherwise abstract concept.

## 3.1. Understanding the 2-Design
An important question to ask in the context of QML is what makes a quantum circuit more expressive than another. Formally this idea is studied using a tool called a **2-design** as defined below. 

> **Definition 1 (2-design).** A unitary 2-design is an ensemble of quantum operations (or a parameterized circuit) whose random distribution perfectly mimics the uniform Haar measure[^pennylane_haar_measure] over all possible unitaries up to the second statistical moment.
{: .prompt-definition }

### 3.1.1. Simplifying and Visualizing Expressivity
To simply this formal but mathematically dense definition , a *2-design* is a parameterized quantum circuit (represented as a single unitary $U(\theta)$) in which uniformly sampling the parameter $\theta$ mimics uniformly sampling from the $SU(N)$ group (the set of all $N\times N$ unitary matrices). This means that **any trainable circuit that represents a true 2-design can implement any other unitary given some parameterization of $\theta$**.  

Look at Fig 2, which of the two circuit is closer to a 2-Design between $U_0$ and $U_1$?  (<a data-bs-toggle="collapse" href="#circuit_two_design_collapse" aria-expanded="false" aria-controls="circuit_two_design_collapse">Show Answer</a>)

<div class="collapse" id="circuit_two_design_collapse" markdown=1>
> As we can clearly see in Fig 2, $U_0$ is closer to a 2-Design since the set of accessible unitaries it generates is larger than that of $U_1$. However, even $U_0$ is not a perfect 2-Design.
{: .prompt-tip }
</div>

![Expressive Circuit]({{image_path}}/expressive_light.svg){: .light }{: width="550" }
![Expressive Circuit]({{image_path}}/expressive_dark.svg){: .dark }{: width="550" }
_**Fig 2.**  Visually understanding expressivity.[Based on Fig 1 from Holmes et al., 2021[^holmes_et_al]]_

Essentially Fig 2 allows us to visualize the how much of the $SU(N)$ group can the [span](https://understandinglinearalgebra.org/sec-span.html) of a given parameterized unitary $U(\theta)$ cover. When trying to analyze highly expressive unitaries, we specifically compare them to the *[Haar Measure](https://en.wikipedia.org/wiki/Haar_measure)*[^wiki_haar_measure].  

### 3.1.2. Breaking down the Haar Measure
To properly understand the Haar Measure, we must first breakdown the concept of a *[measure](https://en.wikipedia.org/wiki/Measure_(mathematics)*. Pennylane has a great blog post on the topic[^pennylane_haar_measure], which I'll try and simplify.

#### What is a Measure?
> **Definition 2 (Measure).** A measure is a formal integration weight that counteracts coordinate distortion, ensuring that the integral of a function over a curved space remains independent of how that space is parameterized.
{: .prompt-definition }

At its core, a measure is a mathematical tool that describes how "stuff" (like length, area, or volume) is distributed across a space. You can think of it as a weighting system or a correction factor that you must use whenever the geometry of a space isn't perfectly flat and grid-like. You can see this in Fig 3, where as we get closer to the poles and closer to the center, the area of the boxes formed by $d\phi$ and $d\theta$ gets smaller. 

![Sphere Volume]({{image_path}}/VolumeCalc.png){: .light }{: width="350" }
![Sphere Volume]({{image_path}}/VolumeCalc_dark.png){: .dark }{: width="350" }
_**Fig 3.** Computing Spherical Volume via integration. [Source: Pennylane Tutorial on the Haar Measure[^pennylane_haar_measure]]_

**The Problem: Warping the Space:** Imagine you want to pick a completely random, uniform point on the surface of the Earth. Your first instinct might be to tell a computer to pick a random latitude (up/down) and a random longitude (left/right). However, lines of longitude **are very far apart at the equator**, but they **squeeze tightly together to a single point at the poles**.

If you just pick random coordinate numbers, your generated points will clump heavily at the poles and be sparse at the equator. You are sampling uniformly in numbers (coordinates), but you are not sampling uniformly in actual physical space. This is because the coordinates are distorting the notion of "uniformity".

The concept of the measure is a correction applied to address this issue. We need to utilize this idea in 2 very specific cases during this analysis: 

1. **During Integration** : The measure tells you exactly how much extra "weight" to give the equator compared to the poles when adding up the total volume or area.
2. **For Sampling**: The measure gives you the exact blueprint for how to skew your random number generator so that every physical square inch of the sphere has a perfectly equal chance of being picked.

**Can you formulate a rough mental picture for how this relates to Quantum States?** (<a data-bs-toggle="collapse" href="#measure_in_QML" aria-expanded="false" aria-controls="measure_in_QML">Show Answer</a>)

<div class="collapse" id="measure_in_QML" markdown=1>
> **Measures in QML**: Just like points on a 3D sphere, quantum operations (unitary matrices) live in a complex, high-dimensional, curved mathematical space called the Unitary Group. If you just randomly guess angles for the gates in your Parameterized Quantum Circuit (PQC), the resulting quantum states will "clump up" in certain mathematical corners of the Hilbert space, just like the points clumped at the poles of our Earth example.
{: .prompt-definition }
</div>

#### Introducing the Haar Measure
As highlighted <a data-bs-toggle="collapse" href="#measure_in_QML" aria-expanded="false" aria-controls="measure_in_QML">here</a>, if we uniformly sample parameters for our unitaries (like those in Fig 2) from the rotational gate range (ex. $[-\pi,\pi]$), the resulting quantum states we produced would be highly concentrated into specific sub-regions of the quantum Hilbert space. 


# References
[^dr_models]: [Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085)
[^embeddings]: [Pennylane Quantum Embeddings](https://pennylane.ai/qml/glossary/quantum_embedding)
[^barren_plateaus_paper]: [McClean et al](https://arxiv.org/abs/1803.11173)
[^wiki_haar_measure]: [Haar Measure Wikipedia](https://en.wikipedia.org/wiki/Haar_measure)
[^pennylane_haar_measure]: [Pennylane Tutorial on the Haar Measure](https://pennylane.ai/qml/demos/tutorial_haar_measure)
[^hermitian_matrix]: [Hermitian Matrix](https://en.wikipedia.org/wiki/Hermitian_matrix)
[^holmes_et_al]: [Connecting ansatz expressibility to gradient magnitudes and barren plateaus](https://arxiv.org/pdf/2101.02138)\