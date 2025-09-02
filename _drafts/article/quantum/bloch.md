---
title: Bloch Sphere Demystified - Hitchhikers Guide to Quantum States
description: A simplified explanation of the Bloch sphere representation of quantum states
author: Vinayak
date: 2025-08-16 11:33:00 +0800
categories: [Article, Quantum Computing]
tags: [quantum computing, quantum machine learning]
pin: false
math: true
comments: true
---

{% assign parts = page.path | split: '/' %}
{% assign image_path = site.post_image_path | append: '/' | append: parts[1] | append: '/' | append: parts[2] | append: '/' | append: "bloch" %}
This article aims to provide a simplified and more intuitive explanation of the Bloch sphere assuming that the reader has a basic understanding of quantum computing such as what is a [qubit](https://www.ibm.com/think/topics/qubit)[^qubit], what are basis states and a basic understanding of high school geometry. The Bloch sphere at its core the projective space of all complex numbers $\mathbb{C}^2$ which is also the true state space of a single qubit. This is well explained in [this video by Gabriele Carcassi](https://www.youtube.com/watch?v=KEzmw6cKOlU)[^bloch_video]. I'll try to explain it in a more intuitive way in this blog post.

## I. Defining the Bloch Sphere
According to [wikipedia](https://en.wikipedia.org/wiki/Bloch_sphere)[^wiki_bloch], we can define the bloch sphere as follows:

<div class="definition">
<b>Definition 1</b> (Bloch Sphere). In quantum mechanics and computing, the Bloch sphere is a geometrical representation of the pure state space of a two-level quantum mechanical system (qubit), named after the physicist Felix Bloch.
</div>

While this definition is technically accurate, it may not be very intuitive for someone new to the concept. Let's break it down and define some of our terms:

1. **Two-level quantum mechanical system**: A two-level quantum mechanical system is simply a system that can exist in two distinct states such as spin the up and spin down states on an electron. Generally, these systems are used to create qubits in quantum computing and their states are labeled as $\ket{0}$ and $\ket{1}$.
2. **Pure state**: A pure state is the most basic form of quantum state that can be fully described as a linear combination of basis states. For example, the states '$\alpha\ket{0} + \beta \ket{1}$' is a pure state as it is written purely as a superposition of the basis states.

Armed with these definitions, we can now understand __what the Bloch Sphere is trying to represent__. The Bloch sphere is a way to visualize the state of a single qubit (a two-level quantum system). However, before we can delve deeper into the true nature of the Bloch Sphere, we need to understand the concept of geometric representation.

## II. Geometric Representations
In really simple terms, a geometric representation translates abstract information into a spatial, visual format, making it easier to understand relationships, patterns, and structures. For example, we can represent real numbers on a number line, which is a one-dimensional geometric space. Similarly, we can represent complex numbers on a two-dimensional plane called the complex plane.

![Complex to plane]({{image_path}}/Complex.svg){: .light }{: width="550" }
![Complex to plane]({{image_path}}/ComplexDark.svg){: .dark }{: width="550" }

> "Geometric representations translate abstract information into a spatial, visual format, making it easier to understand relationships, patterns, and structures."

As the figure shows, we can take the general equation of a complex number defined below and plot them on a 2D plane. 
 
$$
  z = x + iy \in \mathbb{C}  \forall x,y \in \mathbb{R}
$$

In the example, **the key idea is that every single unique assignment of $(x,y)$ corresponds to a unique point on the 2D plane**. More formally, a 

<div class="definition">
<b>Definition 2</b> (Geometric Representation). A geometric representation is mapping from some abstract data space $\mathcal{D}$ to a mathematical space $\mathcal{S}$. 
</div>

Where a [mathematical space](https://en.wikipedia.org/wiki/Space_(mathematics)) is a set with some added structure such as a metric or topology[^math_space]. A [vector space](https://en.wikipedia.org/wiki/Vector_space) is an example of a mathematical space where the elements of the set are vectors and the added structure is vector addition and scalar multiplication.

**Why do we care?** This is important because, as stated earlier, the Bloch sphere is a specific type of '*space*' known as a [*'Projective*'](https://en.wikipedia.org/wiki/Complex_projective_space) space of all complex numbers $\mathbb{C}^2$. 

## III. Projective Spaces
A projective space aims to capture the idea of direction without magnitude. To better understand this idea let's define a new coordinate system in terms of $\theta,\lambda$, where $\theta$ is the angle from the 

![Projective]({{image_path}}/Projective.svg){: .light }{: width="400" }
![Projective]({{image_path}}/ProjectiveDark.svg){: .dark }{: width="400" }

As we can see in the picture above, we can represent any vector in the 2D plane $\ket{v} \in \mathbb{R}^2$ in terms of a point on the unit circle defined by the angle $\theta$ and a scaling factor $\lambda$. Formally, the semi-circle is termed the *real projective line* and is denoted as $\mathbb{RP}^1$.

$$
  \ket{v} = \lambda(\cos\theta, \sin\theta) \;\; \forall \;\; \theta \in [0,\pi], \lambda \in \mathbb{R}
$$

This idea allows us to define the notion of a *projective ray* - 

<div class="definition">
<b>Definition 3</b> (Projective Ray). A projective ray is a line that extends infinitely in the positive and negative direction from the origin and is defined by the angle it makes with an axis. All points on the same ray are considered equivalent in projective space. They can be defined using the equation $\ket{r} = \lambda(\cos\theta, \sin\theta)$ where $\theta$ is the angle and $\lambda \in \{-\infty,\infty\}$.

As the scaling factor is infinite, the only important factor is the angle $\theta$, thus we can represent a projective ray using just the angle $\theta$ as $\ket{r} = (\cos\theta, \sin\theta)$
</div>

The significance of a ray is that **all points on the same ray are considered equivalent in projective space** or more simply, we don't care about the magnitude of a vector in projective space.

### Real Projective Line
The interactive plot below shows how we create the '*real projective line*' from the 2D plane. Try moving changing the angle of the '*rotating vector*' and see how it affects the '*projective shadow*' as well as the point on the real projective line or the '*Projected Vector*'.

<div class="plotly-figure light">
    {% include plotly/bloch/projective_circle.html %}
</div>
<div class="plotly-figure dark">
    {% include plotly/bloch/projective_circle_dark.html %}
</div>

> Move the angle of the rotating vector and see how it affects the projected vector on the real projective line.
{: .prompt-info }

In this plot the coordinates on the real projective line $(p,\theta)$ represent the intercept point of $y=1$ and the angle of the rotating vector respectively. As evident from the figure, all points in $\mathbb{R}^2$ that are on the same ray from the origin (i.e. have the same angle $\theta$) map to the same point on the real projective line. However what may not be entire obvious is why we require only the semi-circle and not the entire circle. Recall that every point in $\mathbb{R}^2$ can be represented as a point on the unit circle with a scaling factor $\lambda$. This means that all points in the 3rd and 4th quadrants can be represented by a point in the 1st and 2nd quadrants with a negative scaling factor. Thus, **we only require half the unit circle to represent all points in $\mathbb{R}^2$** Additionally, notice how after we complete a rotation from $0$ to $\pi$, the projected shadow starts to move in reverse, this is further evidence that the lower half of the semi-circle is not needed to describe the entire space.

## IV. Quantum States and Projective Spaces
Let us think about an arbitary vector ($\ket{v}$) in 2D Complex Hilbert Space ($\mathbb{C}^2$). We can also represent this vector in terms of its basis vectors $\ket{0}$ and $\ket{1}$ as follows:

$$
   \ket{v} = x\ket{0} + y \ket{1} \;\; \forall \;\; x,y \in \mathbb{C}
$$

We can apply the same logic as before and every possible $\ket{v}$ in terms of a point on the a unit circle in $\mathbb{C}^2$ and a scaling factor $\lambda$ as follows:

$$
  \ket{v} = \lambda(\alpha\ket{0} + \beta \ket{1}) \;\; \forall \;\; \alpha,\beta,\lambda \in \mathbb{C} \;\; and \;\; |\alpha|^2 + |\beta|^2 = 1
$$

Additionally, any complex number can be represented in terms of its polar coordinates $re^{i\phi}$ where $r\in\mathbb{R}$ and $\phi \in [0,\pi]$. While not immediately obvious, this polar representation is just an extension of the projective space idea we discussed earlier. Therefore, 

$$
  \ket{v} = r\cdot e^{i\phi}(\alpha\ket{0} + \beta \ket{1}) = r \cdot \ket{\Phi}
$$

where, $r\in\mathbb{R}$, $\alpha,\beta \in \mathbb{C}$ and $\phi \in [0,\pi]$. What we have defined here is a point on a unit circle $\ket{\Phi} \in \mathbb{C}^2$ and a phase factor $e^{i\phi}$ of unit norm.



<br>
<hr>

# References
[^qubit]: [https://www.ibm.com/think/topics/qubit](https://www.ibm.com/think/topics/qubit)
[^wiki_bloch]: [https://en.wikipedia.org/wiki/Bloch_sphere](https://en.wikipedia.org/wiki/Bloch_sphere)
[^math_space]: [https://en.wikipedia.org/wiki/Space_(mathematics)](https://en.wikipedia.org/wiki/Space_(mathematics))
[^bloch_video]: [https://www.youtube.com/watch?v=KEzmw6cKOlU](https://www.youtube.com/watch?v=KEzmw6cKOlU)
