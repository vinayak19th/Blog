---
title: "A Geometric Intro to Metric Learning"
short_title: metric_learning
description: An introduction to metric learning from a more geometric perspective and math perspective
author: vinayak
date: 2026-02-24 11:33:00 +0800
categories: [Article, Machine Learning]
tags: [machine learning, metric learning, geometric deep learning, research]
pin: false
math: true
comments: true
---

## Introduction

Metric learning is a fascinating corner of machine learning where the primary objective isn't just to classify or regress, but to *learn a distance function*. At its core, metric learning seeks to transform data into a space where similar objects are geometrically close to one another, and dissimilar objects are far apart. 

But what does this mean geometrically? How do these transformations sculpt the high-dimensional space where our data lives?

## The Geometric Intuition

Imagine your raw data—perhaps images, text, or quantum states—as points scattered in a high-dimensional space. In this original space, the Euclidean distance between two points might not reflect their true semantic similarity. Two images of the same person under different lighting might be far apart in pixel space.

Metric learning applies a transformation (often non-linear, via neural networks) that warps this space. 

### Euclidean Space vs. Manifolds

1. **Linear Metrics (Mahalanobis Distance):**
   Geometrically, learning a Mahalanobis distance is equivalent to applying a linear transformation (a rotation and scaling) to the input space, followed by calculating the standard Euclidean distance. It acts as an ellipsoid bounding around data clusters.
   
   $$ D_M(x, y) = \sqrt{(x - y)^T M (x - y)} $$
   
   If $M = L^T L$, this is exactly equivalent to projecting the points using $L$ and then finding the Euclidean distance: $\|Lx - Ly\|_2$.

2. **Deep Metric Learning:**
   With deep learning, we map data onto a complex manifold. The neural network learns a smooth manifold where the geodesic distance (or Euclidean distance in the embedding space) aligns with semantic meaning. 
   
In traditional machine learning, we often frame problems as classification tasks: drawing arbitrary boundaries between predefined categories. However, there is a fundamental problem with this approach: what happens when we encounter a class we have never seen before? What if we want to measure how different two inputs are, rather than just assigning them discrete labels?

Enter **Metric Learning**.

A paradigm shift from traditional classification, metric learning teaches a model to learn similarity. The core goal is highly geometric: pull "friends" (similar items) close together in a mathematical space, and push "strangers" (dissimilar items) far apart.

To truly understand how feature spaces are trained to achieve this, we have to strip away the neural networks and look at the underlying geometry and linear algebra. Let's dive into the math.

---
## The Geometry of Feature Spaces

When we feed an image, a sentence, or a piece of audio through an encoder (like a CNN, Vision Transformer, or LLM Encoder), we are mapping raw data into a Latent Space. Geometrically, this space is just a formalized Vector Space.

### 1. Vector Spaces Formalized

> **Definition 1 (Vector Space).** A vector space is a formal collection of objects (vectors) that can be scaled and added together while remaining within the same set, governed by a specific set of algebraic rules.
{: .prompt-definition }

Let $\mathcal{V}$ be a nonempty set over a field $\mathbb{F}$ equipped with addition ($\mathcal{V} \times \mathcal{V} \rightarrow \mathcal{V}$) and scalar multiplication ($\mathbb{F} \times \mathcal{V} \rightarrow \mathcal{V}$). For $\mathcal{V}$ to be a valid vector space, it must satisfy ten strict conditions:

- **A1-A2. Closure**: Closed under vector addition and scalar multiplication.
- **A3. Commutativity of addition**: For all $u, v \in \mathcal{V}$, $u + v = v + u$.
- **A4. Associativity of addition**: For all $u, v, w \in \mathcal{V}$, $(u + v) + w = u + (v + w)$.
- **A5. Zero vector**: Exists $0 \in \mathcal{V}$ such that $v + 0 = v$.
- **A6. Additive inverses**: For each $v \in \mathcal{V}$, there exists $-v$ such that $v + (-v) = 0$.
- **A7. Unit property**: For all $v \in \mathcal{V}$, $1v = v$.
- **A8. Associativity of scalar multiplication**: $(rs)v = r(sv)$.
- **A9-A10. Distributive properties**: $r(u + v) = ru + rv$ and $(r + s)v = rv + sv$.

Common examples include $\mathbb{R}^n$ (n-tuples of real numbers), $\mathbb{C}^n$, and $M_{m \times n}(\mathbb{R})$ (real-valued matrices).

### 2. Measuring Similarity: Inner Product Spaces

Knowing that our data lives in a vector space isn't enough; we need a mathematical tool to compare two vectors. This is the **Inner Product**.

> **Definition 2 (Inner Product Space).** An inner product is a map that associates each pair of vectors $\langle u, v \rangle$ with a scalar.
{: .prompt-definition }

Any function that acts as an inner product must satisfy:

- **Conjugate Symmetry**: $\langle u, v \rangle = \langle v, u \rangle$
- **Linearity in the First Argument**: $\langle au + bv, w \rangle = a\langle u, w \rangle + b\langle v, w \rangle$ (where $a, b$ are scalars).
- **Positive Definiteness**: $\langle u, v \rangle \ge 0$, and $\langle u, v \rangle = 0 \iff u = 0 \text{ or } v = 0$.

In standard $\mathbb{R}^n$ spaces, the "Dot Product" serves as our inner product. Geometrically, it tells us about the angle $\theta$ and magnitude difference between vectors:

$$
\theta = \arccos\left(\frac{x \cdot y}{\|x\| \|y\|}\right)
$$

### 3. Redefining Distance: Metric Functions

What does "Metric" actually mean in Metric Learning? 

> **Definition 3 (Metric Function).** A metric (or distance function) is a rule $d(x, y)$ assigning a non-negative real number to any two points.
{: .prompt-definition }

To be a "true" metric, it must abide by these logical rules:

- **Non-Negativity**: $d(x, y) \ge 0$ (Distance can't be negative).
- **Identity of Indiscernibles**: $d(x, y) = 0 \iff x = y$ (If distance is zero, they are the same point).
- **Symmetry**: $d(x, y) = d(y, x)$.
- **Triangle Inequality**: $d(x, z) \le d(x, y) + d(y, z)$ (A direct path is never longer than a detour).

Because Euclidean distance is derived directly from the inner product ($d(p_0, p_1)^2 = \langle p_0 - p_1, p_0 - p_1 \rangle$), optimizing a metric function in neural networks is often mathematically equivalent to optimizing inner products.

---
## The Training Engine - Loss Functions

```html
 <center>
   <div class="plotly-figure">
       {% include plotly/metric_learning/Metric_Training_triplet.html %}
   </div>
  <i style="margin-top:0.4rem"><b>Fig 3.</b> Interactive Real Projective Space</i> 
</center> -->
```

To shape this vector space so that distance equals semantic meaning, we rely on specific loss functions during training. Let's break down three of the most important ones.

### 1. Triplet Loss

Triplet loss forms the backbone of systems like FaceNet. It works by evaluating three vectors simultaneously:

- **Anchor ($f_A$)**: A baseline data point.
- **Positive ($f_P$)**: A data point of the same class/identity as the Anchor.
- **Negative ($f_N$)**: A data point of a different class/identity.

The objective is geometric: the distance between the Anchor and Positive ($AP$) must be smaller than the distance between the Anchor and Negative ($AN$) by at least a margin $\alpha$.

$$
\mathcal{L}_{A,P,N} = \max\left(0, \|f_A - f_P\|_2^2 - \|f_A - f_N\|_2^2 + \alpha\right)
$$

The margin $\alpha$ gives "strangers" their personal space, preventing the network from collapsing all embeddings into a single point.

### 2. Prototypical Loss

While Triplet loss is great for distinct instance comparisons, Prototypical Loss is built for few-shot and zero-shot learning.

Instead of comparing individual pairs, we create "proto-vectors" ($c_k$)—the mean representations of a set of support vectors for a given class $k$. The network classifies a new query sample $x$ by finding the nearest proto-vector in the metric space.

The loss is calculated by applying a softmax over the distances between the query embedding $f_\phi(x)$ and all class prototypes:

$$
\mathcal{L}_{\phi} = -\log p(y=k|x) = -\log \frac{\exp(-d(f_{\phi}(x), c_k))}{\sum_j \exp(-d(f_{\phi}(x), c_j))}
$$

By minimizing this loss, the network learns a space where classes form tight, distinct clusters around their prototypes.

### 3. CLIP Loss (Contrastive Multi-Modal Loss)

How do we bridge the gap between different domains, like text and images? We use a contrastive loss that aligns two distinct vector spaces.

For a batch of $N$ image-text pairs, the model generates $N$ image embeddings ($I_i$) and $N$ text embeddings ($T_i$). It computes the pairwise cosine similarities. The goal is to maximize the similarity of the $N$ correct pairs along the diagonal of the similarity matrix while minimizing the similarity of the $N^2 - N$ incorrect pairs.

The symmetric cross-entropy loss function used here is:

$$
\mathcal{L}_{CLIP} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{e^{\text{sim}(I_i, T_i)/\tau}}{\sum_{j=1}^N e^{\text{sim}(I_i, T_j)/\tau}} + \log \frac{e^{\text{sim}(I_i, T_i)/\tau}}{\sum_{j=1}^N e^{\text{sim}(I_j, T_i)/\tau}} \right]
$$

(Where $\tau$ is a learnable temperature parameter that scales the logits).

---
## Conclusion

Metric learning strips away the rigid confines of standard classification layers. By forcing models to respect the axioms of inner products and metric spaces, we create robust representations capable of multi-modal search, zero-shot learning, and high-dimensional information retrieval.

Ultimately, deep learning is just geometry in high dimensions—and metric learning is how we teach our models to hold the ruler.