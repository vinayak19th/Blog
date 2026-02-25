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
   
## Contrastive Loss and Triplet Loss

To understand how the space is manipulated during training, we must look at the loss functions.

### Contrastive Loss: Pull and Push
Contrastive loss operates on pairs. 
- **Positive pairs** behave like they are connected by springs. The loss pulls them together until they are at the origin relative to each other.
- **Negative pairs** act like repelling magnets. If they are closer than a defined margin $m$, a repulsive force pushes them apart.

### Triplet Loss: Relative Geometry
Triplet loss considers an anchor $a$, a positive $p$, and a negative $n$.
Instead of absolute distances, it enforces a *relative* geometric constraint:
The anchor should be closer to the positive than to the negative by at least a margin $\alpha$.

$$ d(a, p) + \alpha < d(a, n) $$

Geometrically, this creates a hypersphere around the anchor. The positive sample must be pulled inside this hypersphere, while the negative sample is pushed outside of it.

## The Hypersphere: Hyperspherical Embeddings

Modern metric learning often constrains the embeddings to lie on the surface of a unit hypersphere ($\|x\|_2 = 1$). Why?

1. **Cosine Similarity:** On a unit hypersphere, Euclidean distance and Cosine similarity become tightly coupled. The angle between vectors becomes the sole measure of similarity.
2. **Space Utilization:** High-dimensional spaces suffer from the curse of dimensionality. By constraining points to the surface of a hypersphere, the model is forced to distribute clusters uniformly across the surface, maximizing the margin between different classes.

## Connections to Quantum Machine Learning

In Quantum Machine Learning (QML), quantum states naturally reside on the Bloch sphere (for a single qubit) or a higher-dimensional complex projective space (Hilbert space). The unitary operations in a parameterized quantum circuit are inherently rotations in this space. Metric learning in QML, therefore, has profound geometric interpretations, as we are directly manipulating the angles and distances between quantum states to reflect similarity!

*(Expand on QML metric learning...)*

## Conclusion

By viewing metric learning through a geometric lens, the abstract math translates into a tangible process of molding, stretching, and folding spaces. As algorithms become more complex, holding onto this geometric intuition is essential for designing better loss functions and understanding model behavior.

---
*Draft notes: Expand on the QML section, add custom diagrams for contrastive vs triplet loss, and include an interactive geometric visualization if possible.*
