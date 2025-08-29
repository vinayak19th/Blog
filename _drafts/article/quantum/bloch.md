---
title: Bloch Sphere Demystified - Hitchhikers Guide to Quantum States
description: A simplified explanation of the Bloch sphere representation of quantum states
author: Vinayak
date: 2025-08-16 11:33:00 +0800
categories: [Articles, Quantum Computing]
tags: [quantum computing, quantum machine learning]
pin: false
math: true
comments: true
---

{% assign parts = page.path | split: '/' %}
{% assign image_path = site.post_image_path | append: '/' | append: parts[1] | append: '/' | append: parts[2] | append: '/' | append: "bloch" %}

This article aims to provide a simplified and more intuitive explanation of the Bloch sphere assuming that the reader has a basic understanding of quantum computing such as what is a [qubit](https://www.ibm.com/think/topics/qubit)[^qubit], what are basis states and a basic understanding of high school geometry.

## What is the Bloch Sphere?
The bloch sphere is a geometric representation of the state space of a single '*pure state*' qubit. 

According to [wikipedia](https://en.wikipedia.org/wiki/Bloch_sphere)[^wiki_bloch], we can define the bloch sphere as follows:

**Definition 1 [Bloch Sphere]:**
*In quantum mechanics and computing, the Bloch sphere is a geometrical representation of the pure state space of a two-level quantum mechanical system (qubit), named after the physicist Felix Bloch.*

While this definition is technically accurate, it may not be very intuitive for someone new to the concept. Let's break it down into simpler terms. 

**What exactly is a geometric representation?**
In really simple terms, a geometric representation is a way to represent a set of objects (often numbers) as points in a geometric space. For example, we can represent real numbers on a number line, which is a one-dimensional geometric space. Similarly, we can represent complex numbers on a two-dimensional plane called the complex plane.

![Complex to plane]({{image_path}}/Complex.png){: .light }{: width="500" }
![Complex to plane]({{image_path}}/ComplexDark.png){: .dark }{: width="500" }

As the figure shows, we can take the general equation of a complex number defined below and plot them on a 2D plane. 
$$
  z = x + iy \in \mathbb{C} \;\; \forall x,y \in \mathbb{R}
$$

**The key idea is that every single unique assignment of $(x,y)$ corresponds to a unique point on the 2D plane**

<hr>

## *References*
[^qubit]: https://www.ibm.com/think/topics/qubit
[^wiki_bloch]: https://en.wikipedia.org/wiki/Bloch_sphere
