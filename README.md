# Learning Probability Density Functions using GAN  
## Assignment – PDF Estimation of Transformed NO₂ Distribution

---

## 📌 Objective

The objective of this assignment is to learn the probability density function (PDF) of an unknown transformed random variable using a Generative Adversarial Network (GAN), without assuming any parametric distribution.

We consider **NO₂ concentration** as the feature \( x \), apply a nonlinear transformation, and train a GAN to approximate the resulting distribution.

---

## 📊 Dataset

**India Air Quality Dataset**  
Source: Kaggle  
https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data  

Feature used: **NO₂ concentration**

---

## 🔁 Step 1: Nonlinear Transformation

Each NO₂ value \( x \) is transformed as:

\[
z = x + a_r \sin(b_r x)
\]

Where:

- \( a_r = 0.5 (r \mod 7) \)
- \( b_r = 0.3 ((r \mod 5) + 1) \)
- \( r = 102303068 \)

### ✅ Computed Parameters
a_r = 1.0
b_r = 1.2


The transformation introduces nonlinear distortion into the original distribution.

---

## 🤖 Step 2: GAN Architecture

The GAN learns the distribution of the transformed variable \( z \).

### Generator

- Input: 10-dimensional Gaussian noise \( N(0,1) \)
- Architecture:
  - Linear(10 → 64)
  - ReLU
  - Linear(64 → 32)
  - ReLU
  - Linear(32 → 1)
- Output: Generated sample \( z_f \)

### Discriminator

- Input: 1-dimensional value (real or fake \( z \))
- Architecture:
  - Linear(1 → 32)
  - LeakyReLU
  - Linear(32 → 16)
  - LeakyReLU
  - Linear(16 → 1)
  - Sigmoid
- Output: Probability of real vs fake

Loss Function: Binary Cross Entropy (BCE)  
Optimizer: Adam  
Epochs: 2000  

---

## 📈 Step 3: PDF Estimation

After training:

1. 10,000 samples are generated from the Generator.
2. Kernel Density Estimation (KDE) is applied.
3. The estimated PDF is compared with the real transformed distribution.

---

## 📊 Distribution Comparison

Below is the comparison between:

- Real transformed distribution (Histogram + KDE)
- GAN estimated PDF

![PDF Comparison](comparison_graph.png)

> ⚠️ Upload your saved comparison plot image as `comparison_graph.png` in the repository.

---
