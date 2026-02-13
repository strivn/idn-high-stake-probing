# Dimensionality Reduction and Clustering: Intuitions and Technical Details

This note covers techniques for visualizing and understanding high-dimensional activation spaces in neural networks.

## Quick Intuition: Why Do We Need This?

Neural network activations live in high-dimensional spaces (e.g., 4096 dimensions for Llama 3.1 8B). Humans can't visualize beyond 3D. Dimensionality reduction techniques project high-D data into 2D/3D while preserving important structure, letting us "see" patterns in the data.

**The core question:** Do failure cases cluster together? Do they share common features in activation space?

---

## PCA (Principal Component Analysis)

### Intuition

Imagine you're photographing a pencil. If you rotate the pencil, some viewing angles capture more variation (length) than others (width). PCA finds the "best viewing angles" for your data—the directions with maximum variance.

**Analogy:** You have a cloud of 3D points shaped like a pancake. PCA identifies:
- Principal Component 1 (PC1): The pancake's length (most variance)
- PC2: The pancake's width (second most variance)
- PC3: The pancake's thickness (least variance)

You can now view the pancake from above (PC1 vs PC2) and capture most of its structure in 2D.

### How It Works

1. **Center the data:** Subtract the mean from each dimension
2. **Compute covariance matrix:** How much each dimension varies with others
3. **Eigen-decomposition:** Find eigenvectors (principal components) and eigenvalues (variance explained)
4. **Project:** Transform data onto top-k components

### Math

Given data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d dimensions):

1. Center: $X_c = X - \bar{X}$
2. Covariance: $C = \frac{1}{n-1} X_c^T X_c$
3. Eigen-decomposition: $C = V \Lambda V^T$ where $V$ are eigenvectors, $\Lambda$ are eigenvalues
4. Project onto top 2 components: $X_{2D} = X_c \cdot V[:, :2]$

### When to Use

- **Fast and deterministic:** Same input = same output
- **Linear transformations only:** Good for Gaussian-ish distributions
- **Interpretable:** Components are weighted sums of original features
- **Preserves global structure:** Maintains large-scale distances

### Limitations

- Assumes linear relationships (can't capture manifolds)
- Sensitive to outliers
- May not preserve local neighborhoods well

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Intuition

t-SNE is like a "social network" algorithm. It tries to place similar points close together and dissimilar points far apart, preserving **local neighborhoods** rather than global distances.

**Analogy:** You have a map of friends. t-SNE ensures that:
- Your close friends stay close in the visualization
- Distant acquaintances can be anywhere far away
- It doesn't care if New York and LA are exactly 2,800 miles apart, just that they're "far"

### How It Works

1. **In high-D:** Compute pairwise similarities using Gaussian kernel (nearby points = high similarity)
2. **In low-D:** Compute pairwise similarities using t-distribution (heavy tails = more room for distant points)
3. **Optimize:** Move points in 2D to make the low-D similarities match the high-D similarities (minimize KL divergence)

### Math

**High-D similarity** (conditional probability that point $i$ picks $j$ as neighbor):
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

**Low-D similarity** (using t-distribution with 1 degree of freedom):
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

**Objective:** Minimize KL divergence $\text{KL}(P || Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$

### Key Parameters

- **Perplexity (5-50):** Roughly "how many neighbors to preserve". Higher = global structure, lower = local clusters
- **Learning rate:** Step size for gradient descent
- **Iterations:** More = better convergence (typically 1000-5000)

### When to Use

- **Cluster discovery:** Excellent at revealing hidden groups
- **Non-linear relationships:** Can capture manifolds and complex structures
- **Exploratory analysis:** "What patterns exist?"

### Limitations

- **Stochastic:** Different runs = different layouts (set `random_state` for reproducibility)
- **Slow:** $O(n^2)$ complexity, struggles with >10k points (use openTSNE or MulticoreTSNE)
- **Distances are NOT meaningful:** Only clustering matters, not exact positions or distances between clusters
- **Can create false clusters:** May split continuous data into artificial groups
- **Not deterministic:** Global structure can be misleading

---

## UMAP (Uniform Manifold Approximation and Projection)

### Intuition

UMAP is like t-SNE's faster, more principled cousin. It also preserves local neighborhoods but **better preserves global structure** and scales to larger datasets.

**Analogy:** Imagine a crumpled piece of paper (your high-D manifold). UMAP carefully "unfolds" it into 2D while preserving:
- Nearby regions stay nearby (local structure)
- Distant regions stay roughly in the right direction (global structure)

### How It Works

1. **Build a graph:** Connect each point to its k-nearest neighbors in high-D
2. **Assign edge weights:** Based on distance (closer neighbors = stronger edges)
3. **Optimize a low-D graph:** Minimize difference between high-D and low-D graph structures using force-directed layout

Under the hood, UMAP uses concepts from topological data analysis and Riemannian geometry.

### Math (Simplified)

**High-D graph edges:** Connect each point to k-nearest neighbors with fuzzy weights
$$w_{ij} = \exp\left(-\frac{\max(0, ||x_i - x_j|| - \rho_i)}{\sigma_i}\right)$$

where $\rho_i$ is distance to nearest neighbor, $\sigma_i$ normalizes local density.

**Low-D objective:** Similar to t-SNE but uses cross-entropy loss on graph structure

### Key Parameters

- **n_neighbors (5-100):** More neighbors = more global structure preserved
- **min_dist (0.0-0.99):** How tightly to pack points (0.0 = tight clusters, 0.5 = looser)
- **metric:** Distance function (Euclidean, cosine, etc.)

### When to Use

- **Large datasets:** Much faster than t-SNE (~10-100x)
- **Global + local structure:** Preserves both better than t-SNE
- **Transfer learning:** Can project new points onto existing UMAP embedding
- **Theoretical foundation:** Based on manifold learning theory

### Advantages over t-SNE

- Faster (can handle 100k+ points)
- More consistent global structure
- Can add new points without retraining
- Better hyperparameter stability

---

## Comparison Summary

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Speed** | Very fast | Slow ($O(n^2)$) | Fast ($O(n \log n)$) |
| **Preserves** | Global variance | Local neighborhoods | Local + global |
| **Deterministic** | Yes | No (stochastic) | No (but more stable) |
| **Interpretability** | High (linear) | Low (complex) | Medium |
| **Cluster quality** | Poor | Excellent | Excellent |
| **Scalability** | 100k+ | <10k | 100k+ |
| **Distance meaning** | Yes | No | Somewhat |
| **New points** | Easy | Hard (retrain) | Easy (transform) |

---

## Practical Workflow for Failure Analysis

### Step 1: Initial Exploration
```python
# Quick overview with PCA (fast, deterministic)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(activations)
# Check: explained_variance_ratio_ to see if linear structure exists
```

### Step 2: Cluster Discovery
```python
# Find non-linear clusters with UMAP (or t-SNE if <5k points)
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(activations)
# Visualize with failures colored differently
```

### Step 3: Interpret Clusters
- Do false positives cluster separately from true positives?
- Do EN/ID failures overlap or separate?
- Are failures near the decision boundary or far from it?

---

## TPR @ FPR Intuition

### What It Means

**True Positive Rate @ X% False Positive Rate** answers:
> "If I'm willing to tolerate X% false alarms, how many real high-stakes cases can I catch?"

### Concrete Example

You're monitoring 1000 interactions:
- 100 are truly high-stakes
- 900 are low-stakes

Your probe's threshold is set such that:
- **1% FPR:** You flag 9 low-stakes cases as high-stakes (1% of 900)
- At this threshold, suppose you catch 43 high-stakes cases
- **TPR @ 1% FPR = 43%** (43 out of 100 high-stakes caught)

### Why This Metric Matters

**AUROC is misleading for imbalanced problems.** A classifier can have high AUROC (0.95) but terrible TPR@1%FPR (10%) if it's bad at the decision boundary.

**Real-world deployment constraints:**
- **Standalone monitor (strict):** Need ~1% FPR (can't annoy users with false alarms) → Need high TPR@1%FPR
- **Initial filter (relaxed):** Can tolerate 20-40% FPR (false positives just trigger review) → TPR@20%FPR is acceptable

**From the paper:**
> "At 1% FPR, our best probe detects only ~43% of high-stakes cases vs ~55% for Llama-70B. This means neither probes nor baselines could be deployed standalone. However, at 20-40% FPR, probes achieve substantially better TPR, making them effective initial filters."

### How to Interpret

- **High TPR @ Low FPR:** Excellent—catches most threats with few false alarms
- **Low TPR @ Low FPR:** Poor—misses most threats unless you accept many false alarms
- **TPR @ 10% FPR vs 50% FPR:** Shows how the probe's quality degrades as you tighten the threshold

### Visualization

The ROC curve plots TPR vs FPR at all thresholds. TPR@X%FPR is a single point on that curve.

```
TPR
 ^
 |     /----  (Ideal: TPR=100% at FPR=1%)
 |    /
 |   /
 |  /___      (Realistic: TPR=43% at FPR=1%)
 | /
 |/__________
0%         100% FPR
```

---

## For Your Notebook 6

**Analysis plan:**
1. **PCA first:** Quick sanity check—is there any linear separation?
2. **UMAP main viz:** Find clusters of failures, compare EN vs ID
3. **Annotate clusters:** Map back to SAE features to understand what each cluster represents
4. **TPR@FPR analysis:** Compute TPR@1%, TPR@5%, TPR@20% for EN vs ID to quantify cross-lingual robustness

---

## References

- **PCA:** Pearson (1901), Hotelling (1933)
- **t-SNE:** van der Maaten & Hinton (2008), "Visualizing Data using t-SNE"
- **UMAP:** McInnes et al. (2018), "UMAP: Uniform Manifold Approximation and Projection"
- **Sklearn docs:** https://scikit-learn.org/stable/modules/manifold.html
- **UMAP docs:** https://umap-learn.readthedocs.io/
