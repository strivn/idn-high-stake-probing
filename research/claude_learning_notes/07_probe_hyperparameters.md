# Probe Hyperparameters: Intuition and Why They Matter

## The Probe Configuration

```python
LogisticRegression(C=1e-3, solver='lbfgs', max_iter=1000, fit_intercept=False)
```

preceded by `StandardScaler()`.

---

## Parameter-by-Parameter Breakdown

### L2 Regularization

Logistic regression tries to find a hyperplane that separates high-stakes from low-stakes in activation space. Without regularization, the model could assign huge weights to specific dimensions that happen to work on training data but don't generalize.

L2 (ridge) regularization adds a penalty: minimize `loss + λ × ||w||²`. This shrinks all weights toward zero, preferring solutions where the signal is spread across many dimensions rather than relying on a few.

**Why it matters here:** Activations are 4096-dimensional. Without regularization, the probe would overfit to noise in individual dimensions. L2 forces the probe to find a *robust* direction in activation space.

### C = 1e-3 (Strong Regularization)

`C` is the inverse regularization strength: `C = 1/λ`. So `C=1e-3` means `λ=1000` — very strong regularization.

**Intuition:** This says "I'd rather have a simple, wrong-ish model than a complex model that memorizes the training data." With 4096 features and only 8000 examples, overfitting is a real risk. Strong regularization is a conservative choice.

**What would happen with C=1.0 (weaker)?** The probe would fit the training data more tightly but might not transfer as well to Indonesian — it could latch onto language-specific features.

**Why 1e-3 specifically?** Likely found through cross-validation in the original paper. It's a standard "strong regularization" default for high-dimensional problems.

### fit_intercept=False

This means the decision boundary passes through the origin in (scaled) activation space. The probe is purely directional — it finds a direction where high-stakes examples project positively and low-stakes project negatively.

**Why?** After `StandardScaler` centers the data at zero, adding an intercept would allow the probe to shift the boundary away from center. Setting `fit_intercept=False` prevents this. The idea is that the "stakes" concept should be a *direction* in representation space, not a region offset from center. This makes the probe more interpretable: the weight vector IS the "high-stakes direction."

**Tradeoff:** If the positive and negative classes aren't symmetric around the centroid (after scaling), this forces a suboptimal boundary. But it prevents a degenerate solution where the intercept does all the work.

### solver='lbfgs'

LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization algorithm. Compared to alternatives:

- **SGD:** Stochastic, noisy, needs learning rate tuning. Good for huge datasets.
- **liblinear:** Coordinate descent. Fast for small/medium data. Doesn't support L2 with this setup well.
- **LBFGS:** Uses second-order curvature information (Hessian approximation) → converges faster and more reliably. Deterministic.

**Why it matters:** LBFGS gives deterministic results (given the same data). No learning rate to tune. For 8000×4096 data, it's fast enough and reliable.

### max_iter=1000

Default is 100. Bumping to 1000 ensures convergence. With strong regularization (C=1e-3), the optimization landscape is smooth and LBFGS usually converges in <100 iterations. But 1000 is a safety margin.

---

## The Full Pipeline: StandardScaler → LogisticRegression

```
Raw activations (4096-dim)
    ↓ StandardScaler
    ↓ (zero mean, unit variance per dimension)
Normalized activations
    ↓ LogisticRegression
    ↓ (find separating hyperplane through origin)
P(high-stakes) ∈ [0, 1]
```

**Why StandardScaler first?** Activation dimensions have wildly different scales. Without scaling, the L2 penalty would be dominated by large-magnitude dimensions, and the probe would ignore small-but-informative dimensions. Scaling makes L2 treat all dimensions equally.

**Combined effect:** The pipeline finds a direction in *standardized* activation space that best separates the classes, with strong pressure to keep the weight vector small. This is essentially asking: "what is the most reliable direction for detecting stakes, treating all dimensions as equally important a priori?"

---

## Why This Configuration is Good for Cross-Lingual Transfer

The strong regularization + no intercept combination is actually *ideal* for our cross-lingual experiment:

1. **Strong L2** prevents overfitting to English-specific features → the probe must find robust, language-general directions
2. **No intercept** forces a purely directional decision → if the "stakes" direction is truly language-invariant, the probe should transfer
3. **Deterministic solver** means results are reproducible (modulo floating-point non-determinism in activation extraction)

If we'd used C=1.0 with an intercept, the probe might have achieved higher English AUROC but worse Indonesian transfer — by memorizing English-specific patterns rather than finding the universal signal.
