"""Evaluation metrics and error analysis for probes."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from .data import Example
from .probe import LinearProbe


def tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    valid = np.where(fpr <= target_fpr)[0]
    return tpr[valid[-1]] if len(valid) > 0 else 0.0


def evaluate_probe(
    probe: LinearProbe,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> Dict[str, float]:
    """Evaluate probe and print metrics. Returns metrics dict."""
    probs = probe.predict_proba(X)
    preds = probe.predict(X)

    metrics = {
        "auroc":       roc_auc_score(y, probs),
        "accuracy":    accuracy_score(y, preds),
        "tpr_at_1fpr": tpr_at_fpr(y, probs, 0.01),
        "tpr_at_5fpr": tpr_at_fpr(y, probs, 0.05),
    }

    print(f"\n{name}:")
    print(f"  AUROC:      {metrics['auroc']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  TPR@1%FPR:  {metrics['tpr_at_1fpr']:.4f}")
    print(f"  TPR@5%FPR:  {metrics['tpr_at_5fpr']:.4f}")

    return metrics


def analyze_errors(
    probe: LinearProbe,
    X: np.ndarray,
    examples: List[Example],
    name: str,
    n_examples: int = 5,
):
    """Print detailed error analysis: false positives and false negatives."""
    probs  = probe.predict_proba(X)
    labels = np.array([e.label for e in examples])
    preds  = (probs >= 0.5).astype(int)
    errors = preds != labels

    fp_mask    = errors & (preds == 1)
    fn_mask    = errors & (preds == 0)
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    # sort by confidence: worst false positives first, worst false negatives first
    fp_indices = fp_indices[np.argsort(probs[fp_indices])[::-1]]
    fn_indices = fn_indices[np.argsort(probs[fn_indices])]

    def format_example(ex: Example, max_len: int = 200) -> str:
        for m in ex.messages:
            if m["role"] == "user":
                text = m["content"]
                return f"{text[:max_len]}..." if len(text) > max_len else text
        text = ex.messages[0]["content"]
        return f"{text[:max_len]}..." if len(text) > max_len else text

    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS: {name}")
    print(f"{'=' * 60}")
    print(f"Total errors: {errors.sum()} / {len(errors)} ({100 * errors.mean():.1f}%)")
    print(f"False positives: {fp_mask.sum()}")
    print(f"False negatives: {fn_mask.sum()}")

    print(f"\n--- FALSE POSITIVES (predicted high, actually low) ---")
    for i in fp_indices[:n_examples]:
        print(f"\nP={probs[i]:.3f} | ID: {examples[i].id}")
        print(f"  {format_example(examples[i])}")

    print(f"\n--- FALSE NEGATIVES (predicted low, actually high) ---")
    for i in fn_indices[:n_examples]:
        print(f"\nP={probs[i]:.3f} | ID: {examples[i].id}")
        print(f"  {format_example(examples[i])}")
