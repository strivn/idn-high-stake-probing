# High-Stakes Activation Probes on Indonesian

Research project completed as part of the [Bluedot Technical AI Safety](https://bluedot.org/) cohort.

Explored the mechanisms in [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) (NeurIPS 2025) with cross-lingual evaluation (Indonesian) and SAE-based failure analysis, on Llama 3.1 8B and Gemma 3 12B.

Main Findings:
- Probes cross-lingual performance (Indonesian) degradation is small: 0.9% mean AUROC drop on Llama, 1.0% on Gemma. (Note: it's a language absent from probe training data)
- However, 8–11% of individual predictions flip when the same example is translated, suggesting the probe is less stable on specific examples than the aggregate numbers imply
- The synthetic-to-real domain gap (6-13%) is far larger than the language gap, and is the more pressing concern for deployment
- SAE decomposition on Llama surfaces broad domain correlates of error (technology, finance) but not specific failure mechanisms

## Report

Refer to the [working report](writing/latex/technical_report_v3.pdf) for full methodology and results.

## Repository Structure

- `experiments/notebooks/` — Jupyter notebooks for activation extraction, probe training, layer sweep, cross-lingual eval, and SAE failure analysis
- `experiments/lib/` — shared utilities for model loading, activation extraction, probing, and evaluation
- `writing/latex/` — technical report (PDF + source)
- `data/` — datasets (gitignored)
