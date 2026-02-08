# Project: Title TBD

## Research Question

**Do activation probes for detecting high-stakes LLM interactions generalize across languages, and can we use Sparse Autoencoders (SAEs) to understand why they fail?**

## Motivation

Language models are increasingly deployed globally, but safety mechanisms are primarily developed and tested in English. If a probe can detect dangerous interactions in English but fails in Indonesian, that's a safety gap. Understanding *why* these failures happen mechanistically could help build more robust safety tools.

Main reference: https://arxiv.org/pdf/2506.10805

## Project Phases

### Prelim: Simple Linear Probes Exercise
**Goal**: Understand slightly better about what 

### Phase 1: Baseline Reproduction
**Goal:** Replicate the core experiment from "Detecting High-Stakes Interactions with Activation Probes"

**What we'll do:**
- Train linear probes on Llama-3.1-8B activations using their dataset
- Evaluate on standard test sets
- Document baseline performance and identify failure cases

**What we'll learn:** How to train activation probes, what "high-stakes" detection looks like, where the baseline fails

### Phase 2: Indonesian Cross-Lingual Extension
**Goal:** Test if probes trained on English generalize to Indonesian

**What we'll do:**
- Translate 500 balanced examples (250 high/250 low stakes) from the synthetic test set to Indonesian using the Anthropic API (Haiku 4.5)
- The source data is multilingual (~59% EN, ~14% DE, ~14% FR, ~13% HI), so translations are from all source languages to Indonesian
- Evaluate translation quality with automated scoring (faithfulness + naturalness) and manual spot-checks
- Run two experiments:

**Experiment 2a: Zero-shot cross-lingual transfer**
- Feed the 500 Indonesian translations through Llama-3.1-8B to extract activations
- Evaluate using the **English-trained probe** (no retraining)
- Question: Does the probe generalize to a language it never saw during training?
- Hypothesis: Moderate degradation (AUROC ~0.70-0.80) because the model likely has partially language-agnostic representations of "high-stakes" concepts, but the linear decision boundary was fit to EN/FR/DE/HI activation geometry

**Experiment 2b: Mixed-language training**
- Translate an additional 350 examples from the **train split** to Indonesian
- Replace 350 random examples in the 8000-example training set with their Indonesian translations (keeping total size constant at 8000)
- Hold out 150 Indonesian test translations for evaluation
- Retrain the probe on the mixed-language training set
- Question: Does including Indonesian training data improve Indonesian performance? Does it hurt English performance?
- Hypothesis: Significant recovery on Indonesian (AUROC ~0.82-0.90), minimal degradation on English, suggesting the model *has* the concept in Indonesian but the probe needs calibration to find it

**What we'll learn:** Whether "high-stakes" is a language-universal feature in the model's representation space (Exp 2a works well) or language-specific (Exp 2a fails, 2b recovers). This has direct implications for deploying safety probes globally -- if zero-shot transfer works, you can train on English and deploy anywhere.

### Phase 3: SAE-Based Failure Analysis
**Goal:** Use Sparse Autoencoders to understand *why* probes fail on specific examples

**What we'll do:**
- Load pre-trained SAEs from Llama Scope
- For failure cases (both English and Indonesian):
  - Decompose the model's activations into interpretable SAE features
  - Identify which features are active when the probe succeeds vs fails
  - Use automated interpretation to understand what each feature represents
- Look for patterns: Do certain feature types predict failure?

**What we'll learn:**
- Why do probes fail on jokes about serious topics?
- Why do keyword-based false positives happen?
- Are the failure mechanisms different across languages?
- Can we predict which inputs will cause failures based on their feature activation patterns?

### Phase 4 : SAE-Based Probe Training
**Goal:** Compare probes trained on SAE features vs raw activations

**What we'll do:**
- Train probes using SAE features as input instead of raw activations
- Compare performance, especially on out-of-distribution cases (Indonesian)
- Test if SAE-based probes are more interpretable but less accurate

**What we'll learn:** Is there a trade-off between interpretability and robustness? Do SAE-based probes generalize better or worse cross-lingually?

## Why This Matters

**Practical Safety:** If probes don't work across languages, that's a deployment risk for global applications.

**Scientific Understanding:** Using SAEs to diagnose failure modes connects two interpretability paradigms (probes for detection, SAEs for understanding).

**Novel Contribution:** No existing work systematically studies:
- Cross-lingual robustness of high-stakes probes
- SAE-based diagnosis of *why* probes fail (most work asks *whether* SAE probes work)

## Expected Challenges

1. **Translation quality:** High-stakes scenarios might be culturally/linguistically nuanced. Manual verification is critical.

2. **SAE interpretation:** SAE features might not align with human-interpretable concepts. We'll use automated interpretation (LLMs) but need to validate.

3. **Compute resources:** Need GPU access for running Llama-3.1-8B and extracting activations. Estimated cost: $50-150 if using cloud compute.

4. **SAE limitations:** SAEs themselves can capture spurious correlations (per ICML 2025 findings), so we need to be careful about over-interpreting results.

## Success Criteria

**Minimum viable contribution:**
- Demonstrate whether high-stakes probes generalize to Indonesian
- Identify at least 2-3 concrete failure modes
- Show preliminary SAE analysis of why those failures occur

**Ideal outcome:**
- Quantified cross-lingual generalization gap
- Clear mechanistic explanation of failure modes using SAE features
- Actionable insights for building more robust probes
- Publishable write-up with visualizations

## Timeline (25-30 hours)

- **Phase 1:** 6-8 hours (reproduction and baseline)
- **Phase 2:** 6-8 hours (translation and cross-lingual testing)
- **Phase 3:** 8-10 hours (SAE failure analysis - core contribution)
- **Phase 4:** 4-6 hours (optional, if time permits)
- **Write-up:** 5 hours (documentation, blog post, visualizations)

## Key Resources

**Paper:** "Detecting High-Stakes Interactions with Activation Probes" (arXiv:2506.10805)
**Code:** github.com/arrrlex/models-under-pressure
**SAEs:** Llama Scope (Llama-3.1-8B SAEs on Hugging Face)
**Tools:** SAElens, TransformerLens, PyTorch

## Open Questions

- How much does translation quality affect probe performance?
- Will SAE features reveal interpretable failure patterns, or will they be too abstract?
- Is cross-lingual failure due to language-specific features or translation artifacts?
- Can we build a "failure predictor" that warns when a probe is likely to be wrong?
