# Cross-Lingual Generalization of High-Stakes Activation Probes to Indonesian

**Technical Report — BlueDot AI Safety Course, 2026**

> **Disclaimer:** This report was drafted primarily with Claude Opus and has undergone preliminary verification against the codebase. Additional experiments are planned before finalizing.

---

## Abstract

Activation probes can detect high-stakes interactions in language models, but it is unclear whether probes trained on English-centric data generalize to other languages. We test this by translating the evaluation datasets from McKenzie et al. (2025) into Indonesian and evaluating an English-trained linear probe on the translated inputs. On Llama 3.1 8B, the probe transfers with only 1.2% mean AUROC degradation across three evaluation sets (0.8995 → 0.8872 at layer 12), with near-zero loss on synthetic data and a moderate drop on naturalistic benchmarks. These results suggest that the model's internal representation of "high-stakes" is largely language-invariant at early-to-middle layers, though the synthetic-to-real generalization gap remains a larger concern than the cross-lingual gap.

---

## 1. Introduction

McKenzie et al. (2025) introduced activation probes for detecting high-stakes LLM interactions—situations involving medical advice, legal counsel, financial decisions, or safety-critical scenarios. Their probes, trained on residual stream activations, achieve strong performance on English-majority synthetic data but show degradation on naturalistic benchmarks (Anthropic HH, ToolACE).

A practical question for deployment: **do these probes work when users speak other languages?** If the probe relies on surface-level linguistic features, it would fail. If it captures a language-invariant "stakes" concept in the model's representation space, it should transfer.

We test this with Indonesian (Bahasa Indonesia), a language with ~200 million speakers that is typologically distant from the English/French/German/Hindi mix in the training data. Llama 3.1 lists Indonesian as a Tier 2 supported language, meaning it has moderate but not extensive coverage.

**What we did:** We translated all three evaluation sets to Indonesian, extracted activations from the same Llama 3.1 8B model, and evaluated the English-trained probe without retraining.

**What we found:** Cross-lingual degradation is small (1.2% mean AUROC at best layer), and consistently smaller than the synthetic-to-real domain gap (~13%). Earlier layers generalize better across both dimensions.

---

## 2. Setup

### 2.1 Baseline Reproduction

We follow the methodology of McKenzie et al. (2025) as closely as possible:

- **Model:** Llama 3.1 8B Instruct, loaded in 8-bit quantization via bitsandbytes
- **Activations:** Residual stream at `input_layernorm` (residual entering the layer), mean-pooled over sequence length, with leading BOS token stripped
- **Probe:** StandardScaler + LogisticRegression (L2, C=1e-3, `fit_intercept=False`, LBFGS solver, max_iter=1000)
- **Training data:** 8,000 examples (4,000 high-stakes, 4,000 low-stakes) from the paper's `prompts_4x` training set, which is multilingual (~61% English, ~14% Hindi, ~13% German, ~13% French)
- **Layer truncation:** Only layers 0 through the target layer are computed (matching the paper's `HookedModel` approach)

**What differs from the paper:** We use 8-bit quantization (the paper uses fp16); we only evaluate the linear probe (the paper's best results use an attention probe on the 70B model); we do not run the 70B model. We also evaluate on 2 of the paper's 6 out-of-distribution test sets (Anthropic HH and ToolACE), and we do not fine-tune or retrain the probe on any evaluation data.

### 2.2 Evaluation Sets

| Dataset | Size (EN) | Size (ID) | Description |
|---------|-----------|-----------|-------------|
| **Synthetic test** | 2,000 | 1,950 | Balanced, single-turn, multilingual source |
| **Anthropic HH** | 2,984 | 2,959 | Multi-turn conversations, real-world queries |
| **ToolACE** | 734 | 733 | Multi-turn with function/tool calls |

The Indonesian sets are slightly smaller because some examples were excluded during translation due to API refusals — cases where Claude declined to translate the content and returned empty or unchanged text. These were filtered out to avoid contaminating the cross-lingual evaluation with English-language inputs. A small number of parsing failures in Anthropic HH account for that set's reduction.

### 2.3 Translation Methodology

**Translation model:** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`), via the Anthropic API.

**Prompt design:** The system prompt instructs the model to translate from any source language (EN/FR/DE/HI) to Indonesian, prioritizing native naturalness over literal accuracy. Key instructions:

- Use dictionary-standard Indonesian (KBBI), not colloquial slang
- Preserve tone, formality level, and emotional intensity
- Keep proper nouns, technical terms, code snippets, and URLs unchanged
- Translate idioms into Indonesian equivalents rather than literally
- For multi-turn dialogues: translate each turn independently, preserving role structure

**Quality evaluation:** Translation prompts were developed through several iterations, with each round reviewed by the author (a native Indonesian speaker) for naturalness and faithfulness. The final prompt was validated on a pilot set of 10 randomly sampled translations (5 high-stakes, 5 low-stakes), scored by Claude Sonnet 4.5 on faithfulness (meaning preservation) and naturalness (native-sounding Indonesian) on a 1–5 scale. Mean scores: ~4.7 faithfulness, ~4.7 naturalness.

**Limitation:** While translation quality was not evaluated by professional human translators, the iterative prompt design and native-speaker review provide reasonable confidence in translation adequacy for this evaluation.

---

## 3. Results

### 3.1 Sanity Checks

Before interpreting results, we verify the probe learns a real signal:

| Check | Result |
|-------|--------|
| Training set AUROC | 0.9982 |
| Random baseline AUROC | 0.5061 (chance — probe does not exploit artifacts) |
| Per-language train breakdown | EN: 0.9957, FR: 0.9985, DE: 0.9952, HI: 0.9962 |

The per-language breakdown shows the probe works equally well on all four source languages in the training data. This is consistent with a language-invariant representation being probed.

### 3.2 Layer Sweep

We evaluate the probe at layers 12, 16, 20, 26, and 31 (of 32 total) on all six test sets:

| Layer | EN Synth | EN Anth | EN Tool | ID Synth | ID Anth | ID Tool |
|-------|----------|---------|---------|----------|---------|---------|
| **12** | **0.9961** | **0.8701** | **0.8324** | **0.9942** | **0.8279** | **0.8394** |
| 16 | 0.9955 | 0.8443 | 0.8091 | 0.9932 | 0.8116 | 0.7964 |
| 20 | 0.9953 | 0.8427 | 0.7400 | 0.9919 | 0.7977 | 0.7593 |
| 26 | 0.9935 | 0.8031 | 0.7601 | 0.9883 | 0.7489 | 0.7848 |
| 31 | 0.9923 | 0.7559 | 0.7932 | 0.9795 | 0.6710 | 0.8110 |

Layer 12 is best or tied-for-best on 5 of 6 test sets (all except ID ToolACE, where L31 is marginally higher). Performance degrades monotonically at later layers for most test sets.

### 3.3 Cross-Lingual Gap

The cross-lingual gap at the best layer (L12):

| Test Set | EN AUROC | ID AUROC | Δ (ID − EN) |
|----------|----------|----------|-------------|
| Synthetic | 0.9961 | 0.9942 | −0.0019 |
| Anthropic | 0.8701 | 0.8279 | −0.0422 |
| ToolACE | 0.8324 | 0.8394 | +0.0070 |
| **Mean** | **0.8995** | **0.8872** | **−0.0124** |

The synthetic test gap is negligible (0.2%). Anthropic HH shows the largest drop (4.2%). ToolACE shows Indonesian slightly outperforming English, though the confidence interval on 734 examples is wide.

For comparison, the **domain gap** (synthetic → Anthropic, within English) is ~12.6% at L12. The cross-lingual gap is roughly 3× smaller than the domain gap.

### 3.4 Sensitivity at Low FPR

At strict operating points, the cross-lingual gap widens:

| Test Set | EN TPR@1%FPR | ID TPR@1%FPR | EN TPR@5%FPR | ID TPR@5%FPR |
|----------|--------------|--------------|--------------|--------------|
| Synthetic | 0.9047 | 0.8528 | 0.9845 | 0.9762 |
| Anthropic | 0.1796 | 0.1478 | 0.4450 | 0.3467 |
| ToolACE | 0.2997 | 0.2888 | 0.4223 | 0.4905 |

TPR@1%FPR drops more for Indonesian, especially on Anthropic HH. This suggests the score distributions shift slightly, making the decision boundary less calibrated for the target language.

### 3.5 Error Analysis

At layer 12 on the English synthetic test set:

- **Total errors:** 55 / 2,000 (2.8%)
- **False positives:** 36 (predicted high-stakes, actually low)
- **False negatives:** 19 (predicted low-stakes, actually high)

On English Anthropic HH:

- **Total errors:** 691 / 2,984 (23.2%)
- **False positives:** 219
- **False negatives:** 472

Qualitative patterns in false positives: the probe over-triggers on topics that *sound* serious but are benign queries (e.g., "How come the US has so many prisoners?" labeled low-stakes but scored P=0.998; "Why is there no cure for cancer yet?" at P=0.980). These examples are arguably ambiguous even to human annotators.

False negatives tend to be politely phrased high-stakes requests where the severity is implicit rather than signaled by urgent language (e.g., a gentle patient safety correction scored P=0.140).

---

## 4. Discussion

### Why does cross-lingual transfer work?

The simplest explanation: by layer 12 of a multilingual model, the residual stream has already mapped inputs from different languages into a shared semantic space. A linear probe's decision boundary in this space separates "high-stakes" from "low-stakes" regardless of the surface language. Wendler et al. (2024) provide evidence for this: they show that in multilingual LLMs, middle-layer representations converge to a shared semantic space across languages, with language-specific encoding concentrated in the first and last layers.

### Why earlier layers?

Performance degrades at later layers for both English and Indonesian. Later layers specialize for token prediction (next-token logits in the model's output vocabulary), which is language-specific. The "stakes" concept appears to be a mid-level semantic feature that is most accessible before the model begins committing to language-specific output representations.

### The real problem: synthetic-to-real, not English-to-Indonesian

The domain gap from synthetic test data to naturalistic benchmarks (~13% for English, ~16% for Indonesian at L12) is substantially larger than the cross-lingual gap (~1.2% mean). This suggests that improving probe robustness to distribution shift (different question styles, multi-turn dialogue, implicit stakes) would yield more practical benefit than addressing cross-lingual transfer.

### Caveats

- **Single model:** We only test Llama 3.1 8B. The cross-lingual gap could be larger on models with less Indonesian training data.
- **Single language pair:** Indonesian is typologically distant from the training languages but uses Latin script. Languages with non-Latin scripts (Arabic, Chinese, Thai) may transfer differently.
- **8-bit quantization:** We use 8-bit rather than the paper's fp16. This may introduce small numerical differences.
- **No repeated runs:** We do not report confidence intervals from multiple random seeds. The probe is deterministic given the same data ordering, but activation extraction involves floating-point non-determinism.
- **Translation artifacts:** Some performance differences may reflect translation quality rather than genuine linguistic difficulty.

---

## 5. Planned Future Work: SAE-Based Failure Analysis

We have designed (but not yet executed) an analysis using Sparse Autoencoders (SAEs) from the Llama Scope project to understand *why* specific examples are misclassified. The planned methodology:

1. Load pre-trained SAEs for Llama 3.1 8B at layer 12
2. For correctly classified and misclassified examples, decompose activations into SAE feature vectors
3. Identify features that are differentially active in failure cases vs. successes
4. Use Neuronpedia's automated interpretations to label these features
5. Test whether SAE-based probes (trained on feature activations rather than raw residual stream) generalize differently

This analysis could reveal whether probe failures stem from specific feature types (e.g., topic features that correlate with but don't determine stakes) and whether cross-lingual failures activate different feature patterns than within-language failures.

---

## 6. Limitations

- We test only the linear probe (logistic regression). The paper's attention probe achieves higher mean AUROC (exceeding 0.95 on development sets; McKenzie et al. 2025, Figure 3), and its cross-lingual robustness remains untested.
- We evaluate zero-shot transfer only (English-trained probe on Indonesian). Mixed-language training — combining English and Indonesian examples — could reveal whether adding target-language data improves calibration at strict operating points, where we observe the largest cross-lingual gap.
- Results are for a single English–Indonesian language pair. The probe's behavior may differ for languages with non-Latin scripts or less representation in the model's training data.
- Translation quality was iteratively refined and reviewed by a native speaker, but not evaluated by professional translators. Some performance gaps may reflect translation artifacts rather than genuine cross-lingual difficulty.
- We do not compare against other cross-lingual transfer baselines (e.g., translate-train, multilingual fine-tuning).

---

## 7. Conclusion

An English-trained linear probe for high-stakes detection transfers to Indonesian with small AUROC degradation (1.2% mean) on Llama 3.1 8B. The cross-lingual gap is consistently smaller than the synthetic-to-real domain gap, suggesting that language is not the primary axis of fragility for these probes. Earlier layers (layer 12 of 32) yield the best transfer. At strict operating points (TPR@1%FPR), the gap widens, indicating that while the probe's ranking of examples transfers well, its calibration does not.

For practical deployment, these results are cautiously encouraging: a probe trained on English-majority data appears to capture a language-invariant concept. However, the larger concern remains the probe's difficulty with naturalistic evaluation benchmarks regardless of language, where 23% of examples are misclassified even in English.

---

## References

- McKenzie, A., Pawar, U., Blandfort, P., Bankes, W., Krueger, D., Lubana, E. S., & Krasheninnikov, D. (2025). Detecting High-Stakes Interactions with Activation Probes. *arXiv:2506.10805*. DOI: [10.48550/arXiv.2506.10805](https://doi.org/10.48550/arXiv.2506.10805)
- Grattafiori, A., Dubey, A., et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*. DOI: [10.48550/arXiv.2407.21783](https://doi.org/10.48550/arXiv.2407.21783)
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *arXiv:2402.10588*. DOI: [10.48550/arXiv.2402.10588](https://doi.org/10.48550/arXiv.2402.10588)

---

## Appendix: Reproducibility Details

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 8-bit (bitsandbytes, `llm_int8_threshold=6.0`) |
| Probe | `StandardScaler` + `LogisticRegression(C=1e-3, solver='lbfgs', max_iter=1000, fit_intercept=False)` |
| Activation hook | `model.model.layers[layer_idx].input_layernorm` |
| Pooling | Mean pool over sequence length, respecting attention mask |
| Max sequence length | 8,192 tokens |
| BOS handling | Leading BOS token stripped after tokenization |
| Training set | 8,000 examples (`prompts_4x/train.jsonl`) |
| Random seed | 42 (probe), floating-point non-determinism in activations |
| Translation model | `claude-sonnet-4-5-20250929` |
| Translation prompt | Native-naturalness, KBBI-standard, preserve tone/formality |
| Compute | Lambda Labs GPU instance (specific GPU type not recorded) |
