# Cross-Lingual and Cross-Model Robustness of High-Stakes Activation Probes

**Technical Report — BlueDot AI Safety Course, 2026**

> **Disclaimer:** This report was drafted with Claude Opus 4.6 and verified against the codebase, notebook outputs, and cached results. All quantitative claims reference specific output files in the project repository.

---

## Abstract

Activation probes can detect high-stakes interactions in language models, but two questions matter for deployment: do they work when users speak other languages, and do they transfer across model architectures? We test both by translating the evaluation datasets from McKenzie et al. (2025) into Indonesian and evaluating English-trained linear probes on two models: Llama 3.1 8B and Gemma 3 12B.

On Llama 3.1 8B (layer 12), the probe transfers to Indonesian with only 0.6% mean AUROC degradation (0.9359 to 0.9295, 10-seed mean with 95% bootstrap CIs). On Gemma 3 12B (layer 32), the cross-lingual gap is 0.7% (0.9310 to 0.9241), though cross-lingual agreement analysis reveals that Gemma's failures are less consistent across languages (56.8% both-correct on Anthropic vs Llama's 73.1%).

We then use Sparse Autoencoders to investigate why probes fail. On Llama, differential SAE analysis reveals diagnostic features related to technical infrastructure, domain knowledge, and financial content that are enriched in error cases. On Gemma, the same analysis yields nothing -- all SAE features are universally active with near-zero differential, suggesting the SAE decomposition at the analyzed layer does not capture failure-relevant structure.

The cross-lingual gap (0.6--0.7%) is consistently smaller than the synthetic-to-real domain gap (~6--8%), suggesting that language is not the primary axis of fragility for these probes.

---

## 1. Introduction

McKenzie et al. (2025) introduced activation probes for detecting high-stakes LLM interactions -- situations involving medical advice, legal counsel, financial decisions, or safety-critical scenarios. Their probes, trained on residual stream activations, achieve strong performance on synthetic data but show degradation on naturalistic benchmarks (Anthropic HH, ToolACE).

Two practical questions for deployment remain open:

1. **Cross-lingual robustness:** Do these probes work when users speak languages not well-represented in the training data?
2. **Cross-model robustness:** Does the same methodology produce comparable probes across different model families?

If the probe relies on surface-level linguistic features, it would fail on new languages. If it captures a language-invariant "stakes" concept in the model's representation space, it should transfer. Similarly, if the probe methodology is architecture-specific, it won't scale.

We test with Indonesian (Bahasa Indonesia), a language with ~200 million speakers that is typologically distant from the English/French/German/Hindi mix in the training data. Llama 3.1 lists Indonesian as a Tier 2 supported language; Gemma 3 includes Indonesian in its broader multilingual training. We evaluate on two model families: Llama 3.1 8B (32 layers) and Gemma 3 12B (48 layers).

**What we did:** We translated all three evaluation sets to Indonesian, extracted activations from both models, and evaluated the English-trained probes without retraining. We then used Sparse Autoencoders (Llama Scope and Gemma Scope 2) to investigate why probes fail on specific examples.

**What we found:**

- Cross-lingual degradation is small: 0.6% mean AUROC on Llama, 0.7% on Gemma.
- Earlier-to-middle layers generalize best on both models.
- The synthetic-to-real domain gap is ~10x larger than the cross-lingual gap.
- SAE failure analysis is informative on Llama (differential features identify error-enriched concepts) but uninformative on Gemma (features are universally active, non-diagnostic).

---

## 2. Setup

### 2.1 Baseline Reproduction

We follow the methodology of McKenzie et al. (2025):

- **Models:** Llama 3.1 8B Instruct and Gemma 3 12B IT, loaded in 8-bit quantization via bitsandbytes
- **Activations:** Residual stream at `input_layernorm` (residual entering the layer), mean-pooled over sequence length, with leading BOS token stripped
- **Probe:** StandardScaler + LogisticRegression (L2, C=1e-3, `fit_intercept=False`, LBFGS solver, max_iter=1000)
- **Training data:** 8,000 examples (4,000 high-stakes, 4,000 low-stakes) from the paper's `prompts_4x` training set, which is multilingual (~61% English, ~14% Hindi, ~13% German, ~13% French)
- **Seeds:** 10 random seeds (master seed 20250214), results reported as means with 95% bootstrap confidence intervals
- **Layer truncation:** Only layers 0 through the target layer are computed (matching the paper's `HookedModel` approach)

**What differs from the paper:** We use 8-bit quantization (the paper uses fp16); we only evaluate the linear probe (the paper's best results use an attention probe on the 70B model); we add Gemma 3 12B as a second model family. We evaluate on 2 of the paper's 6 out-of-distribution test sets (Anthropic HH and ToolACE).

### 2.2 Evaluation Sets

| Dataset | Size (Orig) | Size (ID) | Description |
|---------|-------------|-----------|-------------|
| **Synthetic test** | 2,000 | 1,950 | Balanced, single-turn, multilingual source |
| **Anthropic HH** | 2,984 | 2,959 | Multi-turn conversations, real-world queries |
| **ToolACE** | 734 | 733 | Multi-turn with function/tool calls |

Labels for Anthropic HH and ToolACE were assigned by GPT-4o on a 1--10 scale (as described in the original paper). The Indonesian sets are slightly smaller because some examples were excluded during translation due to API refusals or parsing failures.

We refer to the original-language datasets as "Original" rather than "English" because the training and synthetic test data are multilingual (~61% EN, ~14% HI, ~13% DE, ~13% FR).

### 2.3 Translation Methodology

**Translation model:** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`), via the Anthropic API.

**Prompt design:** The system prompt instructs the model to translate from any source language (EN/FR/DE/HI) to Indonesian, prioritizing native naturalness over literal accuracy. Key instructions:

- Use dictionary-standard Indonesian (KBBI), not colloquial slang
- Preserve tone, formality level, and emotional intensity
- Keep proper nouns, technical terms, code snippets, and URLs unchanged
- Translate idioms into Indonesian equivalents rather than literally
- For multi-turn dialogues: translate each turn independently, preserving role structure

**Quality evaluation:** Translation prompts were developed through several iterations, with each round reviewed by the author (a native Indonesian speaker) for naturalness and faithfulness. The final prompt was validated on a pilot set of 10 randomly sampled translations, scored by Claude Sonnet 4.5 on faithfulness and naturalness on a 1--5 scale. Mean scores: ~4.7 faithfulness, ~4.7 naturalness.

### 2.4 Model Details

| Property | Llama 3.1 8B | Gemma 3 12B |
|----------|-------------|-------------|
| Architecture | Decoder-only | Decoder-only |
| Layers | 32 | 48 |
| Hidden dim | 4,096 | 3,840 |
| Layer access | `model.model.layers` | `model.language_model.layers` |
| ID support | Tier 2 | Multilingual training |
| Quantization | 8-bit (bitsandbytes) | 8-bit (bitsandbytes) |

---

## 3. Results

### 3.1 Sanity Checks

Before interpreting results, we verify the probes learn a real signal:

**Llama 3.1 8B (Layer 12):**

| Check | Result |
|-------|--------|
| Training set AUROC | 0.9997 |
| Random baseline AUROC | 0.4441 (chance -- probe does not exploit artifacts) |
| Per-language train breakdown | EN: 0.9980, FR: 0.9995, DE: 0.9999, HI: 0.9978 |

The per-language breakdown shows the probe works equally well on all four source languages in the training data, consistent with a language-invariant representation being probed.

### 3.2 Layer Sweep

We evaluate probes at multiple layers on all six test sets (Original and Indonesian versions of Synthetic, Anthropic, and ToolACE).

**Llama 3.1 8B** (layers 12, 16, 20, 26, 31 of 32):

| Layer | Orig Synth | Orig Anth | Orig Tool | ID Synth | ID Anth | ID Tool |
|-------|-----------|-----------|-----------|----------|---------|---------|
| **12** | **0.9985** | **0.9358** | **0.8734** | **0.9970** | **0.9044** | **0.8870** |
| 16 | 0.9981 | 0.9077 | 0.8773 | 0.9964 | 0.8780 | 0.8707 |
| 20 | 0.9974 | 0.8992 | 0.8540 | 0.9952 | 0.8696 | 0.8607 |
| 26 | 0.9960 | 0.8649 | 0.8363 | 0.9910 | 0.8236 | 0.8543 |
| 31 | 0.9944 | 0.8570 | 0.8528 | 0.9852 | 0.7593 | 0.8417 |

**Gemma 3 12B** (layers 8, 16, 24, 31, 32, 41, 47 of 48):

| Layer | Orig Synth | Orig Anth | Orig Tool | ID Synth | ID Anth | ID Tool |
|-------|-----------|-----------|-----------|----------|---------|---------|
| 8  | 0.9941 | 0.7872 | 0.7850 | 0.9898 | 0.7836 | 0.7838 |
| 16 | 0.9988 | 0.9065 | 0.8568 | 0.9977 | 0.8984 | 0.8665 |
| 24 | 0.9990 | 0.9342 | 0.8531 | 0.9982 | 0.9246 | 0.8389 |
| 31 | 0.9987 | 0.9209 | 0.8542 | 0.9976 | 0.9158 | 0.8452 |
| **32** | **0.9987** | **0.9238** | **0.8706** | **0.9975** | **0.9176** | **0.8571** |
| 41 | 0.9979 | 0.9055 | 0.8751 | 0.9953 | 0.8700 | 0.8594 |
| 47 | 0.9976 | 0.8832 | 0.8752 | 0.9943 | 0.8811 | 0.8515 |

Layer 12 is best for Llama (mean AUROC 0.9359 across Original datasets); layer 32 is best for Gemma (mean AUROC 0.9310). Both models peak at roughly 38--67% network depth.

Performance degrades at the final layers for both models, consistent with later layers specializing for language-specific token prediction rather than semantic classification.

### 3.3 Cross-Lingual Gap

At the best layer for each model:

**Llama 3.1 8B (Layer 12):**

| Test Set | Original | Indonesian | Delta |
|----------|----------|------------|-------|
| Synthetic | 0.9985 | 0.9970 | -0.0015 |
| Anthropic | 0.9358 | 0.9044 | -0.0314 |
| ToolACE | 0.8734 | 0.8870 | +0.0136 |
| **Mean** | **0.9359** | **0.9295** | **-0.0064** |

**Gemma 3 12B (Layer 32):**

| Test Set | Original | Indonesian | Delta |
|----------|----------|------------|-------|
| Synthetic | 0.9987 | 0.9975 | -0.0012 |
| Anthropic | 0.9238 | 0.9176 | -0.0062 |
| ToolACE | 0.8706 | 0.8571 | -0.0135 |
| **Mean** | **0.9310** | **0.9241** | **-0.0070** |

Both models show near-zero synthetic gaps and moderate gaps on the naturalistic benchmarks. The cross-lingual gap is comparable: 0.6% mean for Llama, 0.7% mean for Gemma. Both are small compared to the domain gap.

For comparison, the **domain gap** (Synthetic to Anthropic, within Original) is 6.3% for Llama and 7.5% for Gemma. The cross-lingual gap is roughly 10x smaller than the domain gap in both cases.

### 3.4 Cross-Lingual Agreement Analysis

To understand whether models fail consistently across languages, we examine per-example agreement on Anthropic HH (the dataset with the most errors):

| Category | Llama (L12) | Gemma (L32) |
|----------|-------------|-------------|
| Both correct | 73.1% | 56.8% |
| Both wrong | 18.6% | 27.6% |
| Original correct, ID wrong | 5.0% | 14.3% |
| ID correct, Original wrong | 3.3% | 1.3% |

Llama shows high agreement: when it fails, it almost always fails in both languages (73% + 19% = 92% consistent). Gemma is less consistent: 14.3% of examples are correct in the original language but fail in Indonesian, suggesting Gemma's cross-lingual representations are less aligned at the probe level.

The asymmetry is also informative. Both models have more "Original-correct, ID-wrong" cases than the reverse, meaning Indonesian-specific failures dominate over Original-specific ones. This is expected: the probe was trained on multilingual (Original-language) data and has never seen Indonesian.

### 3.5 Error Analysis

Error counts at the best layer for each model (single seed 0, threshold 0.5):

**Llama 3.1 8B (Layer 12):**

| Dataset | Total Errors | Error Rate | FP | FN |
|---------|-------------|-----------|-----|-----|
| Synthetic | 41 / 2,000 | 2.1% | 17 | 24 |
| Anthropic | 651 / 2,984 | 21.8% | 51 | 600 |
| ToolACE | 344 / 734 | 46.9% | 0 | 344 |

**Gemma 3 12B (Layer 32):**

| Dataset | Total Errors | Error Rate | FP | FN |
|---------|-------------|-----------|-----|-----|
| Synthetic | 36 / 2,000 | 1.8% | 10 | 26 |
| Anthropic | 589 / 2,984 | 19.7% | 91 | 498 |
| ToolACE | 309 / 734 | 42.1% | 1 | 308 |

Several patterns stand out:

**ToolACE is almost entirely false negatives.** Both models produce near-zero false positives on ToolACE but miss 43--47% of high-stakes examples. ToolACE examples involve tool/function calls (tornado alerts, infrastructure deletion, stock trading) where the high-stakes nature comes from the *consequence* of the action, not from explicit danger language. The probe, trained on conversational data, lacks this concept of "consequential tool use."

**Anthropic HH is dominated by false negatives.** On both models, FNs outnumber FPs by 10--12x on Anthropic. Many of these are multi-turn conversations where early turns look innocent but later turns escalate -- matching the paper's "context mixing" failure mode. The mean-pooling aggregation dilutes the late-appearing high-stakes signal.

**Label quality matters.** Manual inspection of Anthropic false negatives reveals that many are arguably label disagreements. GPT-4o's 1--10 labeling captures the full conversation trajectory (correctly identifying that a conversation about ice cream toppings eventually escalates to harmful content), while the probe only sees the aggregated activation. Some "errors" reflect genuine ambiguity about whether a conversation is high-stakes.

---

## 4. SAE Failure Analysis

### 4.1 Methodology

To understand why probes fail on specific examples, we decompose model activations using pre-trained Sparse Autoencoders (SAEs):

- **Llama:** Llama Scope SAE at layer 12, residual position, 32K features (8x expansion). Model: `fnlp/Llama3_1-8B-Base-L12R-8x`.
- **Gemma:** Gemma Scope 2 SAE at layer 41, 16K features. Model: `google/gemma-scope-2-12b-it`.

The analysis pipeline:
1. Run the model forward, hooking into `input_layernorm` at the target layer
2. Encode each token's activation through the SAE (per-token, not mean-pooled -- SAEs are trained on individual token activations)
3. Max-pool across the sequence to preserve the strongest activation of each feature
4. Compare feature prevalence between error cases and correct predictions

**Why per-token encoding:** SAEs are trained on individual token activations. Mean-pooling before encoding produces an average vector that the SAE was never trained to decompose, resulting in poor sparsity and uninterpretable features.

**Why max-pooling after:** Max-pooling preserves the strongest activation of each feature across all tokens. If any token in the sequence strongly activates a "danger" feature, that signal is kept. Mean-pooling would dilute it -- exactly the context mixing problem the paper describes.

**Why differential analysis:** Naive top-k frequency analysis (which features appear most often in errors) is confounded by universally active features. Some features appear in nearly every example regardless of correctness. Differential analysis (`error_prevalence - correct_prevalence`) controls for this, isolating features specifically enriched in error cases.

### 4.2 Llama Results

**Sparsity:** L0 ~ 111--205 active features per example (out of 32K). This is sparse enough to be interpretable.

**Naive top-k analysis (confounded):** The most frequent features in error cases -- 604, 17722, 19230 -- appear in 100% of errors AND 100% of correct predictions. They represent generic conversational concepts (communication, emotional reactions) and tell us nothing about failure.

**Differential analysis (informative):**

False negative enriched features (high-stakes examples the probe misses):

| Feature | Differential | Neuronpedia Description |
|---------|-------------|------------------------|
| 6989 | +15.3% | Technical/infrastructure concepts |
| 16646 | +14.3% | Knowledge and information |
| 9766 | +7.9% | Events and happenings |

False positive enriched features (low-stakes examples the probe flags):

| Feature | Differential | Neuronpedia Description |
|---------|-------------|------------------------|
| 2035 | +21.7% | Financial terminology |
| 6672 | +13.2% | Historical references |

These make sense: technical infrastructure conversations (deploying servers, managing databases) involve consequential actions but don't use danger language, causing false negatives. Financial and historical topics sound serious to the probe but are often informational queries, causing false positives.

**Signal word analysis:** We searched for classic trigger words ("emergency," "urgent," "danger," "kill," "hack") in false positive examples. None appeared, suggesting the Llama probe's false positives are not caused by simple keyword triggering at this layer (unlike the attention probe analyzed in the paper).

### 4.3 Gemma Results

**Sparsity:** L0 ~ 1,614 active features per example (out of 16K). This is an order of magnitude less sparse than Llama.

**Naive top-k analysis:** All top-10 features are universally active, as with Llama.

**Differential analysis:** Unlike Llama, differential analysis reveals nothing useful. The top features show <0.1% difference between error and correct prevalence. Every feature that appears frequently in errors appears equally frequently in correct predictions.

The features themselves are multilingual token-level features (subword patterns, punctuation, formatting) rather than semantic concepts. At layer 41, the Gemma Scope 2 SAE appears to decompose activations into token-identity features rather than the kind of semantic features we see in Llama Scope at layer 12.

**Why the difference?** Two factors likely contribute:

1. **Sparsity:** L0 of 1,614 means ~10% of all features are active per example. The SAE is not achieving meaningful decomposition -- it spreads activation across too many features, each carrying little specific information.
2. **Layer choice:** Layer 41 of 48 is deep in Gemma's architecture. At this depth, the SAE may be capturing output-preparation features rather than the mid-level semantic concepts that would be relevant to failure analysis. Llama's analysis at layer 12 (37.5% depth) captures more abstract representations.

### 4.4 Cross-Model SAE Comparison

| Property | Llama (L12, 32K) | Gemma (L41, 16K) |
|----------|-------------------|-------------------|
| L0 (sparsity) | ~111--205 | ~1,614 |
| Naive top-k | Confounded (universally active) | Confounded (universally active) |
| Differential analysis | Informative (15%+ differentials) | Uninformative (<0.1% differentials) |
| Feature type | Semantic concepts | Token/subword patterns |
| Diagnostic of failure | Yes | No |

The contrast is stark. Llama Scope at layer 12 decomposes activations into features that meaningfully distinguish error cases from correct predictions. Gemma Scope 2 at layer 41 does not. This could be an artifact of the specific layer and SAE width, not a fundamental limitation -- analyzing Gemma at an earlier layer (e.g., layer 24 or 32) with a wider SAE might yield better results.

---

## 5. Discussion

### Why does cross-lingual transfer work?

The simplest explanation: by layer 12 of a multilingual model, the residual stream has already mapped inputs from different languages into a shared semantic space. A linear probe's decision boundary in this space separates "high-stakes" from "low-stakes" regardless of the surface language. Wendler et al. (2024) provide evidence for this: in multilingual LLMs, middle-layer representations converge to a shared semantic space across languages, with language-specific encoding concentrated in the first and last layers.

### Why earlier layers?

Performance degrades at later layers for both models. Later layers specialize for token prediction (next-token logits in the model's output vocabulary), which is language-specific. The "stakes" concept appears to be a mid-level semantic feature that is most accessible before the model begins committing to language-specific output representations.

Both models confirm this pattern, though Gemma's optimal layer is proportionally deeper (67% vs 38%). This may reflect architectural differences in how the two model families distribute semantic processing across their layer stacks.

### The real problem: synthetic-to-real, not cross-lingual

The domain gap from synthetic test data to naturalistic benchmarks is substantially larger than the cross-lingual gap in both models:

| Model | Cross-lingual gap (mean) | Domain gap (Synth to Anth, Original) |
|-------|-------------------------|--------------------------------------|
| Llama | 0.6% | 6.3% |
| Gemma | 0.7% | 7.5% |

Improving probe robustness to distribution shift (multi-turn dialogue, implicit stakes, tool-use contexts) would yield more practical benefit than addressing cross-lingual transfer.

### Cross-model consistency

The two models tell a consistent story: cross-lingual transfer works, earlier-to-middle layers are best, and ToolACE is hard. But they differ in interesting ways:

- **Llama has slightly higher absolute performance** (0.936 vs 0.931 mean AUROC on Original datasets at best layer), despite being a smaller model. This may reflect training data composition or the specific layer choice.
- **Gemma has more cross-lingual inconsistency** (14.3% of Anthropic examples correct in Original but wrong in Indonesian, vs Llama's 5.0%). Gemma's representations may be less aligned across languages at the probe level, even though the aggregate AUROC gap is similar.
- **SAE analysis works on Llama but not Gemma.** This is likely layer/SAE-specific rather than fundamental, but it underscores that SAE-based interpretability results depend heavily on the quality and characteristics of the available SAEs.

---

## 6. Limitations

- **Two models, one language pair.** We test Llama 3.1 8B and Gemma 3 12B with Indonesian only. The cross-lingual gap could be larger on models with less Indonesian training data or for languages with non-Latin scripts (Arabic, Chinese, Thai).
- **Linear probe only.** The paper's attention probe achieves higher AUROC (>0.95 on development sets with the 70B model). Its cross-lingual robustness remains untested.
- **8-bit quantization.** We use 8-bit rather than the paper's fp16. This may introduce small numerical differences, compounded with the Base-to-Instruct SAE mismatch.
- **GPT-4o labels.** Anthropic HH and ToolACE labels were assigned by GPT-4o, not human annotators. Some "errors" may be label disagreements rather than genuine probe failures. This particularly affects Anthropic HH, where multi-turn conversation labeling is inherently ambiguous.
- **Translation artifacts.** Some performance differences may reflect translation quality rather than genuine linguistic difficulty. Translation was reviewed by a native speaker but not by professional translators.
- **SAE layer sensitivity.** The Gemma SAE analysis at layer 41 was uninformative, but analyzing at layer 32 (the best probe layer) might yield different results. Our negative result should not be interpreted as "SAE failure analysis doesn't work on Gemma" -- it means "it didn't work at this specific layer with this specific SAE."
- **Zero-shot transfer only.** We do not test mixed-language training (adding Indonesian examples to the training set), which could reveal whether the probe needs calibration for new languages.
- **No confidence intervals on error analysis.** While AUROC results are 10-seed means with bootstrap CIs, error counts are from a single seed (seed 0) and should be interpreted as representative, not precise.

---

## 7. Future Work

Several directions could strengthen and extend these findings:

- **Iterative synthetic data generation.** The current probe is trained on a fixed synthetic dataset. A natural extension would be an active learning loop: evaluate the probe, identify failure patterns, generate synthetic examples targeting those patterns, and retrain. This could address the synthetic-to-real gap by making the training distribution closer to the kinds of examples the probe encounters in deployment.
- **Evaluation-attached probes.** Rather than training probes on pre-labeled data, one could attach probe evaluation directly to the labeling process -- using probe confidence to flag ambiguous cases for human review, or training on continuously updated evaluation sets.
- **Aggregation strategy.** Mean-pooling is a known weakness for multi-turn conversations where stakes escalate late. Alternative aggregation strategies (max-pooling, attention-weighted pooling, or per-turn probes with aggregation) could improve performance on the Anthropic HH failure mode. The choice between balance-preserving and semantic aggregation deserves systematic study.
- **Mixed-language training.** Adding target-language examples to the training set could improve calibration, especially at strict operating points where we observe the largest cross-lingual gaps.
- **More languages and scripts.** Testing non-Latin script languages (Arabic, Chinese, Thai) would test whether the Latin-script similarity between Indonesian and the training languages artificially inflates transfer performance.
- **Gemma SAE at earlier layers.** Analyzing Gemma with SAEs at layer 24 or 32 (where probe performance is strongest) and with wider feature dictionaries could yield the kind of informative decomposition we see with Llama.

---

## 8. Conclusion

An English-trained linear probe for high-stakes detection transfers across both languages and model families with modest degradation. On Llama 3.1 8B, the cross-lingual gap to Indonesian is 0.6% mean AUROC (10-seed mean). On Gemma 3 12B, the gap is 0.7%. Both are substantially smaller than the synthetic-to-real domain gap (6--8%), confirming that language is not the primary axis of fragility for these probes.

Earlier-to-middle layers yield the best transfer in both models. Llama's representations are more cross-lingually consistent (73% per-example agreement on Anthropic vs Gemma's 57%), and its SAE decomposition reveals interpretable failure patterns: technical infrastructure and financial terminology confuse the probe in predictable ways. Gemma's SAE analysis at layer 41 was uninformative, highlighting that SAE-based interpretability depends heavily on layer choice and SAE quality.

The dominant failure mode across both models is not cross-lingual but cross-domain: ToolACE's consequential tool-use concept and Anthropic HH's multi-turn escalation patterns are fundamentally different from the conversational training distribution. For practical deployment, improving the probe's coverage of these distribution shifts matters more than multilingual adaptation.

---

## References

- McKenzie, A., Pawar, U., Blandfort, P., Bankes, W., Krueger, D., Lubana, E. S., & Krasheninnikov, D. (2025). Detecting High-Stakes Interactions with Activation Probes. *arXiv:2506.10805*. DOI: [10.48550/arXiv.2506.10805](https://doi.org/10.48550/arXiv.2506.10805)
- Grattafiori, A., Dubey, A., et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*. DOI: [10.48550/arXiv.2407.21783](https://doi.org/10.48550/arXiv.2407.21783)
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *arXiv:2402.10588*. DOI: [10.48550/arXiv.2402.10588](https://doi.org/10.48550/arXiv.2402.10588)
- Liao, Z., et al. (2024). Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders. *arXiv:2410.20526*. DOI: [10.48550/arXiv.2410.20526](https://doi.org/10.48550/arXiv.2410.20526)
- Google DeepMind. (2025). Gemma Scope 2: Helping the AI Safety Community Deepen Understanding of Complex Language Model Behavior. [Blog post](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/).
- Gemma Team. (2025). Gemma 3 Technical Report. *arXiv:2503.19786*. DOI: [10.48550/arXiv.2503.19786](https://doi.org/10.48550/arXiv.2503.19786)

---

## Appendix A: Reproducibility Details

| Parameter | Llama 3.1 8B | Gemma 3 12B |
|-----------|-------------|-------------|
| Model ID | `meta-llama/Llama-3.1-8B-Instruct` | `google/gemma-3-12b-it` |
| Quantization | 8-bit (bitsandbytes) | 8-bit (bitsandbytes) |
| Best probe layer | 12 | 32 |
| Hidden dim | 4,096 | 3,840 |
| Activation hook | `model.model.layers[L].input_layernorm` | `model.language_model.layers[L].input_layernorm` |
| Pooling | Mean pool over sequence, attention mask | Mean pool over sequence, attention mask |
| Max sequence length | 8,192 tokens | 8,192 tokens |
| BOS handling | Strip leading BOS | Strip leading BOS |
| Probe | `StandardScaler` + `LogisticRegression(C=1e-3, solver='lbfgs', max_iter=1000, fit_intercept=False)` | Same |
| Training set | 8,000 examples (`prompts_4x/train.jsonl`) | Same |
| Seeds | 10 (master seed 20250214) | Same |
| SAE (failure analysis) | `fnlp/Llama-Scope` L12R-8x (32K features) | `google/gemma-scope-2-12b-it` L41 (16K features) |
| Translation model | `claude-sonnet-4-5-20250929` | N/A (same translations) |
| Compute | Lambda Labs GPU (A10 24GB) | Lambda Labs GPU (A10 24GB) |

## Appendix B: Layer Sweep Selection

**Llama 3.1 8B:** Layers [12, 16, 20, 26, 31] were chosen to span early-middle through final, with denser sampling in the range the paper identified as most promising.

**Gemma 3 12B:** Layers [8, 16, 24, 31, 32, 41, 47] were chosen to cover the proportionally equivalent range across 48 layers, with extra density around the expected optimum.
