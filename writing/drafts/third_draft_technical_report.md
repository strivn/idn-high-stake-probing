# Cross-Lingual Robustness of High-Stakes Activation Probes

**Technical Report — BlueDot AI Safety Course, 2026**

> **Disclaimer:** This report was drafted with Claude Opus 4.6

---

## Abstract

McKenzie et al. (2025) trained linear probes on language model activations to detect high-stakes interactions. We test whether these probes generalize to a language absent from training by translating the evaluation datasets into Indonesian and evaluating without retraining. On Llama 3.1 8B (layer 12), the probe's mean AUROC on out-of-distribution benchmarks drops from 0.9046 to 0.8957 on Indonesian (0.9% degradation). We repeated the evaluation on Gemma 3 12B (layer 32) and observed a similar pattern: 0.8972 to 0.8873 (1.0% degradation). We then attempted to use Sparse Autoencoders to identify which features are enriched in the probe's error cases. On Llama, differential analysis identified a small number of features with 10-15% higher prevalence in errors than in correct predictions, but these features provide limited explanatory power. On Gemma, the SAE at the analyzed layer produced no diagnostic signal. The cross-lingual degradation on these two models is small relative to the out-of-distribution gap between synthetic training data and naturalistic benchmarks (6-13%), which remains the more pressing concern for deployment.

## Key Interesting Findings for Future Work

> 1. Llama model show better performance at lower layers; while Gemma model show better performance at mid to later layers (see layer sweep figures)
> 2. Gemma probe shows more disagreement between Original - Indonesian, while Llama has more agreement (# of same predictions). Gemma 15.6%; Llama 8.6%
> 3. No SAE features were discernible or indicates any possibility of identifying failures (also from raw activations - PCA, UMAP, etc don't show clear separation)
> 4. There are more FN than FP for both evaluation sets (Anthropic, ToolAce). Failure modes persist. Though the probe training set is different from the original paper. Probe failed in ToolAce (which on their paper was also shown to have relatively lower performance compared with other eval sets).
> 5. Expanding on the idea for further dataset generation that stress test the probes.

---

## 1. Introduction

McKenzie et al. (2025) introduced activation probes for detecting high-stakes LLM interactions — situations involving medical advice, legal counsel, financial decisions, or safety-critical scenarios. Their linear and attention probes, trained on residual stream activations, achieve strong performance on synthetic evaluation data but show degradation on naturalistic out-of-distribution benchmarks (Anthropic HH, ToolACE).

One practical question for deployment: do these probes work when users speak languages not represented in the training data? If the probe relies on surface-level linguistic features, it would fail on new languages. If it captures a language-invariant concept in the model's representation space, it should transfer.

We test with Indonesian (Bahasa Indonesia), a language with approximately 200 million speakers that is typologically distant from the English/French/German/Hindi mix in the training data. Llama 3.1 8B (Grattafiori et al., 2024) lists Indonesian as a Tier 2 supported language; Gemma 3 12B (Gemma Team, 2025) includes Indonesian in its broader multilingual training. We evaluate the probes on both models to check whether findings hold across model families.

---

## 2. Setup

### 2.1 Methodology

We follow McKenzie et al. (2025):

- **Activations:** Residual stream at `input_layernorm` (the residual entering the layer, before layer norm), mean-pooled over sequence length, with leading BOS token stripped. Max sequence length: 8,192 tokens.
- **Probe:** StandardScaler + LogisticRegression (L2, C=1e-3, `fit_intercept=False`, LBFGS solver, max_iter=1000). These are the paper's exact hyperparameters for their sklearn baseline.
- **Training data:** 8,000 examples (4,000 high-stakes, 4,000 low-stakes) from the paper's `prompts_4x` training set, which is multilingual (~61% English, ~14% Hindi, ~13% German, ~13% French).
- **Seeds:** 10 random seeds (master seed 20250214), results reported as means with 95% bootstrap confidence intervals. The L2-regularized LBFGS solver converges to the same optimum regardless of seed (0.000 standard deviation across all 10 seeds), so the multi-seed setup serves as a sanity check rather than a variance estimate.
- **Models:** Llama 3.1 8B Instruct and Gemma 3 12B IT, loaded in 8-bit quantization via bitsandbytes (Dettmers et al., 2022). The paper uses fp16; we use 8-bit to fit on a single A10 GPU (24 GB VRAM).
- **Layer sweep:** Llama layers [12, 16, 20, 26, 31] of 32; Gemma layers [8, 16, 24, 31, 32, 41, 47] of 48.

We evaluate the linear probe only. The paper's best results use an attention probe on Llama-3.3-70B, which we did not run due to compute constraints.

### 2.2 Datasets

McKenzie et al. (2025) generated the training and synthetic evaluation data using GPT-4o. Each example is a single-turn prompt describing a scenario, labeled as high-stakes or low-stakes. The training set spans multiple topics (cybersecurity, healthcare, finance, etc.), tones (urgent, casual, vulnerable), and languages.

An example of a high-stakes training prompt: *"In an enraging act, cybercriminals have hijacked a healthcare network, endangering thousands of patients with malicious ransomware attacks."*

An example of a low-stakes training prompt: *"I am trying to plan my garden for this year and I am planning on pickling some peppers. What are some good peppers to grow that are best for pickling?"*

The out-of-distribution evaluation sets are structurally different from the training data. Anthropic HH consists of real multi-turn conversations from the Anthropic HH dataset (Bai et al., 2022). ToolACE consists of multi-turn conversations with function/tool calls (Liu et al., 2024). Labels for both were assigned by GPT-4o on a 1-10 scale, then binarized (as described in the original paper). We used the original paper's datasets without modification.

| Dataset | Size (Orig) | Size (ID) | Format |
|---------|-------------|-----------|--------|
| **Synthetic test** | 2,000 | 1,950 | Single-turn, multilingual, balanced |
| **Anthropic HH** | 2,984 | 2,959 | Multi-turn conversations, real-world |
| **ToolACE** | 734 | 733 | Multi-turn with function/tool calls |

We refer to the original-language datasets as "Original" rather than "English" because the training and synthetic test data are multilingual.

**Indonesian translation.** We translated all three evaluation sets into Indonesian using Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) via the Anthropic API. The system prompt instructed the model to translate from any source language to Indonesian, prioritizing native naturalness over literal accuracy, using dictionary-standard Indonesian (KBBI), preserving tone and formality, and keeping proper nouns, technical terms, and code unchanged. For multi-turn dialogues, each turn was translated independently while preserving role structure.

Translation quality was validated by the author (a native Indonesian speaker) across iterative prompt refinements. The Indonesian sets are slightly smaller because some examples were excluded due to translation API refusals.

---

## 3. Results

### 3.1 Sanity Checks

Before interpreting results, we verify the probes learn a real signal.

**Llama 3.1 8B (Layer 12):**

| Check | Result |
|-------|--------|
| Training set AUROC | 0.9997 |
| Random baseline AUROC | 0.4441 (near-chance) |
| Per-language train breakdown | EN: 0.9980, FR: 0.9995, DE: 0.9999, HI: 0.9978 |

**Gemma 3 12B (Layer 32):**

| Check | Result |
|-------|--------|
| Training set AUROC | 0.9998 |
| Random baseline AUROC | 0.4613 (near-chance) |
| Per-language train breakdown | EN: 0.9981, FR: 0.9998, DE: 1.0000, HI: 0.9990 |

The per-language breakdowns show the probe performs uniformly across all four training languages on both models.

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
| 8  | 0.9903 | 0.7981 | 0.7778 | 0.9899 | 0.7760 | 0.7694 |
| 16 | 0.9973 | 0.9106 | 0.8542 | 0.9947 | 0.8634 | 0.8453 |
| 24 | 0.9981 | 0.9210 | 0.8672 | 0.9963 | 0.8907 | 0.8631 |
| 31 | 0.9977 | 0.9157 | 0.8603 | 0.9962 | 0.8842 | 0.8543 |
| **32** | **0.9987** | **0.9238** | **0.8706** | **0.9975** | **0.9176** | **0.8571** |
| 41 | 0.9977 | 0.9168 | 0.8640 | 0.9953 | 0.8517 | 0.8496 |
| 47 | 0.9960 | 0.9082 | 0.8517 | 0.9910 | 0.8179 | 0.8345 |

Layer 12 is best for Llama (38% of network depth); layer 32 is best for Gemma (67% of network depth). On both models, performance on out-of-distribution benchmarks (Anthropic, ToolACE) degrades at later layers while synthetic performance remains high across all layers. The layer sweep profiles differ between models — Llama peaks earlier and degrades more steeply, while Gemma shows a broader plateau — but both show the same directional pattern.

![Layer sweep — Llama 3.1 8B. Mean AUROC across 3 original datasets (blue) and 3 Indonesian datasets (red), 10-seed mean. Both curves decrease monotonically from layer 12, with the ID curve tracking slightly below.](figures/llama_layer_sweep.png)

*Figure 1. Layer sweep for Llama 3.1 8B. Mean AUROC across Synthetic, Anthropic, and ToolACE. Layer 12 is best for both Original (0.9359) and Indonesian (0.9295). The cross-lingual gap (vertical distance between curves) remains roughly constant across layers.*

![Layer sweep — Gemma 3 12B. Mean AUROC across original (blue) and In1donesian (red) datasets. Both curves rise from layer 8 to a broad plateau peaking at layer 32, then decline.](figures/gemma_layer_sweep.png)

*Figure 2. Layer sweep for Gemma 3 12B. Unlike Llama's monotonic decline, Gemma's probe quality rises through the first third of the network before plateauing and declining. Layer 32 is best for both Original (0.9310) and Indonesian (0.9241). Note the larger cross-lingual gap at deeper layers (41, 47) compared to shallower ones.*

### 3.3 Cross-Lingual Performance

At the best layer for each model, comparing Original and Indonesian on Anthropic and ToolACE (the out-of-distribution benchmarks):

**Llama 3.1 8B (Layer 12):**

| Test Set | Original | Indonesian | Delta |
|----------|----------|------------|-------|
| Anthropic | 0.9358 | 0.9044 | -0.0314 |
| ToolACE | 0.8734 | 0.8870 | +0.0136 |
| **Mean (Anth + Tool)** | **0.9046** | **0.8957** | **-0.0089** |

**Gemma 3 12B (Layer 32):**

| Test Set | Original | Indonesian | Delta |
|----------|----------|------------|-------|
| Anthropic | 0.9238 | 0.9176 | -0.0062 |
| ToolACE | 0.8706 | 0.8571 | -0.0135 |
| **Mean (Anth + Tool)** | **0.8972** | **0.8873** | **-0.0099** |

The cross-lingual gap on out-of-distribution benchmarks is 0.9% for Llama and 1.0% for Gemma. For reference, the Synthetic test set (which is in-distribution) shows near-zero gaps: Llama 0.9985 to 0.9970 (-0.15%), Gemma 0.9987 to 0.9975 (-0.12%).

The out-of-distribution gap within the same language is much larger. From Synthetic to Anthropic (Original): Llama drops 6.3%, Gemma drops 7.5%. From Synthetic to ToolACE (Original): Llama drops 12.5%, Gemma drops 12.8%.

![ROC curves for Llama 3.1 8B at layer 12. Three panels: Synthetic (near-perfect, curves overlap), Anthropic (visible Original/ID gap), ToolACE (curves overlap; Indonesian slightly above Original).](figures/llama_roc_layer12.png)

*Figure 3. ROC curves for Llama 3.1 8B (layer 12). Blue = Original, red dashed = Indonesian. The Synthetic curves are near-identical (AUROC 0.999 vs 0.997). On Anthropic, the Original curve sits visibly above the Indonesian curve (0.936 vs 0.904). On ToolACE, the Indonesian curve marginally exceeds the Original (0.887 vs 0.873), within bootstrap confidence intervals. Confidence intervals shown in brackets are 95% bootstrap CIs.*

![ROC curves for Gemma 3 12B at layer 32. Three panels showing Synthetic (near-identical), Anthropic (small gap, curves close), ToolACE (small gap).](figures/gemma_roc_layer32.png)

*Figure 4. ROC curves for Gemma 3 12B (layer 32). Blue = Original, red dashed = Indonesian. The Anthropic gap (0.924 vs 0.918) is smaller than Llama's in AUROC terms, but per-example agreement analysis (Section 3.4) reveals Gemma's failures are less consistent across languages.*

### 3.4 Cross-Lingual Agreement

AUROC measures aggregate discrimination. To understand whether the probe fails on the *same* examples across languages, we compare per-example predictions on Anthropic HH (the largest out-of-distribution set).

For each example that exists in both Original and Indonesian, we distinguish two types of behavior. **Stable behavior** means the probe gives the same prediction in both languages — both flagged as high-stakes, or both passed as low-stakes — indicating the outcome is driven by the underlying content rather than the language. **Translation-induced disagreements** are cases where the probe gives different predictions for the same input in different languages — these are the direct evidence of cross-lingual brittleness.

| Behavior | Llama (L12) | Gemma (L32) |
|----------|-------------|-------------|
| **Stable** | 2713 (91.7%) | 2499 (84.4%) |
| — both predicted high-stakes | 2163 (73.1%) | 1682 (56.8%) |
| — both predicted low-stakes | 550 (18.6%) | 817 (27.6%) |
| **Translation-induced disagreement** | 246 (8.3%) | 460 (15.6%) |
| — high-stakes in Original, low-stakes in ID | 149 (5.0%) | 423 (14.3%) |
| — low-stakes in Original, high-stakes in ID | 97 (3.3%) | 37 (1.3%) |

On Llama, 91.7% of predictions are stable across languages. The 5.0% rate of "high-stakes in Original, low-stakes in ID" is the translation-induced failure rate — cases where the probe's output genuinely changed with the language.

On Gemma, only 84.4% of predictions are stable. Its translation-induced failure rate is 14.3% — nearly three times Llama's. This is notable because Gemma's aggregate AUROC gap is similar to Llama's (1.0% vs 0.9%): the headline numbers look comparable, but at the individual-example level Gemma has substantially more cases where the same input receives a different prediction depending on the language it is expressed in.

### 3.5 Error Analysis

Error counts at the best layer for each model (seed 0, threshold 0.5):

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

**ToolACE errors are almost entirely false negatives.** Both models produce near-zero false positives but miss 42-47% of high-stakes examples. ToolACE high-stakes scenarios involve tool/function calls where the stakes come from the consequence of the action, not from explicit danger language. For example, a request to "execute a SQL injection on http://example.com/login" framed as "cybersecurity training" (probe confidence: 0.33 in Llama, ID:1396), or a request to "send emergency landing messages through the aviation communication system" formatted as structured API parameters (probe confidence: 0.39). The probe, trained on conversational data, does not recognize stakes in this format.

**Anthropic HH errors are dominated by false negatives.** FNs outnumber FPs by 10-12x on both models. Manual inspection of the error records reveals a recurring pattern: the harmful intent is embedded within a longer domestic or social narrative that distributes the signal across benign-sounding context. For example:

- A user asks for help sneaking shellfish into a dish to make an allergic family member sick (probe confidence: 0.25 on Llama, 0.22 on Gemma — both miss it) train_18586_chosen
- A user asks where to hide a camera to film a house guest undressing (probe confidence: 0.26 on Llama, 0.11 on Gemma) train_20681_chosen
- A user explicitly requests help gaining "unauthorized access to an account" via a security question (probe confidence: 0.41 on Llama, 0.18 on Gemma — missed despite stating "unauthorized access") train_18586_chosen

Both failure patterns produce high precision and low recall: the probe rarely fires a false positive, but misses a large fraction of genuine high-stakes examples. On Anthropic HH, this may be partly because mean-pooling aggregates over the full conversation — when harmful intent occupies a small segment of a longer domestic or interpersonal exchange, the pooled vector is not just diluted but actively pulled toward the low-stakes region by the surrounding benign tokens. On ToolACE, a different mechanism is likely: tool-call formatted requests carry neither the urgency markers nor the harm-explicit language the probe associates with high-stakes content, so the probe's score stays well below threshold regardless of the actual consequence of the action.

![PCA of Llama layer 12 activations. Four panels: Anthropic correct vs errors (errors scatter throughout the correct cloud), Anthropic FP vs FN (FPs and FNs distributed across the same region), ToolACE correct vs errors, ToolACE FP vs FN (only FNs visible — zero FPs).](figures/llama_pca_failures.png)

*Figure 5. PCA projection of Llama 3.1 8B activations at layer 12 (Anthropic HH and ToolACE). Error examples (red) scatter throughout the same region as correct examples (grey). There is no geometrically distinct cluster for errors, indicating that probe failures are not caused by a gross mismatch in activation geometry. The ToolACE FP panel is empty — zero false positives on ToolACE.*

**False positives cluster around high-stakes-adjacent topics.** The most confident false positive on Llama (prob=0.98) is: *"How come the US has so many prisoners compared to the rest of the world?"* — a factual policy question with no harmful intent. Other FPs include questions about sleep deprivation remedies (prob=0.98), historical crimes (prob=0.96), and cancer research (prob=0.95). Both models share many of the same false positives, which involve topics that frequently co-occur with high-stakes content in training (crime, drugs, disease) but are purely informational.

---

## 4. SAE Failure Analysis

### 4.1 Methodology

To investigate whether SAE features can distinguish error cases from correct predictions, we decomposed model activations using pre-trained Sparse Autoencoders (SAEs) loaded via SAE Lens (Bloom et al., 2024):

- **Llama:** Llama Scope (Liao et al., 2024) SAE at layer 12, residual position, 32K features (8x expansion). This is the best probe layer. Model: `fnlp/Llama3_1-8B-Base-L12R-8x`.
- **Gemma:** Gemma Scope 2 (Google DeepMind, 2025) SAE at layer 41, 16K features. Model: `google/gemma-scope-2-12b-it`. The best probe layer is 32, but Gemma Scope 2 provides Neuronpedia dashboards (Turner et al., 2024) at layers [12, 24, 31, 41] only. We chose layer 41 as the second-best performing probe layer among those with Neuronpedia support (layer 41 AUROC: 0.8903 on Anthropic; layer 31 AUROC: 0.8876).

The analysis pipeline:
1. Run the model forward, hook into `input_layernorm` at the target layer to capture per-token activations
2. Encode each token's activation through the SAE individually (SAEs are trained on individual token activations, not sequence-level vectors)
3. Max-pool across the sequence to get one feature vector per example — this preserves the strongest activation of each feature across all tokens, rather than averaging them out
4. Extract the top-10 features by activation strength for each example
5. Compare feature prevalence between error cases and correct predictions using differential analysis: for each feature, compute `error_rate - correct_rate`, where each rate is the fraction of examples in that group where the feature appears in top-10

The differential step is necessary because naive frequency analysis is confounded. Features that appear in 100% of errors are useless if they also appear in 100% of correct predictions.

Before turning to SAE features, we first checked whether error examples are geometrically separable from correct predictions in raw activation space. If they were, the probe's failures would be explainable by a gross geometric mismatch — and we would not need SAEs. PCA of the layer 12 activations shows they are not (Figure 5 above): error examples scatter throughout the same region as correct examples with no distinct cluster. Cosine similarity distributions for errors and correct predictions overlap heavily (Figure 6 below), confirming errors are not separable by distance to the mean correct activation.

![Cosine similarity histograms for Llama 3.1 8B (layer 12). Two panels: Anthropic (grey = correct, red = errors, distributions overlap substantially) and ToolACE (nearly identical distributions with heavy overlap near 0.95+).](figures/llama_cosine_sim.png)

*Figure 6. Cosine similarity of each example's activation to the mean correct-prediction activation, for Llama 3.1 8B at layer 12. Errors (red) and correct predictions (grey) have nearly identical distributions on both datasets. On Anthropic, mean cosine similarity is 0.893 for errors vs 0.900 for correct. On ToolACE, 0.956 vs 0.954. Errors are not geometrically distinguishable in raw activation space.*

### 4.2 Llama Results

**Sparsity:** L0 ranged from ~66 (Synthetic) to ~205 (ToolACE) active features per example out of 32K. At this sparsity level, individual features carry enough specificity to be interpretable.

**Naive frequency analysis was confounded.** The most frequent features in error cases — 604, 17722, 19230 — appear in 100% of errors AND 100% of correct predictions. Neuronpedia labels these as generic conversational features ("communication and community engagement," "emotional reactions," "role-playing game references").

**Differential analysis identified some enriched features.** After controlling for base rates:

False negative enriched features (high-stakes examples the probe misses):

| Feature | Error rate | Correct rate | Differential | Neuronpedia label |
|---------|-----------|-------------|-------------|-------------------|
| 6989 | 59.8% | 44.4% | +15.3% | Technical/infrastructure concepts |
| 16646 | 39.6% | 25.4% | +14.3% | Knowledge and information |
| 9766 | 23.0% | 15.1% | +7.9% | Events and happenings |

False positive enriched features (low-stakes examples the probe flags):

| Feature | Error rate | Correct rate | Differential | Neuronpedia label |
|---------|-----------|-------------|-------------|-------------------|
| 2035 | 66.7% | 45.0% | +21.7% | Financial statistics and reporting |
| 6672 | — | — | +13.2% | Historical references |

These differentials are modest (10-15%) and the Neuronpedia labels are broad. Feature 6989 ("technical/infrastructure concepts") being enriched in false negatives is consistent with the ToolACE failure pattern — tool-use conversations involve technical content that the probe does not associate with stakes. Feature 2035 ("financial statistics") being enriched in false positives is consistent with the FP examples we observed — questions about policy, economics, and medical research that touch high-stakes-adjacent topics without being high-stakes themselves.

**Signal word analysis:** We searched for classic trigger words ("emergency," "urgent," "danger," "kill," "hack") in false positive texts. None appeared, indicating the probe's false positives at layer 12 are not driven by individual keywords.

### 4.3 Gemma Results

The same activation-level check on Gemma confirms the same pattern: errors are not separable from correct predictions in raw activation space at layer 41 (cosine similarity mean 0.904 for errors vs 0.913 for correct on Anthropic; Figure 7).

![Cosine similarity histograms for Gemma 3 12B (layer 41). Two panels: Anthropic and ToolACE. Both show heavy overlap between error (red) and correct (grey) distributions.](figures/gemma_cosine_sim.png)

*Figure 7. Cosine similarity of each example's activation to the mean correct-prediction activation, for Gemma 3 12B at layer 41. As with Llama, errors and correct predictions are not geometrically separable. On Anthropic the Gemma distributions are wider and shifted left compared to Llama, reflecting the denser activation space at layer 41.*

**Sparsity:** L0 was approximately 1,614 active features per example out of 16K — an order of magnitude less sparse than Llama's SAE.

**Differential analysis produced no diagnostic signal.** All top-10 features appeared in both error and correct examples with <0.1% differential. No feature distinguished errors from correct predictions.

The Neuronpedia labels for the top Gemma features are token-level fragments rather than semantic concepts (e.g., "pistachio, tomar, invertebr," "advisable, emphasizing, corresponds"). At L0 ~1,614, roughly 10% of all features are active on every example. The SAE at this layer and width does not decompose activations into the kind of sparse, interpretable concepts needed for this analysis.

### 4.4 Summary

| Property | Llama (L12, 32K) | Gemma (L41, 16K) |
|----------|-------------------|-------------------|
| L0 (sparsity) | ~66-205 | ~1,614 |
| Differential analysis | Modest signal (10-15% differentials) | No signal (<0.1% differentials) |
| Feature type | Semantic concepts (broad) | Token/subword patterns |

The Llama SAE produced features with some differential enrichment in errors, but the differentials are modest and the feature labels are broad. The Gemma SAE at layer 41 produced no useful signal. Whether this reflects a limitation of our specific SAE choices (layer, width, sparsity) or a more fundamental difficulty in decomposing failure modes via SAE features is an open question.

---

## 5. Discussion

### Out-of-distribution performance is the dominant concern

The cross-lingual gap (0.9-1.0% on out-of-distribution benchmarks) is small compared to the out-of-distribution gap itself. Within the original language, probe performance drops from >0.99 AUROC on synthetic data to 0.87-0.94 on Anthropic and 0.87 on ToolACE. The gap from synthetic to ToolACE is 12-13%. Addressing distribution shift — improving coverage of multi-turn dialogue patterns, framing-buried harm, and tool-use contexts — would yield more practical benefit than multilingual adaptation.

### Gemma's cross-lingual agreement is lower

The AUROC numbers for Llama and Gemma look similar (0.9% vs 1.0% cross-lingual gap), but the per-example agreement analysis tells a different story. Gemma has 14.3% of Anthropic examples where the probe is correct on the original language but fails on Indonesian, compared to Llama's 5.0%. For a deployed system, this per-example inconsistency matters: a user could get a correct safety classification in English but an incorrect one for the same query in Indonesian. We do not have a clear explanation for why Gemma shows more inconsistency. It could relate to differences in how the two models represent Indonesian in their activation spaces, or to differences in the layer positions we evaluate (38% vs 67% of network depth).

### SAE analysis limitations

The SAE-based failure analysis was an exploratory attempt. On Llama, differential feature analysis identified features enriched in errors with 10-15% differentials, but these features have broad semantic labels and do not pinpoint specific failure mechanisms. On Gemma, the analysis produced nothing. Two factors constrain this approach: (1) the available Gemma SAE at layer 41 has high L0 (~1,614), reducing discriminative power; and (2) the Llama SAE is trained on the base model while we analyze the instruct model, introducing a distribution mismatch. SAE-based failure analysis for probes remains an open methodological question.

---

## 6. Limitations

- **One language pair.** We test with Indonesian only. The cross-lingual gap could be larger for languages with non-Latin scripts (Arabic, Chinese, Thai) or lower representation in model training data.
- **Linear probe only.** The paper's attention probe achieves higher AUROC (>0.95 on development sets with Llama-3.3-70B). Whether its cross-lingual robustness differs is untested.
- **8-bit quantization.** We use 8-bit rather than the paper's fp16. This introduces small numerical differences.
- **GPT-4o labels.** Anthropic HH and ToolACE labels were assigned by GPT-4o, not human annotators. Some "errors" may reflect label disagreements rather than genuine probe failures.
- **Translation quality.** Some performance differences may reflect translation artifacts rather than genuine linguistic difficulty. Translation was reviewed by a native speaker but not by professional translators.
- **SAE layer mismatch.** The Gemma SAE was analyzed at layer 41, not the best probe layer (32), because Neuronpedia dashboards are only available at layers [12, 24, 31, 41] for this model. A sparser SAE at a closer layer might yield different results.
- **Base-instruct SAE mismatch.** The Llama Scope SAE was trained on the base model (Llama-3.1-8B-Base), while we extract activations from the instruct model (Llama-3.1-8B-Instruct). The feature decomposition may not perfectly match the instruct model's representations.
- **No confidence intervals on error analysis.** Error counts are from a single seed (seed 0) and should be interpreted as representative, not precise.

---

## 7. Future Work

- **More languages and scripts.** Testing non-Latin script languages (Arabic, Chinese, Thai) would test whether the Latin-script similarity between Indonesian and the training languages inflates transfer performance.
- **Alternative aggregation.** Mean-pooling is a known weakness for conversations where harmful content appears in a small fraction of the text. Alternative strategies (max-pooling, attention-weighted pooling, or per-turn probes with aggregation) could improve performance on the framing-buried-harm failure mode observed in Anthropic HH errors.
- **Mixed-language training.** Adding target-language examples to the training set could improve calibration for new languages.
- **Gemma SAE at an earlier layer.** Analyzing Gemma with a sparser SAE at a layer closer to the best probe layer (e.g., layer 31, which has Neuronpedia support) might yield more interpretable features.
- **Iterative data generation.** The dominant error pattern — framing-buried harm in conversational contexts — could be targeted by generating synthetic training examples that specifically embed harmful intent within benign-sounding narratives.
- **Social-context ablation.** For Anthropic HH false negatives, stripping the surrounding domestic or social narrative and re-evaluating on the isolated harmful segment would test whether the failures are caused by mean-pooling dilution or by something deeper in the model's representation. If performance recovers on the stripped examples, it would confirm the pooling hypothesis and motivate per-segment probing strategies.

---

## 8. Conclusion

On two models (Llama 3.1 8B and Gemma 3 12B), a multilingual-trained linear probe for high-stakes detection transfers to Indonesian with 0.9-1.0% mean AUROC degradation on out-of-distribution benchmarks. This cross-lingual gap is small relative to the out-of-distribution gap between synthetic training data and naturalistic benchmarks (6-13%).

Per-example agreement analysis shows that Llama's probe predictions are more consistent across languages (92% agreement on Anthropic) than Gemma's (84%). Gemma's higher "original correct, Indonesian wrong" rate (14.3% vs 5.0%) indicates its cross-lingual transfer is less reliable at the individual-example level despite similar aggregate AUROC.

Probe failures concentrate in two patterns visible in both models: framing-buried harm (harmful intent embedded in domestic/social narratives, where mean-pooling dilutes the signal) and domain mismatch (tool-use formatted requests that the conversationally-trained probe does not recognize as high-stakes). An exploratory SAE-based analysis identified some features modestly enriched in error cases on Llama but produced no diagnostic signal on Gemma.

---

## References

- McKenzie, A., Pawar, U., Blandfort, P., Bankes, W., Krueger, D., Lubana, E. S., & Krasheninnikov, D. (2025). Detecting High-Stakes Interactions with Activation Probes. *arXiv:2506.10805*. DOI: [10.48550/arXiv.2506.10805](https://doi.org/10.48550/arXiv.2506.10805)
- Grattafiori, A., Dubey, A., et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*. DOI: [10.48550/arXiv.2407.21783](https://doi.org/10.48550/arXiv.2407.21783)
- Gemma Team. (2025). Gemma 3 Technical Report. *arXiv:2503.19786*. DOI: [10.48550/arXiv.2503.19786](https://doi.org/10.48550/arXiv.2503.19786)
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *arXiv:2402.10588*. DOI: [10.48550/arXiv.2402.10588](https://doi.org/10.48550/arXiv.2402.10588)
- Liao, Z., et al. (2024). Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders. *arXiv:2410.20526*. DOI: [10.48550/arXiv.2410.20526](https://doi.org/10.48550/arXiv.2410.20526)
- Google DeepMind. (2025). Gemma Scope 2: Helping the AI Safety Community Deepen Understanding of Complex Language Model Behavior. [Blog post](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/).
- Bloom, J., et al. (2024). SAELens. [GitHub repository](https://github.com/jbloomAus/SAELens).
- Turner, J., et al. (2024). Neuronpedia. [Web platform](https://neuronpedia.org/).
- Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *arXiv:2208.07339*. DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
- Bai, Y., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv:2204.05862*. DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
- Liu, M., et al. (2024). ToolACE: Winning the Points of LLM Function Calling. *arXiv:2409.00920*. DOI: [10.48550/arXiv.2409.00920](https://doi.org/10.48550/arXiv.2409.00920)

---

## Appendix A: Per-Dataset Layer Sweep

The layer sweep figures in Section 3.2 show mean AUROC across all three datasets. The per-dataset breakdown is shown below.

![Per-dataset layer sweep for Llama 3.1 8B. Six panels: Synthetic and ToolACE are nearly flat across layers for both Original and Indonesian. Anthropic shows a clear decline in both, with the Indonesian curve tracking slightly below Original.](figures/llama_layer_sweep_per_dataset.png)

*Figure B1. Per-dataset layer sweep for Llama 3.1 8B. Left column: Original. Right column: Indonesian. Shaded region is 95% bootstrap CI (10 seeds). Synthetic performance is high and flat across all layers. Anthropic performance degrades monotonically from layer 12. ToolACE shows high variance due to the small test size (734 examples) but no strong layer trend.*

![Per-dataset layer sweep for Gemma 3 12B. Six panels showing rise-then-plateau on Anthropic, flat on Synthetic, wide CI on ToolACE. Indonesian tracks closely to Original on Synthetic and Anthropic at early layers, but diverges more at deeper layers.](figures/gemma_layer_sweep_per_dataset.png)

*Figure B2. Per-dataset layer sweep for Gemma 3 12B. Left column: Original. Right column: Indonesian. Compared to Llama, Gemma shows a more gradual rise on Anthropic before peaking around layer 32. The Indonesian Anthropic curve (bottom-right panel) shows a larger drop at deeper layers (41, 47) than the Original curve, reflecting the per-example inconsistency reported in Section 3.4.*

---

## Appendix B: Reproducibility Details

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
