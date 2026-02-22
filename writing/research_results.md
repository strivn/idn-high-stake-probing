# Research Results: Notebook Walkthrough and Findings

This document walks through what we did in each notebook, how we did it, and what we found. It's the detailed "how + what" companion to the research tracker (which covers the "why").

---

## Notebook 05: Probe Evaluation (Llama 3.1 8B and Gemma 3 12B)

Two copies of the same notebook, differing only in config: `05_probe_evaluation.ipynb` (Llama) and `05_probe_evaluation_gemma.ipynb` (Gemma). The notebook is designed to be model-agnostic — swap the `MODEL_NAME` and `LAYERS` config, and everything else runs identically.

### What we wanted to figure out

1. Can we reproduce the paper's probe results? (baseline validation)
2. Which layer is best for each model? (layer sweep)
3. Do probes generalize to Indonesian? (cross-lingual robustness)
4. Where do probes fail? (error analysis by dataset)
5. Are results stable across random seeds? (reproducibility)

### Thinking pattern

The paper reports ~0.91 mean AUROC on Llama 3.3 70B with an attention probe. We can't run 70B easily, so we test on 8B (Llama) and 12B (Gemma) with the simpler linear probe. If a linear probe on a smaller model gets reasonable results, that tells us the high-stakes signal is strong and accessible early in the network. If it generalizes cross-lingually, that tells us the signal is language-invariant — a meaningful finding for deployment.

### Technical details

**Activation extraction:**
- Hook point: `input_layernorm` of the target layer (residual stream *before* the layer norm, not after). This matches the paper's methodology — it captures the residual stream state that feeds into the layer.
- Chat template applied: all examples are formatted as chat conversations using the model's tokenizer template, just as they'd appear during inference.
- BOS token stripped: after tokenization, the leading BOS token is removed (`v[:, 1:]`) because the BOS activation carries no content-specific information.
- Max sequence length: 8192 tokens. Longer conversations are truncated.
- Aggregation: mean-pooling across all token positions. Each example becomes a single vector of shape `(hidden_dim,)`.
- Architecture differences: Llama uses `model.model.layers[i]`, Gemma3 uses `model.language_model.layers[i]`. Handled by `get_model_layers()` in `lib/model.py`.
- Quantization: 8-bit (bitsandbytes) for all models. Gemma 3 requires `torch.bfloat16` — using float16 produces NaN activations (transformers#39972, bitsandbytes#1572).

**Probe:**
- `StandardScaler` + `LogisticRegression(C=1e-3, fit_intercept=False, max_iter=5000, solver='lbfgs')`
- Exactly the paper's configuration.
- Deterministic: 0.0000 std across 10 seeds. L2-regularized LBFGS converges to the same optimum regardless of initial conditions. The 10-seed setup confirms this — it's a sanity check, not a source of variance.

**Evaluation:**
- Master seed `20250214` generates 10 child seeds via `np.random.RandomState`.
- AUROC with 1000-sample bootstrap 95% confidence intervals.
- 3 original-language test sets + 3 Indonesian test sets = 6 evaluations per layer.

**Data:**
- Training: 8000 examples (4000 high-stakes, 4000 low-stakes), multilingual (~59% EN, ~14% DE, ~14% FR, ~13% HI).
- Test sets: Synthetic (2000), Anthropic HH (2984), ToolACE (734). All balanced.
- Indonesian: translated via GPT-4o (NB04). Translation refusals filtered — 1 synthetic, 25 anthropic, 1 toolace where GPT-4o kept the original English text.

### Part-by-part walkthrough

**Part 0-2: Setup, Auth, Data Loading**
Standard setup. Environment detection (colab/remote/local), HuggingFace auth from `.env`, dataset loading with refusal filtering for Indonesian.

**Part 3: Extract Activations, Train Probes, Evaluate**
The main loop. For each layer:
1. Extract activations for all 6 datasets (cached as `.npy` files — only computed once per model/layer).
2. For each of 10 seeds, train a probe and evaluate on all datasets.
3. Report mean AUROC +/- std with bootstrap CIs.

**Part 4: ROC Curves**
ROC curves for the best layer, overlaying Original vs Indonesian on each dataset. Visual confirmation of the cross-lingual gap.

**Part 5: Layer Sweep Plot**
Two visualizations:
1. Mean AUROC across 3 datasets vs layer number (the "inverted U" curve).
2. Per-dataset breakdown showing each dataset's AUROC trajectory across layers.

**Part 6: Error Analysis**
Using the canonical probe (seed 0) at the best layer, print the 5 most confident FP and FN examples for each dataset. This gives a qualitative feel for failure modes.

**Part 7: Sense Checks**
Three sanity checks:
1. Training AUROC (should be high but <1.0 — overfitting check).
2. Random label baseline (should be ~0.5 — confirms signal is real).
3. Per-source-language breakdown of synthetic test (EN, FR, DE, HI separately).

**Part 8: Save Artifacts**
All results serialized to JSON for later analysis. Lists all cached files.

### Results: Llama 3.1 8B

**Layer sweep (AUROC, 10-seed mean):**

| Layer | Orig Synthetic | Orig Anthropic | Orig ToolACE | ID Synthetic | ID Anthropic | ID ToolACE |
|------:|---------------:|---------------:|-------------:|-------------:|-------------:|-----------:|
|    12 |         0.9985 |         0.9358 |       0.8734 |       0.9970 |       0.9044 |     0.8870 |
|    16 |         0.9981 |         0.9077 |       0.8773 |       0.9964 |       0.8780 |     0.8707 |
|    20 |         0.9974 |         0.8992 |       0.8540 |       0.9952 |       0.8696 |     0.8607 |
|    26 |         0.9960 |         0.8649 |       0.8363 |       0.9910 |       0.8236 |     0.8543 |
|    31 |         0.9944 |         0.8570 |       0.8528 |       0.9852 |       0.7593 |     0.8417 |

Best layer: **12** (mean Original AUROC 0.9359).
Layer 12 is at 38% of network depth (12/32). The paper used layer 16.

**Sense checks:**
- Training AUROC: 0.9997 (high but not 1.0 — good)
- Random baseline: 0.4441 (near-chance — signal is real)
- Per-language: EN 0.9980, FR 0.9995, DE 0.9999, HI 0.9978 (all excellent)

**Error counts (layer 12, seed 0):**
- Synthetic: 41/2000 (2.1%) — 17 FP, 24 FN
- Anthropic: 651/2984 (21.8%) — 51 FP, 600 FN
- ToolACE: 344/734 (46.9%) — 0 FP, 344 FN
- ID Synthetic: 48/1950 (2.5%) — 19 FP, 29 FN

**Qualitative error patterns:**
- Anthropic FPs: topics like "gruesome crimes", "sleep schedule fixes", "cancer cures" — benign questions that touch high-stakes-adjacent topics.
- Anthropic FNs: multi-turn conversations where early turns are innocent but later turns escalate (e.g., "ice cream toppings" → eating dog poop). The probe only sees the mean-pooled activation, so early-turn innocence dilutes the late-turn danger signal.
- ToolACE: all 344 errors are FNs. The probe doesn't recognize tool-mediated stakes (photo conversion, property lookup, news fetching).

### Results: Gemma 3 12B

**Layer sweep (AUROC, 10-seed mean):**

| Layer | Orig Synthetic | Orig Anthropic | Orig ToolACE | ID Synthetic | ID Anthropic | ID ToolACE |
|------:|---------------:|---------------:|-------------:|-------------:|-------------:|-----------:|
|     8 |         0.9903 |         0.7981 |       0.7778 |       0.9899 |       0.7760 |     0.7694 |
|    16 |         0.9973 |         0.9106 |       0.8542 |       0.9947 |       0.8634 |     0.8453 |
|    24 |         0.9981 |         0.9210 |       0.8672 |       0.9963 |       0.8907 |     0.8631 |
|    31 |         0.9977 |         0.9157 |       0.8603 |       0.9962 |       0.8842 |     0.8543 |
|    32 |         0.9987 |         0.9238 |       0.8706 |       0.9975 |       0.9176 |     0.8571 |
|    41 |         0.9977 |         0.9168 |       0.8640 |       0.9953 |       0.8517 |     0.8496 |
|    47 |         0.9960 |         0.9082 |       0.8517 |       0.9910 |       0.8179 |     0.8345 |

Best layer: **32** (mean Original AUROC 0.9310).
Layer 32 is at 67% of network depth (32/48). Similar relative position to Llama's 38% — both models peak in the upper-middle range.

**Sense checks:**
- Training AUROC: 0.9998
- Random baseline: 0.4613
- Per-language: EN 0.9981, FR 0.9998, DE 1.0000, HI 0.9990

**Error counts (layer 32, seed 0):**
- Synthetic: 36/2000 (1.8%) — 10 FP, 26 FN
- Anthropic: 589/2984 (19.7%) — 91 FP, 498 FN
- ToolACE: 309/734 (42.1%) — 1 FP, 308 FN
- ID Synthetic: 45/1950 (2.3%) — 30 FP, 15 FN

**Notable differences from Llama:**
- Gemma has more Anthropic FPs (91 vs 51) but fewer FNs (498 vs 600).
- Gemma's cross-lingual gap is wider — Anthropic ID AUROC drops from 0.9238 to 0.9176 at best layer, but the gap widens significantly at deeper layers.
- ToolACE pattern is identical: near-zero FPs, all errors are FNs.

---

## Notebook 06: Failure Analysis (Llama 3.1 8B and Gemma 3 12B)

Two copies: `06_failure_analysis.ipynb` (Llama, layer 12) and `06_failure_analysis_gemma.ipynb` (Gemma, layer 41).

### What we wanted to figure out

1. When the probe fails on Original and Indonesian versions of the same example, is it the same failure? (cross-lingual agreement)
2. Do errors live in a distinct region of activation space? (PCA, UMAP, cosine similarity)
3. Can SAE features explain *why* the probe fails? (mechanistic analysis)
4. Which specific features are diagnostic of failures vs. universally active noise? (differential analysis)

### Thinking pattern

The probe makes errors. Are those errors systematic or random? If errors cluster in activation space, the probe has a consistent blind spot. If SAE features can distinguish errors from correct predictions, we can name the blind spot — the features tell us what concept the probe is confused by.

The key methodological challenge: naive frequency analysis is confounded. Features that appear in 100% of errors are useless if they also appear in 100% of correct predictions. We need *differential* analysis: how much more common is a feature in errors vs. correct predictions?

### Technical details

**SAE setup:**
- Llama: `fnlp/Llama3_1-8B-Base-L12R-8x` — Llama Scope, 32K features, layer 12 residual stream. L0 (average active features): 66-205 depending on dataset.
- Gemma: `layer_41_width_16k_l0_medium` via Gemma Scope 2 — 16K features, layer 41. L0: ~1614.

Critical difference: Llama's SAE has L0 ~66-205 (sparse, ~0.3-0.6% of features active). Gemma's SAE has L0 ~1614 (~10% of features active). The Gemma SAE is far less sparse, which directly affects diagnostic power.

**Why layer 41 for Gemma SAE?**
Gemma Scope 2 provides SAEs at layers [12, 24, 31, 41] for the 12B model. The best probe layer is 32, which has no SAE available. Layer 41 is the closest available layer above 32. This is a constraint — ideally we'd analyze the same layer the probe was trained on.

**Per-token SAE encoding:**
1. Run the model's forward pass, hooking into `input_layernorm` at the target layer to capture per-token activations.
2. Encode each token's activation through the SAE's encoder.
3. Max-pool across token positions to get one feature vector per example.

We use max-pooling (not mean-pooling) because SAE features are sparse. Mean-pooling would dilute rare but important feature activations. Max-pooling preserves the strongest activation of each feature across the entire sequence.

**Top-k extraction:**
For each example, extract the 10 features with highest activation (after max-pooling). These are the features most strongly expressed in that example.

**Differential analysis:**
For each feature in the top-k sets:
- `error_rate` = fraction of error examples where this feature appears in top-k
- `correct_rate` = fraction of correct examples where this feature appears in top-k
- `diff = error_rate - correct_rate`

Features with large positive diff are *enriched* in errors — they fire more in examples the probe gets wrong. Features with diff ~0 are confounded (universally active).

**Neuronpedia lookup:**
Top features are looked up on Neuronpedia for human-readable explanations. Llama Scope features have semantic auto-interp labels. Gemma Scope 2 features only have top-token lists (less interpretable).

### Part-by-part walkthrough

**Part 1: Load Results & Identify Failures**
Load the trained probe and cached activations at the analysis layer. Compute predictions, identify FP/FN indices.

**Part 2: Cross-Lingual Flip Analysis**
For each example that exists in both Original and Indonesian, classify into:
- Both correct
- Both wrong
- Original correct, ID fails
- Original fails, ID correct

This tells us whether cross-lingual failures are the *same* failures or different ones.

**Part 3: Activation-Level Analysis (No SAE)**
Before SAE decomposition, check if errors are distinguishable in raw activation space:
- PCA: Project activations to 2D, color by correct/error. If errors cluster, the probe has a geometric blind spot.
- UMAP: Non-linear projection. Reveals clusters that PCA misses.
- Cosine similarity: Compute average cosine similarity of error activations vs correct activations to the "correct centroid." If errors are equally similar to the correct centroid, they're not geometrically separable.

**Part 4: Load SAE and Extract Features**
Load the SAE via SAE Lens, run per-token encoding for all 3 datasets (synthetic, anthropic, toolace). Cache results as `.npy` files.

**Part 5: Feature Frequency and Strength Analysis**
Naive analysis:
1. Feature frequency: how often does each feature appear in top-k of error examples?
2. Feature activation strength: sum of activation values across all error examples.

**Part 5.1: Differential Feature Analysis**
The core methodological contribution. Filters out universally-active features and ranks by `error_rate - correct_rate`. This is where diagnostic features emerge (for Llama) or fail to emerge (for Gemma).

**Part 6: Neuronpedia Explanations**
Look up top features' semantic labels. For Llama, these are meaningful (e.g., "financial statistics", "emotional reactions"). For Gemma, they're token-level fragments (e.g., "pistachio, tomar, invertebr").

**Part 7: Signal Word Analysis**
Check if specific high-stakes signal words ("emergency", "urgent", "crisis", etc.) appear in FP texts. Tests the paper's hypothesis that FPs are triggered by surface-level signal words in benign contexts.

**Part 8: Annotated Failure Examples**
Print sampled failure examples with their top-10 SAE features and Neuronpedia explanations. Gives a qualitative sense of what the model "sees" in misclassified examples.

**Part 9-10: Summary and Export**
Save analysis results to JSON. Export all error records to JSONL for external analysis.

### Results: Llama 3.1 8B (layer 12)

**Cross-lingual agreement:**

| Category | Anthropic (n=2959) | ToolACE (n=733) |
|----------|-------------------:|----------------:|
| Both correct | 2163 (73.1%) | 378 (51.6%) |
| Both wrong | 550 (18.6%) | 329 (44.9%) |
| Orig ok, ID fail | 149 (5.0%) | 11 (1.5%) |
| Orig fail, ID ok | 97 (3.3%) | 15 (2.0%) |

High "both wrong" rates mean failures are mostly language-invariant — the same examples are hard regardless of language. The low flip rates (5.0% and 1.5% for Orig→ID fail) confirm strong cross-lingual robustness.

**Activation-level analysis:**
- Cosine similarity: errors and correct predictions are nearly identical in cosine distance to the correct centroid (Anthropic: 0.893 vs 0.900, ToolACE: 0.956 vs 0.954). Errors are NOT geometrically separable in raw activation space.
- PCA and UMAP: errors scatter throughout the correct-prediction cloud with no distinct clustering.

This is important: it means the probe's errors aren't caused by a gross geometric mismatch. The signal is subtle.

**SAE feature analysis:**

Naive frequency: Features 604, 17722, 19230 appear in 100% of both errors and correct predictions. These are universally active and non-diagnostic.

Differential analysis (Anthropic):

| Feature | Error % | Correct % | Diff | Explanation |
|---------|--------:|----------:|-----:|-------------|
| 6989 | 59.8% | 44.4% | +15.3% | (not auto-labeled) |
| 16646 | 39.6% | 25.4% | +14.3% | (not auto-labeled) |
| 9766 | 23.0% | 15.1% | +7.9% | (not auto-labeled) |
| 11897 | 12.9% | 6.6% | +6.3% | (not auto-labeled) |
| 5401 | 22.0% | 15.8% | +6.1% | (not auto-labeled) |

Anthropic FP-enriched features:

| Feature | Explanation |
|---------|-------------|
| 2035 (66.7% in FPs) | financial statistics and reporting |
| 19651 (35.3%) | student engagement and organizational activities |
| 24577 (33.3%) | video and animation production tools |
| 6672 (not in top-10 but enriched) | historical references |

These make sense: FPs are triggered by benign content about finance, education, and media — topics that share surface features with genuinely high-stakes scenarios.

Universally active (non-diagnostic) features with Neuronpedia labels:
- Feature 604: "phrases related to communication and community engagement"
- Feature 17722: "emotional reactions and expressions related to discussions"
- Feature 19230: "references to role-playing games (RPGs) and related gaming concepts"
- Feature 2462: "references to academic articles or publications"

These features fire on virtually all text, regardless of stakes level. They represent general conversational and textual structure, not high-stakes-specific concepts.

**SAE sparsity:**
- Synthetic: L0 = 66.0 (very sparse — short, clean examples)
- Anthropic: L0 = 111.6 (moderate — longer, conversational)
- ToolACE: L0 = 205.1 (dense — tool-calling has structured schemas)

**Signal word analysis:**
No signal words found in Anthropic FP texts. The paper predicted signal words like "emergency" and "crisis" would trigger FPs. Our results don't confirm this — the FPs in our analysis are driven by topical similarity (finance, medicine), not individual trigger words.

### Results: Gemma 3 12B (layer 41)

**Important caveat:** The SAE analysis was done at layer 41, not the best probe layer (32). Gemma Scope 2 doesn't provide an SAE for layer 32 — the available layers are [12, 24, 31, 41]. Layer 41 was chosen as the closest available above the best probe layer. This means the error counts in NB06 Gemma (862 Anthropic errors) are based on the layer 41 probe, not the layer 32 probe (589 errors). The SAE features at layer 41 may not reflect the same representations that the best probe uses.

**Cross-lingual agreement:**

| Category | Anthropic (n=2959) | ToolACE (n=733) |
|----------|-------------------:|----------------:|
| Both correct | 1682 (56.8%) | 395 (53.9%) |
| Both wrong | 817 (27.6%) | 287 (39.2%) |
| Orig ok, ID fail | 423 (14.3%) | 44 (6.0%) |
| Orig fail, ID ok | 37 (1.3%) | 7 (1.0%) |

Much worse cross-lingual robustness than Llama. The "Orig ok, ID fail" rate is 14.3% for Anthropic (vs Llama's 5.0%). Gemma breaks on Indonesian examples much more often.

**Activation-level analysis:**
Similar to Llama — errors are not geometrically separable. Cosine similarities: Anthropic 0.904 vs 0.913, ToolACE 0.962 vs 0.959.

**SAE feature analysis:**

This is the most striking result: **Gemma Scope 2 features are completely non-diagnostic.**

All top-10 features appear in 100% of both errors and correct predictions. The differential analysis shows <0.1% difference for every feature:

| Feature | Error % | Correct % | Diff |
|---------|--------:|----------:|-----:|
| 11338 | 99.8% | 99.7% | +0.1% |
| 5141 | 0.2% | 0.3% | -0.1% |
| (all others) | 100.0% | 100.0% | 0.0% |

The Neuronpedia explanations confirm why: Gemma Scope 2 features at layer 41 are not semantic concepts — they're multilingual token-level fragments:
- Feature 14802: "advisable, emphasizing, corresponds, categorically, அது"
- Feature 13700: "courtship, ngram, shenanigans, snippet, huevos"
- Feature 1209: "pistachio, tomar, invertebr, fallback, phospholip"
- Feature 14074: "=, *, ', ==, !="

These are essentially vocabulary-level features, not concept-level features. With L0 ~1614 (10% of all features active on every example), the SAE is too dense to provide discriminative decomposition.

**Root cause of Gemma SAE failure:**
1. **Sparsity:** L0 ~1614 vs Llama's ~66-205. When 10% of features are active on every example, top-k analysis can't distinguish signal from noise.
2. **Feature quality:** The auto-interp labels are token lists, not semantic descriptions. This suggests the features haven't learned clean, interpretable concepts at this width (16K) and layer.
3. **Layer mismatch:** Layer 41 is not where the best probe operates (layer 32). The representations at layer 41 may have already transformed past the high-stakes-relevant features.

---

## Cross-Notebook Comparison

### Probe quality is model-general

Both models achieve similar overall performance:
- Llama 3.1 8B: mean Original AUROC 0.9359 at layer 12
- Gemma 3 12B: mean Original AUROC 0.9310 at layer 32

The ~0.5% difference is negligible. High-stakes detection via linear probes works across model families.

### Cross-lingual robustness varies by model

Llama is significantly more robust to Indonesian:
- Llama Anthropic: 0.9358 → 0.9044 (3.4% drop)
- Gemma Anthropic: 0.9238 → 0.9176 (0.7% drop at best layer, but 14.3% flip rate in agreement analysis)

The flip analysis tells a different story than the AUROC gap: Gemma has more *directional* failures (Original ok → ID fail), even though its absolute AUROC drop is small. This suggests Gemma's cross-lingual transfer is less reliable at the individual-example level.

### SAE interpretability depends on SAE quality, not model quality

The key finding: **SAE quality determines whether mechanistic analysis succeeds.** Llama Scope (32K features, L0 ~66-205) produces diagnostically useful features. Gemma Scope 2 (16K features at layer 41, L0 ~1614) does not.

This isn't a statement about Llama vs Gemma as models — it's about the SAE training and layer choice. A sparser Gemma SAE (higher width, lower L0) at a closer layer might work better. But with the tools available to us, only Llama Scope produced interpretable failure analysis.

### Error patterns are consistent across models

Both models show:
- Synthetic: very low error rate (1.8-2.1%), dominated by edge cases
- Anthropic: significant error rate (19.7-21.8%), heavily FN-skewed, driven by multi-turn escalation
- ToolACE: high error rate (42.1-46.9%), exclusively FN, driven by domain mismatch (tool-mediated stakes)
- Both share many of the same hard examples (e.g., IDs OAVPnS9W, Z8Ob5Xb2, b71CLEFD appear in both models' error lists)

---

## Summary of key technical decisions and their rationale

| Decision | Rationale |
|----------|-----------|
| `input_layernorm` hook (not full layer output) | Matches paper. Captures residual stream before layer processing. |
| Mean-pooling tokens → single vector | Matches paper. Alternative (last token) loses multi-turn context. |
| Max-pooling for SAE features | Preserves sparse activations. Mean-pooling kills sparsity. |
| 8-bit quantization for all models | Saves ~8GB VRAM, negligible quality impact on probe. |
| `bfloat16` for Gemma (not `float16`) | Gemma3 produces NaN with float16 quantization. |
| C=1e-3, fit_intercept=False | Paper's exact hyperparameters. |
| 10 seeds with master seed | Reproducibility check. Turned out to be uninformative (0.000 std) because L2+LBFGS is deterministic. |
| Layer 41 for Gemma SAE (not 32) | Layer 32 SAE doesn't exist in Gemma Scope 2. Layer 41 is closest available. |
| Differential analysis (not naive frequency) | Universally-active features confound naive analysis. Differential reveals actual diagnostic features. |
| Top-k=10 features per example | Balances coverage with noise. More features = more universally-active noise. |
