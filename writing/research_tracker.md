# Research Tracker

Project: Cross-Lingual Robustness of High-Stakes Probes with SAE-Based Failure Analysis
Paper: arXiv:2506.10805 — Detecting High-Stakes Interactions with Activation Probes

---

### RQ-1: Can we reproduce the paper's probe results on Llama 3.1 8B?

- **Question:** Does a linear probe on Llama 3.1 8B activations achieve comparable AUROC to the paper's sklearn baseline?
- **Why:** Establishes our baseline and validates the pipeline before extending it.
- **Method:** NB05, StandardScaler + LogisticRegression (C=1e-3), layer 12, 10 seeds (master seed 20250214).
- **Status:** `done`
- **Findings:** Mean AUROC 0.9359 across 3 original datasets. Synthetic 0.9985, Anthropic 0.9358, ToolACE 0.8734. Per-language training AUROC: EN 0.9980, FR 0.9995, DE 0.9999, HI 0.9978. Training AUROC 0.9997, random baseline 0.4441. Paper reports ~0.91 on 70B with attention probe — our 8B linear probe now exceeds that with proper 10-seed averaging. Error counts (seed 0, layer 12): Synth 41/2000 (17FP, 24FN), Anth 651/2984 (51FP, 600FN), ToolACE 344/734 (0FP, 344FN). (NB05 refreshed run, Feb 2026)
- **Decisions:** Use paper's exact hyperparameters (C=1e-3, fit_intercept=False, max_length=8192). Layer 12 selected over 16 based on sweep results.
- **Open issues:** None.

### RQ-2: Which layer is best for probe performance?

- **Question:** Does the optimal layer match the paper's choice (layer 16), or is there a better one?
- **Why:** Layer choice directly affects probe quality. The paper used layer 16 for 8B but didn't exhaustively sweep.
- **Method:** NB05, layer sweep [12, 16, 20, 26, 31] for Llama; [8, 16, 24, 31, 32, 41, 47] for Gemma. 10 seeds each.
- **Status:** `done`
- **Findings:** Llama 3.1 8B: layer 12 best (mean 0.9359). Gemma 3 12B: layer 32 best (mean 0.9310 across 3 orig datasets). Both models peak at ~60-70% network depth — high-stakes detection happens in the upper-middle layers, before generation strategy interferes. (NB05 + NB05_gemma, Feb 2026)
- **Decisions:** Use layer 12 for Llama, layer 32 for Gemma in all downstream analysis.
- **Open issues:** None.

### RQ-3: Do probes generalize cross-lingually (English-trained, Indonesian-tested)?

- **Question:** Does a probe trained on multilingual data (EN/DE/FR/HI) transfer to Indonesian without retraining?
- **Why:** Tests whether the "high-stakes" concept is language-invariant in activation space — key for deployment in non-training languages.
- **Method:** NB04 (translation via GPT-4o), NB05 (evaluation). Same probe, Indonesian test sets. 10 seeds.
- **Status:** `done`
- **Findings:** Llama: ID Synth 0.9970, ID Anth 0.9044, ID ToolACE 0.8870. ID Synth errors: 48/1950 (19FP, 29FN). Cross-lingual robustness is strong — high-stakes concept is language-invariant. Gemma: ID Synth 0.9975, ID Anth 0.9176, ID ToolACE 0.8571 at best layer (32). AUROC drop is small, but cross-lingual agreement analysis (NB06) reveals Gemma has 14.3% orig-ok/ID-fail rate on Anthropic vs Llama's 5.0% — less reliable at the individual-example level. (NB05, Feb 2026)
- **Decisions:** Filtered translation refusals (1 synthetic, 25 anthropic, 1 toolace where GPT-4o kept original English).
- **Open issues:** None.

### RQ-4: Why do probes fail on Anthropic and ToolACE?

- **Question:** What are the mechanistic reasons probes misclassify real-world conversations?
- **Why:** Understanding failure modes tells us where probes are safe to deploy and where they need augmentation.
- **Method:** NB06, error categorization, manual inspection of FP/FN examples.
- **Status:** `done`
- **Findings:** Anthropic: 51 FP, 600 FN. Many FNs are multi-turn escalation (early turns innocent, later turns dangerous — e.g. "ice cream toppings" conversation escalates). FPs are driven by topical similarity (finance, medicine, security topics) rather than individual signal words — our signal word analysis found no matches in FP texts. ToolACE: 0 FP, 344 FN — all errors are FN where GPT-4o labels tool interactions as high-stakes by consequence severity (tornado alerts, infrastructure deletion) but the probe doesn't recognize tool-use stakes. (NB06, Feb 2026)
- **Decisions:** Error counts from JSONL are authoritative (NB06 cell outputs are stale from earlier run).
- **Open issues:** None.

### RQ-5: Can SAE features explain probe failures mechanistically?

- **Question:** Do specific SAE features correlate with probe errors vs correct predictions?
- **Why:** Moves beyond "the probe is wrong" to "here's the interpretable feature pattern that confuses it."
- **Method:** NB06, per-token SAE encoding + max-pooling, top-k feature extraction, differential prevalence analysis (error_rate - correct_rate).
- **Status:** `done`
- **Findings:**
  - **Llama (fnlp/Llama3_1-8B-Base-L12R-8x, 32K features, layer 12, L0 ~66-205):** Naive frequency confounded — features 604, 17722, 19230 appear in 100% of errors AND correct predictions. Differential analysis identifies diagnostic features: 6989 (+15.3% in errors), 16646 (+14.3%), 9766 (+7.9%). FP-enriched features: 2035 (66.7% in FPs, "financial statistics"). Neuronpedia labels are semantic and meaningful. SAE features successfully distinguish failure modes.
  - **Gemma (layer_41_width_16k_l0_medium, 16K features, layer 41, L0 ~1614):** ALL top-10 features universally active (100% in errors AND correct). Differential analysis shows <0.1% difference — SAE features completely non-diagnostic. Neuronpedia labels are token-level fragments ("pistachio, tomar, invertebr"), not semantic concepts. Root cause: L0 ~1614 means ~10% of features active on every example (vs Llama's ~0.3-0.6%), destroying discriminative power. Note: SAE analyzed at layer 41, not best probe layer 32 — Gemma Scope 2 only provides SAEs at [12, 24, 31, 41].
- **Decisions:** Use 8x SAE (32K features) for Llama, not 32x (128K). Per-token encoding with max-pooling (not mean-pool-then-encode, which kills sparsity).
- **Open issues:** Gemma SAE results may improve with a sparser SAE (higher width, e.g. 65K/262K) or at a layer closer to the best probe layer. Not pursued due to time constraints.

### RQ-6: Do the same SAE features drive failures across languages?

- **Question:** When the probe fails on both English and Indonesian versions of the same example, are the same SAE features active?
- **Why:** If yes, failures are concept-level (language-invariant). If no, failures are surface-level (language-specific features confuse the probe differently).
- **Method:** Extract SAE features for Indonesian datasets, compare feature overlap between EN/ID failures on aligned examples.
- **Status:** `dropped`
- **Findings:** Not executed — SAE features only extracted for Original datasets. Cross-lingual agreement measured at probe level (NB06 flip analysis) but not at SAE feature level. The flip analysis already shows failures are mostly language-invariant (73.1% "both wrong" for Llama Anthropic), which partially answers this question at the probe level.
- **Decisions:** Dropped — requires GPU time for Indonesian SAE extraction. Given time constraints, the probe-level flip analysis provides sufficient evidence that failures are concept-level.
- **Open issues:** None (dropped Feb 17 2026).

### RQ-7: Does Gemma 3 12B show the same probe behavior as Llama 3.1 8B?

- **Question:** Do probes on a different model family reproduce similar AUROC and failure patterns?
- **Why:** Tests whether findings are model-general or Llama-specific.
- **Method:** NB05_gemma, Gemma 3 12B (google/gemma-3-12b-it), layer sweep [8, 16, 24, 31, 32, 41, 47], 10 seeds. NB06_gemma for SAE analysis.
- **Status:** `done`
- **Findings:** Layer 32 best (mean 0.9310). Orig: Synth 0.9987, Anth 0.9238, ToolACE 0.8706. ID: Synth 0.9975, Anth 0.9176, ToolACE 0.8571. Error counts (layer 32, seed 0): Synth 36/2000 (10FP, 26FN), Anth 589/2984 (91FP, 498FN), ToolACE 309/734 (1FP, 308FN). Sanity: Training AUROC 0.9998, random 0.4613. SAE analysis at layer 41 shows features are non-diagnostic (see RQ-5). Cross-lingual agreement worse than Llama (see RQ-8). (NB05_gemma + NB06_gemma, Feb 2026)
- **Decisions:** Fixed bfloat16 dtype issue (transformers#39972). Used Gemma Scope 2 SAE at layer 41.
- **Open issues:** None.

### RQ-8: Cross-model comparison

- **Question:** How do Llama 3.1 8B and Gemma 3 12B compare on probe quality, cross-lingual robustness, and SAE interpretability?
- **Why:** Determines whether findings generalize across model families or are architecture-specific.
- **Method:** Side-by-side comparison of NB05/NB05_gemma results and NB06/NB06_gemma SAE analyses.
- **Status:** `done`
- **Findings:**
  - **Probe quality:** Similar overall — Llama mean 0.9359, Gemma mean 0.9310 (orig datasets). Both peak in the upper-middle layers: Llama layer 12 (38% depth), Gemma layer 32 (67% depth).
  - **Cross-lingual robustness:** Llama significantly better. Cross-lingual agreement on Anthropic: Llama 73.1% both correct vs Gemma 56.8% both correct. Llama has 18.6% both wrong vs Gemma 27.6% both wrong. Gemma shows 14.3% orig-ok/id-fail (vs Llama 5.0%) — much more fragile on Indonesian.
  - **SAE interpretability:** Llama Scope features are semantically diagnostic (differential prevalence up to +15.3%). Gemma Scope 2 features at layer 41 are non-diagnostic (<0.1% differential) — features are multilingual token-level, not semantic concepts. SAE quality matters for mechanistic analysis.
  - **Error patterns:** Gemma has more Anthropic FPs (91 vs 51) and fewer FNs (498 vs 600). ToolACE similar (308 vs 344 FN, both near-zero FP). Gemma Synth slightly better (36 vs 41 errors).
- **Decisions:** None — comparative analysis complete.
- **Open issues:** None.

---

## Housekeeping Log

- **Feb 21, 2026:** LaTeX report corrected — Gemma layer sweep table had stale values from an earlier notebook run. Fixed using NB05_gemma outputs as sole source of truth. Corrected rows: layers 8, 16, 24, 31, 41, 47 (all six AUROC columns). Also corrected SAE section: layer 41 Orig Anth 0.9168 -> 0.9055, layer 31 comparison value 0.8876 -> 0.9209. LaTeX build artifacts (*.aux, *.bbl, *.blg, *.log, *.out) added to .gitignore.
