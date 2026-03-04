# Final Touches — Things to Think About

Items to consider for the third draft, pending Lambda results.

---

## 1. Cross-Architecture Depth Comparison

Both models peak at different relative depths: Llama at ~38% (L12/32), Gemma at ~67% (L32/48).
At matched 50% depth, performance is nearly identical (0.9277 vs 0.9288).

- Why does Llama peak earlier? Smaller model = features form sooner?
- The NB06 early-vs-late runs (L12 vs L26, L24 vs L41) should clarify whether late-layer SAE features are fundamentally different or just noisier
- Worth framing as: "where in the network does high-stakes representation live?"

## 2. Error Asymmetry (FN >> FP)

Both models are heavily FN-dominated on real-world data (Anthropic: 11.8x for Llama, 5.5x for Gemma; ToolACE: effectively all FN).

- What metric should we recommend for deployment? TPR@FPR thresholds vs fixed 0.5?
- ToolACE is a domain mismatch problem, not a threshold problem — should we say this explicitly?
- Synthetic AUROC >0.998 is misleading as a field performance proxy — how strongly to flag this?

## 3. SAE Diagnostics Across Layers and Models

Llama L12 SAE was diagnostic (features 6989, 16646 at 14-15% differential). Gemma L41 was not (L0~1614, all features <0.1% differential).

- [ ] Does Gemma L24 produce sparser, more diagnostic features? (L0 expected ~100-300)
- [ ] Does Llama L26 lose diagnostic power compared to L12? (would confirm "early = better for SAE")
- [ ] Cross-lingual feature overlap: do EN and ID errors activate the same SAE features?
- If both early layers are diagnostic and both late layers aren't, that's a clean story about where safety-relevant representations form

## 4. Native Indonesian Synthetic Data

NB07 generates ~250 native Indonesian prompts (standard + failure-mode-targeted).

- [ ] Does the probe perform differently on native vs translated Indonesian?
- [ ] Do failure-mode prompts (humor, signal words, context mixing) actually fool the probe?
- If native ID performs worse than translated ID, that suggests translation preserves English-like structure that the probe relies on

## 5. Framing and Narrative

- Main story: probes are cross-lingually robust (~0.6% gap) but domain-fragile (ToolACE)
- SAE analysis adds *why*: early-layer features encode situational stakes, late-layer features don't
- What's the takeaway for practitioners? "Train on diverse domains, not diverse languages"?
- How much weight to give the negative Gemma SAE result? It's informative but risks reading as "we tried and it didn't work"

---

*Last updated: 2026-03-03*
