# Llama 3.1 Language Support: What It Means for Cross-Lingual Probing

## Official Language Support

Llama 3.1 officially supports **8 languages**:

| Language | Code | Script |
|----------|------|--------|
| English | en | Latin |
| German | de | Latin |
| French | fr | Latin |
| Italian | it | Latin |
| Portuguese | pt | Latin |
| Hindi | hi | Devanagari |
| Spanish | es | Latin |
| Thai | th | Thai |

These are the languages Meta specifically optimized for -- they receive dedicated benchmarking, safety testing, and quality assurance. The model card states: "Llama 3.1 has been tested for English and these additional languages."

Notable: 6 of 8 are European languages using Latin script. Hindi and Thai are the only non-Latin-script languages with official support. No Southeast Asian language besides Thai is included, despite the region having ~700 million people.

## The Training Data Picture

Llama 3.1 was trained on **~15.6 trillion tokens** with the following data mix:
- ~50% general knowledge
- ~25% mathematical and reasoning data
- ~17% code
- **~8% multilingual tokens**

The 8% multilingual portion covers approximately **34 languages**. Meta's paper ("The Llama 3 Herd of Models", arXiv:2407.21783) does not provide per-language token counts for the non-official languages. What we can infer:

1. The 8 official languages likely consume most of the multilingual budget
2. The remaining ~26 languages share a smaller fraction
3. Meta explicitly warns: "We do not expect the same level of performance in these languages as in English"

### The Language Support Spectrum

This creates a three-tier system:

```
Tier 1 (official):     EN, DE, FR, IT, PT, HI, ES, TH
                        - Benchmarked, safety-tested, optimized
                        - Substantial training data

Tier 2 (seen):          ~26 other languages including Indonesian, Arabic,
                        Chinese, Japanese, Korean, Vietnamese, etc.
                        - Present in training data
                        - Not benchmarked or optimized
                        - Quality not guaranteed

Tier 3 (minimal/none):  Languages with very little or no web presence
                        - Essentially zero-shot from related languages
                        - Unpredictable behavior
```

Indonesian sits firmly in Tier 2: enough exposure to develop basic capability, but not enough to match Tier 1 quality.

## Where Indonesian Sits Specifically

Indonesian (Bahasa Indonesia) has some characteristics that affect how well it might be represented:

**Favorable factors:**
- Uses Latin script (same as 6 of 8 official languages) -- no script barrier
- ~200 million speakers, significant web presence
- Relatively simple morphology (no grammatical gender, no conjugation)
- Many loanwords from English, Dutch, Arabic, Sanskrit

**Unfavorable factors:**
- Not in the official 8
- Likely a small fraction of the ~8% multilingual budget
- Indonesian-specific concepts, idioms, and cultural context may be underrepresented
- Closely related to Malay, which may cause confusion in the model's representations

## Community Fine-Tuning Projects

The gap between base capability and production needs has spawned several community projects:

### 1. Sahabat-AI (GoToCompany, Indonesia)

**Model:** `GoToCompany/llama3-8b-cpt-sahabatai-v1-base`

GoToCompany (formerly GoJek's tech arm, one of Indonesia's largest tech companies) continued pretraining Llama 3 8B with:
- ~448K Indonesian instruction pairs
- Additional Javanese and Sundanese data (regional languages with ~100M speakers combined)

The fact that they needed **continued pretraining** (not just fine-tuning) tells us the base model's Indonesian is functional but not production-grade. Continued pretraining adds more Indonesian tokens to the model's training, improving its internal representations -- this is a stronger intervention than instruction tuning alone.

### 2. SEA-LION v3 (AI Singapore)

**Model:** `aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct`

AI Singapore's Southeast Asian Language Intelligence and Open Network (SEA-LION) project:
- Takes Llama 3.1 8B as a base
- Continues pretraining on Southeast Asian language data (Indonesian, Malay, Thai, Vietnamese, Tagalog, etc.)
- Adds instruction tuning for the region

Their SEA-HELM benchmark provides systematic evaluation across Southeast Asian languages, consistently showing that fine-tuning substantially improves performance. The improvements are not marginal -- they're significant enough to justify the compute cost.

### 3. MERaLiON (National University of Singapore)

**Model:** `MERaLiON/LLaMA-3-MERaLiON-8B-Instruct`

Focuses specifically on English, Chinese, and Indonesian -- targeting Singapore's trilingual needs. Built from Llama 3 8B with curated multilingual instruction data.

### What These Projects Tell Us

All three projects follow the same pattern: **the base model has enough Indonesian to build on, but not enough to deploy directly**. This is important for our experiment:

1. **Internal representations exist** -- the model does form representations for Indonesian text
2. **Those representations are weaker** -- they need additional training to reach production quality
3. **The architecture can learn** -- fine-tuning works, so the model's capacity can support Indonesian

The question for our probing experiment becomes: are the base model's Indonesian representations structured enough that a probe trained on English/German/French/Hindi activations can still detect "high-stakes" concepts? Or is the Indonesian representation too noisy/different for the linear decision boundary to transfer?

## What This Means for Internal Representations

### The Multilingual Representation Hypothesis

Research on multilingual models (from multilingual BERT onwards) shows that LLMs develop **partially shared conceptual spaces** across languages. The intuition: if the model learns that "danger" and "Gefahr" (German) both predict similar continuations, it starts mapping them to nearby regions in activation space. Over enough training, abstract concepts become partially language-invariant.

Evidence for this:
- **Cross-lingual transfer** works in practice (fine-tune on English, evaluate on German)
- **Probing studies** show similar feature structures across languages in well-supported language pairs
- **Translation emerges** without explicit training -- models can translate between languages they've only seen separately

### But It's Not Perfect

The shared space is approximate, not exact. Several factors cause language-specific divergence:

1. **Training data imbalance** -- Languages with more data develop richer, more structured representations. English dominates, so the conceptual space is organized primarily around English semantics.

2. **Script and tokenization effects** -- Different scripts tokenize differently. Indonesian uses Latin script (good for us -- same tokenizer efficiency as English), but languages like Hindi or Thai get tokenized into more pieces, changing the activation patterns.

3. **Grammatical structure** -- Word order, morphology, and syntax affect how information flows through transformer layers. Indonesian (SVO, no inflection) is structurally closer to English than, say, Japanese (SOV).

4. **Cultural framing** -- "High-stakes" situations may be expressed differently across cultures. An Indonesian text about a medical emergency might use different framing than an English one, even if the underlying stakes are the same.

### The Prediction for Our Experiment

For a **Tier 1 language** like German (official, ample training data, same script):
- Expect high overlap with English conceptual space
- Probe trained on multilingual data should transfer well
- Prediction: minimal AUROC degradation

For **Indonesian** (Tier 2, seen but not optimized, same script):
- Expect partial overlap -- abstract concepts still similar, but noisier
- More variance in how the same concept is represented
- Prediction: moderate AUROC degradation (0.65-0.80 range)
- The degradation could come from:
  - (a) **Weaker concept encoding** -- model doesn't strongly represent "high-stakes" in Indonesian
  - (b) **Geometric shift** -- the concept exists but in a different part of activation space that the probe's linear boundary doesn't cover
  - (c) Both

## Implications for Our Experiment Design

### Why Indonesian Is a Good Choice

1. **Genuinely novel test** -- No published work on cross-lingual probing for high-stakes detection
2. **Informative middle ground** -- Not trivially easy (like German) or trivially hard (like Klingon). The result will be genuinely informative.
3. **Practical relevance** -- Indonesia has 270M+ people, growing AI deployment, and limited safety tooling. If probes don't transfer, that's a real gap.
4. **Same script advantage** -- Latin script means tokenization is efficient and we avoid confounding script effects with language effects.

### Existing Multilingual Signal in Training Data

Our training data is already multilingual: ~59% EN, ~14% DE, ~14% FR, ~13% HI. This means the probe has already learned some cross-lingual invariance. Before testing Indonesian, we should check the per-language performance within the test set to establish a baseline:
- If the probe already performs differently on DE/FR/HI, that tells us the cross-lingual generalization is imperfect even within Tier 1 languages
- The degradation gradient (EN -> DE/FR -> HI -> ID) would be informative

### SAE Connection

If the probe fails on Indonesian, SAE decomposition can reveal whether:
- The model activates "high-stakes" features for Indonesian text (concept exists, probe can't find it = geometric problem)
- The model doesn't activate those features (concept not encoded = representational problem)

This distinction matters for safety: a geometric problem is fixable with more diverse training data; a representational problem requires model-level changes.

---

## References

- Meta AI, "The Llama 3 Herd of Models" (arXiv:2407.21783), 2024
- Llama 3.1 Model Card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
- Sahabat-AI: https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base
- SEA-LION v3: https://huggingface.co/aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct
- MERaLiON: https://huggingface.co/MERaLiON/LLaMA-3-MERaLiON-8B-Instruct
- Pires et al., "How Multilingual is Multilingual BERT?" (ACL 2019)
- Conneau et al., "Emerging Cross-lingual Structure in Pretrained Language Models" (ACL 2020)
