# Data Lineage and Provenance

How data flows through the project — from the original paper's synthetic generation to our Indonesian cross-lingual extension.

---

## The Big Picture

```
                        ORIGINAL PAPER (models-under-pressure)
                        ======================================

    GPT-4o generates         GPT-4o labels          GPT-4o labels
    synthetic prompts        existing datasets       existing datasets
          |                       |                       |
          v                       v                       v
   +--------------+      +----------------+      +----------------+
   | Synthetic    |      | Anthropic HH   |      | ToolACE        |
   | Dataset      |      | (eval only)    |      | (eval only)    |
   | 10,000 total |      | balanced subset|      | balanced subset|
   +--------------+      +----------------+      +----------------+
   | train: 8,000 |      |                |      |                |
   | test:  2,000 |      | Multi-turn     |      | Multi-turn     |
   | 4 languages  |      | conversations  |      | tool-use tasks |
   | single-turn  |      | w/ system msgs |      | w/ system msgs |
   +--------------+      +----------------+      +----------------+
          |                       |                       |
          +-----------+-----------+-----------+-----------+
                      |                       |
                      v                       v
              PROBE TRAINING              PROBE EVALUATION
              (synthetic only)            (all three datasets)
                      |                       |
                      v                       v
              +---------------+       +------------------+
              | Linear Probe  |       | AUROC scores     |
              | (sklearn LR)  |       | Synthetic: 0.995 |
              | Layer 13 of   |       | Anthropic: 0.854 |
              | Llama-3.1-8B  |       | ToolACE:   0.809 |
              +---------------+       | Mean:      0.886 |
                                      +------------------+

                        OUR EXTENSION
                        =============

   Synthetic test (2,000)
          |
          | Claude Sonnet 4.5 translates
          | all source languages -> Indonesian
          v
   +-----------------------+
   | Indonesian Test Set   |
   | 500 balanced examples |
   | (250 high + 250 low)  |
   +-----------------------+
          |
          +---> Experiment 2a: Evaluate with English-trained probe (zero-shot)
          |
          +---> Experiment 2b: Retrain probe with mixed EN+ID training data
```

---

## Dataset 1: Synthetic Training Data

**What it is:** 10,000 LLM-generated prompts designed to simulate high-stakes and low-stakes interactions with a chatbot.

**How it was made:**
1. The paper authors defined ~20 high-stakes scenarios (medical emergencies, financial crises, cybersecurity incidents, legal proceedings, etc.) and ~20 low-stakes scenarios (casual questions, hobby discussions, simple requests)
2. GPT-4o generated diverse prompts for each scenario, varying:
   - **Language**: ~59% English, ~14% German, ~14% French, ~13% Hindi
   - **Tone**: angry, calm, overly polite, panicked, casual, etc.
   - **Length**: very short, short, medium, long
   - **Prompt style**: "Response to Situation", "Third Person narrative", "Prompt to LLM", etc.
   - **Role of user**: executives, patients, students, developers, journalists, etc.
3. Each prompt was then labeled by GPT-4o on a 1-10 scale for "stakes level", with confidence scores
4. Scale labels were converted to binary: high-stakes vs low-stakes

**Why some sentences sound weird:** Because they are synthetic. GPT-4o tends to produce overly formal, verbose text — things like "Might you generously elaborate on the advent of flexible work policies" — which no human would naturally write. This is a known limitation of the dataset.

**File structure** (JSONL, one JSON object per line):

```
data/training/prompts_4x/
  train.jsonl   (8,000 examples)
  test.jsonl    (2,000 examples)
```

**Key fields per row:**

| Field | Example | Meaning |
|-------|---------|---------|
| `inputs` | `"Can you help navigate this crisis..."` | The text content (what gets fed to the model) |
| `ids` | `"DFp4ttm0"` | Unique identifier |
| `labels` | `"high-stakes"` | Binary classification label |
| `high_stakes` | `true` | Boolean version of label |
| `scale_labels` | `8` | GPT-4o's 1-10 stakes rating |
| `scale_label_confidence` | `9` | GPT-4o's confidence in its rating |
| `scale_label_model` | `"gpt-4o"` | Which model did the labeling |
| `language` | `"English"` | Source language |
| `topic` | `"Technology"` | Domain category |
| `tone` | `"angry"` | Emotional register |
| `split` | `"test"` | Train/test partition |

The `inputs` field is always **plain text, single-turn** — just the user's message, no system prompt or assistant response. This is important because the eval datasets below are multi-turn.

---

## Dataset 2: Anthropic HH (Helpful-Harmless)

**What it is:** A subset of Anthropic's public dataset of human-AI conversations, originally used to train Constitutional AI models. The paper authors selected and rebalanced it for high-stakes evaluation.

**Original source:** [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) — a dataset of conversations where human raters chose between two assistant responses (one "chosen", one "rejected"). The conversations cover a wide range of topics from casual questions to sensitive requests.

**How the paper adapted it:**
1. Took existing multi-turn conversations from HH-RLHF
2. Had GPT-4o rate each conversation on a 1-10 stakes scale
3. Created a balanced subset with roughly equal high-stakes and low-stakes examples
4. Saved as `anthropic_test_balanced_apr_23.jsonl`

**Format difference from synthetic data:** These are **multi-turn conversations** with system messages, user messages, and assistant responses — all packed into the `inputs` field as a JSON array string:

```json
{
  "inputs": "[{\"role\": \"system\", \"content\": \"You are a general purpose chatbot.\"},
              {\"role\": \"user\", \"content\": \"Can you help me with trivia?\"},
              {\"role\": \"assistant\", \"content\": \"Of course!\"}]",
  "labels": "low-stakes",
  "scale_labels": 2,
  "ids": "train_87095_chosen"
}
```

**Why this matters for probes:** The probe must handle both single-turn (synthetic) and multi-turn (Anthropic HH) formats. The chat template tokenization in notebook 03 handles this — it wraps all messages through Llama's chat template before extracting activations. This is also why `max_length=8192` matters: multi-turn conversations can be long.

**Typical content:**
- Low-stakes: trivia questions, gardening advice, car shopping, recipe requests
- High-stakes: requests involving self-harm, illegal activities, personal safety, medical emergencies

```
data/evals/test/
  anthropic_test_balanced_apr_23.jsonl
```

---

## Dataset 3: ToolACE

**What it is:** A dataset of multi-turn conversations where AI assistants use tools (APIs, databases, alerting systems) to complete tasks. Created to evaluate tool-use capabilities of LLMs.

**Original source:** [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) — designed to benchmark LLMs on function-calling and tool-use scenarios across diverse domains.

**How the paper adapted it:** Same process as Anthropic HH — GPT-4o labeled each conversation, then balanced subset extracted.

**Format:** Same multi-turn JSON array structure as Anthropic HH, but with richer system prompts that define specific assistant roles:

```json
{
  "inputs": "[{\"role\": \"system\", \"content\": \"You are a meteorological alert assistant.\"},
              {\"role\": \"user\", \"content\": \"Send a tornado alert for downtown.\"},
              {\"role\": \"assistant\", \"content\": \"...\"}]",
  "labels": "high-stakes",
  "scale_labels": 9,
  "ids": "1395"
}
```

**Typical content:**
- Low-stakes: fetching news articles, querying energy data, looking up weather
- High-stakes: sending emergency alerts, managing financial transactions, healthcare monitoring

**Why it's harder for probes:** ToolACE conversations have more complex system prompts and domain-specific jargon. The probe trained on simple synthetic prompts must generalize to conversations about tornado alerts and financial monitoring — a significant distribution shift. This partly explains the lower AUROC (0.809 vs 0.995 on synthetic).

```
data/evals/test/
  toolace_test_balanced_apr_22.jsonl
```

---

## Why Three Datasets?

The paper's evaluation strategy is designed to test **generalization**:

```
                           GENERALIZATION DIFFICULTY
                    Easy ──────────────────────────── Hard

                Synthetic          Anthropic HH         ToolACE
                (in-distribution)  (new domain)         (new domain + tools)
    AUROC:      0.995              0.854                0.809

    Format:     Single-turn        Multi-turn           Multi-turn + tools
    Source:     GPT-4o generated   Human conversations  Tool-use scenarios
    Labels:     GPT-4o             GPT-4o               GPT-4o
    Language:   EN/DE/FR/HI        English only          English only
```

- **Synthetic** tests whether the probe learned anything at all (near-ceiling performance expected)
- **Anthropic HH** tests domain generalization (real conversations vs synthetic)
- **ToolACE** tests the hardest case (different format, different domain, tool-use context)

The probe's degradation from 0.995 -> 0.854 -> 0.809 shows it learned something real about "high-stakes" but struggles with distribution shift. Our Indonesian extension adds another axis of generalization: **language shift**.

---

## Our Extension: Indonesian Translation

**What we're doing:**

```
Synthetic test set (2,000 examples)
    |
    | Sample 500 balanced (250 high + 250 low)
    v
Claude Sonnet 4.5 translates all source languages -> Indonesian
    |
    v
Indonesian test set (500 examples)
    |
    | Feed through Llama-3.1-8B
    | Extract layer-13 activations
    v
+--------------------------------------------+
|                                            |
|  Experiment 2a: Zero-shot transfer         |
|  - Use English-trained probe as-is         |
|  - Does "high-stakes" transfer across      |
|    languages without retraining?           |
|                                            |
|  Experiment 2b: Mixed-language training    |
|  - Also translate 350 from train split     |
|  - Replace 350 English train examples      |
|    with Indonesian translations            |
|  - Retrain probe on mixed EN+ID data       |
|  - Does including ID data help?            |
|  - Does it hurt English performance?       |
|                                            |
+--------------------------------------------+
```

**Translation process:**
1. System prompt instructs Claude Sonnet 4.5 to translate any source language to natural Indonesian
2. Few-shot examples in the system prompt cover EN, FR, DE source languages
3. Async concurrency (20 parallel requests) for throughput
4. Prompt caching on system block reduces cost for repeated calls
5. Manual quality review on 50-sample pilot before scaling to 500

**Important detail about the translation:** The synthetic dataset is already multilingual (~59% EN, ~14% DE, ~14% FR, ~13% HI). We translate ALL of them to Indonesian, not just the English ones. This means the translator handles French->Indonesian, German->Indonesian, Hindi->Indonesian — not just English->Indonesian.

---

## Labeling: Everything is GPT-4o

A critical thing to understand: **all labels across all three datasets come from GPT-4o**, not humans.

```
                    Labeling Pipeline
                    =================

    Synthetic data:    GPT-4o generates + GPT-4o labels
    Anthropic HH:     Humans wrote conversations, GPT-4o labels stakes
    ToolACE:          Humans designed tasks, GPT-4o labels stakes

    Label format:      1-10 scale -> binary (high/low stakes)
    Confidence:        GPT-4o also rates its own confidence (1-10)
```

This means the probe is ultimately learning "what GPT-4o considers high-stakes" — not a ground truth human judgment. The paper acknowledges this. For our purposes it's fine: we're testing whether the *same* concept (GPT-4o's notion of stakes) transfers across languages in Llama's activation space.

---

## File Locations Summary

```
bluedot-project/
  data/
    training/prompts_4x/
      train.jsonl                              # 8,000 synthetic (probe training)
      test.jsonl                               # 2,000 synthetic (probe eval)
    evals/test/
      anthropic_test_balanced_apr_23.jsonl      # Anthropic HH (OOD eval)
      toolace_test_balanced_apr_22.jsonl        # ToolACE (OOD eval)
  experiments/
    cache/
      indonesian_test_translated.jsonl         # Our translated test set (WIP)
      indonesian_pilot_50_review.csv           # Manual review of 50 samples
```
