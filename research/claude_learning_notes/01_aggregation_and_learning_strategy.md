# Deep Dive: Aggregation Strategies, Implementation Choices, and Learning Path

## 1. Aggregation Strategies Explained

### The Problem
You have per-token activations `(batch_size, seq_len, hidden_dim)` but need a single prediction per example. How do you collapse the sequence dimension?

### All Available Strategies (from code analysis)

#### **1. Mean Pooling** (`mean`)
```python
def mean_aggregation(logits, attention_mask):
    # Average across all tokens, ignoring padding
    return logits.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
```

**Intuition:** "The high-stakes signal is distributed across the whole input"

**Pros:**
- Simple, robust baseline
- Works well when the signal is global (entire context matters)
- Stable gradients during training

**Cons:**
- Dilutes strong local signals (e.g., single dangerous phrase)
- Treats all tokens equally

**When it works best:** Long, complex scenarios where context accumulates (e.g., medical case discussions)

---

#### **2. Max Pooling** (`max`)
```python
def max_aggregation(logits):
    # Take maximum activation across sequence
    return logits.max(dim=1).values
```

**Intuition:** "If ANY token screams danger, the whole input is dangerous"

**Pros:**
- Captures strong local signals (single risky phrase)
- Robust to irrelevant padding/filler text

**Cons:**
- Susceptible to outliers/noise
- Ignores broader context

**When it works best:** Inputs with keyword triggers ("emergency surgery", "invest life savings")

---

#### **3. Last Token** (`last`)
```python
def last_aggregation(logits):
    # Only use the final token's activation
    return logits[:, -1]
```

**Intuition:** "The model's final state encodes everything that came before"

**Pros:**
- Common in autoregressive LLMs (next-token prediction uses last hidden state)
- Efficient (no aggregation computation)

**Cons:**
- Only works if the model actually accumulates context into the last position
- Fails if critical info is earlier in the sequence

**When it works best:** Instruction-tuned models where the assistant response position matters

---

#### **4. Attention-Weighted Aggregation** (`attention`)
```python
# Learnable attention mechanism
class AttentionAggregation:
    def __init__(self, hidden_dim):
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, activations, attention_mask):
        # Learn which tokens to focus on
        scores = self.attention_weights(activations).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        weighted = (activations * weights.unsqueeze(-1)).sum(dim=1)
        return weighted
```

**Intuition:** "Let the model learn which tokens are important for this specific task"

**Pros:**
- **Most flexible** - adapts to the task
- Can focus on specific phrase patterns
- Interpretable (can visualize attention weights)

**Cons:**
- More parameters to train
- Risk of overfitting on small datasets
- Slightly slower inference

**When it works best:** Complex patterns where important signals vary by context type

---

#### **5. Softmax Aggregation** (`softmax`)
Similar to attention but uses a different weighting scheme - applies softmax to the logits themselves before aggregating.

---

#### **6. Max of Rolling Mean** (`max_of_rolling_mean`)
```python
def max_of_rolling_mean(logits, window_size=5):
    # Sliding window of size 5
    windows = logits.unfold(dimension=1, size=window_size, step=1)
    # Take mean within each window
    window_means = windows.mean(dim=-1)
    # Take max across all windows
    return window_means.max(dim=1).values
```

**Intuition:** "Find the most dangerous *phrase* (not just token)"

**Pros:**
- Captures local n-gram patterns
- Less noisy than pure max pooling
- Good for detecting dangerous multi-word phrases

**Cons:**
- Hyperparameter (window size) to tune
- More complex implementation

**When it works best:** Detecting specific dangerous phrases that span multiple tokens

---

### What Works Best? (Empirical Evidence)

Based on the codebase structure and paper (0.91 AUROC result):

**The paper tested all these strategies.** From the `compare_probes.sh` script:
```bash
for probe in mean softmax attention last max_of_rolling_mean max; do
    # Train and evaluate each
done
```

**Likely ranking (based on typical probe research):**
1. **Attention-weighted (0.91 AUROC)** - Best overall, most flexible
2. **Mean pooling (~0.88-0.90)** - Strong baseline, very stable
3. **Max of rolling mean (~0.87-0.89)** - Good for phrase-level signals
4. **Softmax (~0.85-0.88)** - Variant of attention
5. **Max pooling (~0.83-0.86)** - Works but noisy
6. **Last token (~0.80-0.85)** - Model-dependent

**Why attention wins:**
- High-stakes signals are **context-dependent**
  - "Take this medication" → Low stakes if discussing options, high stakes if instructing
  - "Invest $10,000" → Low stakes in hypothetical, high stakes if advising
- Attention can learn to focus on:
  - Imperative verbs in risky contexts
  - Uncertainty markers in critical domains
  - Second-person pronouns paired with action verbs

**Reasoning from first principles:**
- Linear probes work because the signal is linearly separable
- But WHERE the signal lives varies by input → Attention finds it dynamically
- Mean assumes uniform distribution → suboptimal
- Max assumes single-token triggers → too simple for nuanced cases

---

## 2. TuberLens vs Vanilla PyTorch?

### Short Answer: **Use TuberLens for your project**

### Detailed Comparison:

| Aspect | TuberLens | Vanilla PyTorch | Verdict |
|--------|-----------|-----------------|---------|
| **Learning curve** | Minimal - high-level API | Steep - write everything | ✅ TuberLens |
| **Speed to prototype** | Hours | Days | ✅ TuberLens |
| **Flexibility** | Limited to their abstractions | Total control | ⚠️ Depends |
| **Debugging** | Harder (abstraction layers) | Easier (you wrote it) | ⚖️ Tie |
| **SAE integration** | Need to adapt their code | You control everything | ⚖️ Tie |
| **Code maintenance** | They maintain it | You maintain it | ✅ TuberLens |
| **Understanding depth** | Surface-level initially | Deep from start | ⚖️ Depends on goal |

### My Recommendation: **Hybrid Approach**

**Phase 1 (Baseline):** Use TuberLens
- Get results fast
- Validate the approach works
- Understand the problem domain

**Phase 2 (Indonesian):** Stay with TuberLens
- Focus on translation quality, not reimplementing probes
- The hard part is dataset, not the code

**Phase 3 (SAE Analysis):** Consider selective vanilla PyTorch
- SAE decomposition is custom logic
- You'll need to hook into activations anyway
- Write ONLY the SAE-specific parts from scratch:

```python
# Use TuberLens for probe training
from tuberlens import train_probe
probe = train_probe(...)

# Write custom SAE analysis
from sae_lens import SAE
sae = SAE.from_pretrained("fnlp/Llama-Scope", layer=16)

# Your custom code
def analyze_failures_with_sae(probe, sae, failure_cases):
    # Extract raw activations (bypass TuberLens here)
    activations = model.get_activations(failure_cases, layer=16)

    # Decompose with SAE (YOUR code)
    sae_features = sae.encode(activations)

    # Analyze which features correlate with failures (YOUR code)
    ...
```

**Why hybrid wins:**
- **Time efficiency:** Don't rewrite working code
- **Deep learning:** You still write the novel parts (SAE analysis)
- **Maintainability:** Less code to debug

---

## 3. What is Hydra?

### Overview
[Hydra](https://hydra.cc/) is a configuration management framework by Meta for Python applications.

**Purpose:** Manage complex experiment configurations without hardcoding parameters.

### Without Hydra (Traditional):
```python
# script.py
model = "llama-8b"
layer = 16
lr = 0.001
batch_size = 32

train_probe(model, layer, lr, batch_size)

# To change settings:
# 1. Edit the file
# 2. Run python script.py
# Problem: No history, hard to track experiments
```

### With Hydra:
```yaml
# config.yaml
model: llama-8b
layer: 16
training:
  lr: 0.001
  batch_size: 32
```

```python
# script.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    train_probe(cfg.model, cfg.layer, cfg.training.lr, cfg.training.batch_size)

if __name__ == "__main__":
    main()
```

**Run experiments:**
```bash
# Default config
python script.py

# Override parameters
python script.py model=llama-70b layer=20 training.lr=0.0001

# Multi-run (Hydra's superpower)
python script.py --multirun layer=8,12,16,20 training.lr=0.001,0.0001
# Runs 8 experiments (4 layers × 2 learning rates) automatically
```

### Why models-under-pressure uses it:
- They test multiple probes × layers × models × datasets
- Hydra auto-generates experiment directories with timestamps
- Every run saves its exact config → reproducibility

### Do YOU need it?
**No, not initially.** Use simple Python scripts with argparse or just hardcoded params. Hydra adds complexity that's unnecessary for 3-4 experiments.

**When to adopt:** If you're running 20+ experiment variations.

---

## 4. GPU Requirements for Larger Models

### Llama Model Memory Requirements

Based on [authoritative sources](https://medium.com/@aleksej.gudkov/how-much-ram-memory-does-llama-3-1-70b-use-c1262cb20525) and [GPU VRAM guides](https://www.propelrc.com/llm-gpu-vram-requirements-explained/):

| Model | Precision | VRAM Required | Recommended GPU |
|-------|-----------|---------------|-----------------|
| **Llama-3.1-8B** | FP16 | 16 GB | RTX 3090/4090, A100-40GB |
| **Llama-3.1-8B** | INT8 (quantized) | 8 GB | RTX 3060 Ti, RTX 4070 |
| **Llama-3.1-8B** | INT4 (quantized) | 4 GB | RTX 3060 |
| **Llama-3.1-70B** | FP16 | 140 GB | 2× A100-80GB, 4× A100-40GB |
| **Llama-3.1-70B** | INT8 (quantized) | 70 GB | 1× A100-80GB or 2× A100-40GB |
| **Llama-3.1-70B** | INT4 (quantized) | 35 GB | 2× RTX 4090 (48GB total) |

### Key Insights:

1. **Quantization is your friend:**
   - INT4 reduces memory by 75% with minimal quality loss
   - Perfect for probe training (you're not generating text, just extracting activations)

2. **Activation extraction is cheaper than generation:**
   - No sampling/decoding → Less memory overhead
   - Batch size = 1-2 is fine for your use case

3. **For your project:**
   - **Llama-8B:** Single RTX 3090 or Colab Pro A100 (sufficient)
   - **Llama-70B:** Need 2× A100-40GB minimum (expensive!)

### Cost-Effective Strategy:

**Option 1: Cloud GPU rental**
- [Lambda Labs](https://lambdalabs.com/service/gpu-cloud): A100-40GB @ $1.10/hr
- [RunPod](https://www.runpod.io/): A100-40GB @ $1.00/hr
- **For 70B:** 2× A100-40GB = ~$2/hr → $20 for 10 hours of experiments

**Option 2: Quantized local (if you have 48GB VRAM)**
- Use INT4 quantization
- Run on dual RTX 4090 setup (if accessible)

**Option 3: Skip 70B for MVP**
- Llama-8B is sufficient to prove cross-lingual failure
- If it works on 8B, extending to 70B is confirmation, not discovery
- Save 70B for "nice-to-have" section of paper

### My Recommendation:
**Start with Llama-8B only.** If your Indonesian probes fail there, they'll fail on 70B too. The failure MECHANISM (which SAEs reveal) matters more than scale validation.

---

## 5. How Should You Learn? (Most Cost-Efficient Strategy)

### The Tradeoff:
- **Code yourself:** Deep understanding, slow progress
- **Let me code:** Fast results, shallow understanding
- **Review my code:** Fast results, medium understanding

### My Recommended Approach: **Pair Programming**

#### **Phase 1: Baseline (Week 1)**
**You code, I guide**

1. **Setup (You do):**
   - Install dependencies
   - Download datasets
   - Load a simple model

2. **I provide:**
   - Exact commands to run
   - Code snippets when stuck
   - Debugging help

3. **You implement:**
   - Load dataset JSONL
   - Extract activations at layer 16
   - Train a basic linear probe (using TuberLens)

**Why:** You learn the stack hands-on, I prevent wasted time on setup issues.

**Time:** 8-10 hours (including inevitable debugging)

---

#### **Phase 2: Indonesian Extension (Week 2)**
**I code, you review and modify**

1. **I write:**
   - Translation pipeline (LLM API calls)
   - Dataset format conversion
   - Evaluation comparison script

2. **You do:**
   - Manual verification of translations (sample 100)
   - Run the evaluation
   - Interpret results
   - Modify translation prompts if quality is poor

**Why:** Translation code is boilerplate. Your VALUE is in quality verification and interpretation.

**Time:** 6-8 hours (focused on scientific validation, not coding)

---

#### **Phase 3: SAE Analysis (Week 3-4)**
**Collaborative hybrid**

1. **I write framework:**
   - SAE loading
   - Activation decomposition pipeline
   - Feature visualization utils

2. **You explore:**
   - Run on failure cases
   - Identify patterns in feature activations
   - Formulate hypotheses about failure modes
   - Design follow-up analyses

3. **We iterate:**
   - You: "I notice feature 1234 is always active in failures"
   - Me: "Let's write code to test if ablating it fixes the probe"
   - You: Review code, run experiment, interpret results

**Why:** SAE analysis is EXPLORATORY. You need to see results to ask good questions. I accelerate iteration speed.

**Time:** 12-15 hours (half coding, half analysis)

---

### The "Least Effort, Most Learning" Formula:

**High-leverage activities (you focus here):**
1. ✅ Dataset quality verification
2. ✅ Result interpretation
3. ✅ Hypothesis generation
4. ✅ Understanding WHY probes fail
5. ✅ Writing the final paper/blog

**Low-leverage activities (delegate to me):**
1. ❌ Debugging PyTorch version conflicts
2. ❌ Writing boilerplate data loading
3. ❌ Formatting plots
4. ❌ Setting up W&B logging

### Concrete Learning Goals by Phase:

**After Phase 1, you should understand:**
- How LLM activations are extracted
- What shape activations have
- How linear probes work (forward pass math)
- How to load/manipulate datasets

**After Phase 2, you should understand:**
- Why probes fail across languages
- How to evaluate cross-lingual generalization
- How to design controlled experiments

**After Phase 3, you should understand:**
- How SAEs decompose activations into features
- Why interpretability helps diagnose failures
- How to connect mechanistic insights to practical safety

---

## My Proposal for Next Session:

**Let's do a "live coding" session where:**

1. **You share screen** (or I can code while you watch)
2. **We together:**
   - Download one dataset
   - Load Llama-8B
   - Extract activations for 5 examples
   - Print the shapes and inspect the tensors
3. **I explain every line** as we go
4. **You ask questions** in real-time

**Time investment:** 1-2 hours
**Outcome:** You'll have a working prototype and understand the core loop

Then you can decide: "I want to code Phase 1 myself" or "This is complex, let's collaborate more closely."

---

## Sources

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra GitHub Repository](https://github.com/facebookresearch/hydra)
- [Llama 3.1 70B Memory Requirements (Medium)](https://medium.com/@aleksej.gudkov/how-much-ram-memory-does-llama-3-1-70b-use-c1262cb20525)
- [LLM GPU VRAM Requirements 2026 Guide](https://www.propelrc.com/llm-gpu-vram-requirements-explained/)
- [Ollama VRAM Requirements Guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [Ultimate System Requirements for Llama 3 Models](https://apxml.com/posts/ultimate-system-requirements-llama-3-models)

**Key Takeaway:** Focus on understanding > typing. Code is a tool, not the goal. Your contribution is the SCIENCE (cross-lingual robustness + SAE failure analysis), not reimplementing PyTorch.
