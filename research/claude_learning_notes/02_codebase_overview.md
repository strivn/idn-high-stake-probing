# Comprehensive Codebase Overview: models-under-pressure & TuberLens

This document provides a detailed analysis of how the high-stakes probe code works, what you need to understand, and how to use it for your project.

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Core Concepts](#core-concepts)
3. [models-under-pressure Deep Dive](#models-under-pressure-deep-dive)
4. [TuberLens Simplified Architecture](#tuberlens-simplified-architecture)
5. [Key Differences](#key-differences)
6. [What You Need for Your Project](#what-you-need-for-your-project)

---

## High-Level Architecture

### The Probe Training Pipeline (End-to-End)

```
Input Dataset (JSONL)
        ↓
    [Tokenize & Load]
        ↓
    LLM Model (e.g., Llama-3.1-8B)
        ↓
    [Extract Activations at Layer N]
        ↓
    Activations: (batch_size, seq_len, hidden_dim)
        ↓
    [Aggregate across sequence] ← Different strategies: mean, max, attention-weighted, etc.
        ↓
    Aggregated: (batch_size, hidden_dim)
        ↓
    [Train Linear Probe]  ← Simple classifier: activations → high-stakes probability
        ↓
    Trained Probe (Linear/MLP/Attention)
        ↓
    [Evaluate on Test Sets]
        ↓
    Metrics: AUROC, Accuracy, TPR@FPR
```

---

## Core Concepts

### 1. **Activations**
**What they are:** The internal hidden states of the LLM at a specific layer when processing text.

**Shape:** `(batch_size, sequence_length, hidden_dimension)`
- `batch_size`: Number of examples processed together
- `sequence_length`: Number of tokens in the input
- `hidden_dimension`: Size of the model's representation (e.g., 4096 for Llama-8B)

**Why they matter:** Activations encode the model's "understanding" of the input. High-stakes scenarios trigger different activation patterns than low-stakes ones.

**Example:**
```python
# Input: "Should I take this experimental medication?"
# Llama-3.1-8B at layer 16 produces:
activations.shape  # (1, 48 tokens, 4096 dimensions)
```

### 2. **Probes**
**What they are:** Small classifiers (usually linear) trained to detect a concept from activations.

**Types in the codebase:**
- **Linear probes:** Single linear layer (most common)
- **MLP probes:** Multiple layers with nonlinearities
- **Attention probes:** Use attention mechanism to aggregate across tokens

**Why linear works:** The paper found that linear probes achieve ~0.91 AUROC, suggesting high-stakes information is linearly separable in activation space.

### 3. **Aggregation Strategies**
**Problem:** Activations are per-token `(batch, seq_len, hidden_dim)`, but we need a single prediction per input.

**Solutions:**
1. **Mean pooling:** Average across all tokens
2. **Max pooling:** Take maximum activation per dimension
3. **Attention-weighted:** Learn which tokens to focus on
4. **Last token:** Use only the final token's activation
5. **Softmax aggregation:** Softmax-weighted average

**In the code:**
```python
# Mean aggregation
def mean_aggregation(activations, attention_mask):
    # activations: (batch, seq_len, hidden)
    # attention_mask: (batch, seq_len) - marks real vs padding tokens
    masked = activations * attention_mask.unsqueeze(-1)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    # Returns: (batch, hidden)
```

### 4. **Dataset Format**

**JSONL structure (one example per line):**
```json
{
  "input": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Should I invest my life savings in crypto?"}
  ],
  "id": "example_001",
  "labels": "high-stakes",
  "metadata": "finance"
}
```

**After loading into Dataset object:**
```python
dataset = LabelledDataset.load_from("data.jsonl")
# dataset.inputs: List of dialogues
# dataset.ids: List of unique IDs
# dataset.other_fields["labels"]: Array of labels (0=low-stakes, 1=high-stakes)
# dataset.other_fields["metadata"]: Any additional fields
```

---

## models-under-pressure Deep Dive

### Directory Structure
```
models_under_pressure/
├── config.py                    # Global settings, paths, model names
├── model.py                     # LLM loading and activation extraction
├── activation_store.py          # Caching activations to disk/cloud
├── dataset_utils.py             # Loading, splitting, filtering datasets
├── interfaces/
│   ├── dataset.py              # Dataset classes (BaseDataset, LabelledDataset)
│   ├── activations.py          # Activation container class
│   ├── probes.py               # ProbeSpec configuration
│   └── results.py              # Result containers
├── probes/
│   ├── base.py                 # Abstract Probe interface
│   ├── sklearn_probes.py       # Sklearn-based probes
│   ├── pytorch_probes.py       # PyTorch-based probes
│   ├── pytorch_classifiers.py  # Neural network classifiers
│   ├── pytorch_modules.py      # Attention modules
│   ├── probe_factory.py        # Factory for creating probes
│   └── aggregations.py         # Aggregation functions
├── experiments/
│   ├── evaluate_probes.py      # Main evaluation script
│   ├── cross_validation.py     # Layer selection via CV
│   └── calibration.py          # Probe calibration analysis
└── scripts/
    └── compare_probes.sh        # Run all probe types
```

### Key Components Explained

#### 1. **Activation Extraction** (`model.py`)

```python
class LLMModel:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_activations(self, inputs: List[Dialogue], layer: int) -> Activation:
        """
        Extract activations at a specific layer.

        Process:
        1. Tokenize inputs using chat template
        2. Forward pass through model
        3. Hook into layer N to extract hidden states
        4. Return activations + metadata (input_ids, attention_mask)
        """
        # Apply chat template
        formatted = [self.tokenizer.apply_chat_template(inp) for inp in inputs]

        # Tokenize
        encoded = self.tokenizer(formatted, padding=True, return_tensors="pt")

        # Register hook to capture layer N's output
        activations = []
        def hook(module, input, output):
            activations.append(output[0])  # output[0] is hidden states

        handle = self.model.model.layers[layer].register_forward_hook(hook)

        # Forward pass
        with torch.no_grad():
            self.model(**encoded)

        handle.remove()

        return Activation(
            activations=torch.stack(activations),
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask
        )
```

**Important:** Different model architectures (Llama, Gemma, GPT) have different layer access patterns. The `ModelArchitecture` classes handle this abstraction.

#### 2. **Activation Storage** (`activation_store.py`)

**Why it exists:** Computing activations is expensive (requires full forward pass). Cache them for reuse.

```python
class ActivationStore:
    def store(self, spec: ActivationsSpec, activations: Activation):
        """
        Save activations to disk/cloud.

        Format: Compressed with zstandard (.zst)
        Location: Local cache + Cloudflare R2 bucket (optional)
        Manifest: JSON tracking all stored activations
        """

    def load(self, spec: ActivationsSpec) -> Activation:
        """Load cached activations if available."""
```

**For your project:** You might skip R2 and just cache locally to avoid infrastructure setup.

#### 3. **Probe Training** (`probes/pytorch_probes.py`)

```python
class PytorchProbe:
    def fit(self, dataset: LabelledDataset, validation_dataset: LabelledDataset):
        """
        Train the probe.

        Steps:
        1. Extract activations from dataset
        2. Aggregate across sequence (mean/max/attention)
        3. Train linear classifier: activations → probability
        4. Early stopping based on validation loss
        """
        # Get activations
        activations = Activation.from_dataset(dataset)
        # Shape: (batch, seq_len, hidden)

        # Aggregate
        aggregated = self.aggregation_fn(activations)
        # Shape: (batch, hidden)

        # Train
        self.classifier.train(aggregated, dataset.labels_torch())

    def predict_proba(self, dataset: BaseDataset) -> np.ndarray:
        """Predict high-stakes probability for each example."""
        activations = self.aggregation_fn(Activation.from_dataset(dataset))
        return self.classifier.probs(activations)
        # Returns: (batch_size,) array of probabilities
```

#### 4. **Evaluation Pipeline** (`experiments/evaluate_probes.py`)

```python
def evaluate_probe_and_save_results(
    probe: Probe,
    eval_dataset: LabelledDataset,
    fpr: float = 0.01
) -> DatasetResults:
    """
    Evaluate probe on a dataset.

    Returns:
    - AUROC: Area under ROC curve
    - Accuracy: % correct predictions (threshold=0.5)
    - TPR@FPR: True positive rate at fixed false positive rate
    """
    # Predict probabilities
    y_pred = probe.predict_proba(eval_dataset)
    y_true = eval_dataset.labels_numpy()

    # Compute metrics
    auroc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    tpr = tpr_at_fixed_fpr_score(y_true, y_pred, fpr=fpr)

    return DatasetResults(
        layer=probe.layer,
        metrics={"auroc": auroc, "accuracy": accuracy, "tpr_at_fpr": tpr}
    )
```

#### 5. **Configuration System** (`config.py`)

Uses Hydra for hierarchical configs:

```yaml
# config/probe/attention.yaml
probe_type: attention
aggregation: attention_weighted
hyper_params:
  lr: 0.001
  num_epochs: 10
  batch_size: 32
  hidden_dim: 4096
```

**Usage:**
```bash
uv run mup exp +experiment=evaluate_probe probe=attention layer=16 model=llama-8b
```

This loads:
- Base config from `config/config.yaml`
- Experiment config from `config/experiments/evaluate_probe.yaml`
- Probe config from `config/probe/attention.yaml`
- Overrides: `layer=16`, `model=llama-8b`

---

## TuberLens Simplified Architecture

### Key Simplifications

1. **No activation caching:** Compute on-the-fly each time
2. **No R2/cloud storage:** Local-only
3. **Simplified config:** Direct Python API instead of Hydra
4. **Focused API:** Only probe training/evaluation, no dataset generation

### Example Usage (from their notebooks)

```python
from tuberlens import train_probe, evaluate_probe
from tuberlens.interfaces.dataset import LabelledDataset
from tuberlens.interfaces.probes import ProbeSpec

# Load dataset
train_data = LabelledDataset.load_from("train.jsonl")
test_data = LabelledDataset.load_from("test.jsonl")

# Define probe
probe_spec = ProbeSpec(
    probe_type="linear_then_mean",  # Linear layer + mean aggregation
    aggregation="mean",
    hyper_params={
        "lr": 0.001,
        "num_epochs": 10,
        "batch_size": 32
    }
)

# Train
probe = train_probe(
    train_dataset=train_data,
    validation_dataset=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    layer=16,
    probe_spec=probe_spec,
    verbose=True
)

# Evaluate
results = evaluate_probe(
    probe=probe,
    eval_dataset=test_data,
    fpr=0.01
)

print(f"AUROC: {results.metrics['auroc']:.3f}")
```

### What's Removed vs Original

| Feature | models-under-pressure | TuberLens |
|---------|----------------------|-----------|
| Activation caching | ✅ (R2 + local) | ❌ |
| Dataset generation | ✅ (synthetic pipeline) | ❌ |
| Baseline methods | ✅ (LLM prompting, finetuning) | ❌ |
| Hydra config | ✅ | ❌ |
| W&B logging | ✅ | ✅ (optional) |
| Probe types | 7+ types | 7+ types |
| Chat template support | ✅ | ✅ |
| Position-specific probing | ❌ | ✅ (new) |

---

## Key Differences: models-under-pressure vs TuberLens

### When to Use Each

**Use models-under-pressure if:**
- You want to exactly reproduce paper results
- You need the full experimental pipeline
- You're doing large-scale experiments with activation reuse
- You need baseline comparisons (prompted LLMs, finetuned models)

**Use TuberLens if:**
- You're extending/adapting the work (like your Indonesian project)
- You prefer simple Python API over CLI/Hydra
- You don't need cloud storage
- You want cleaner, more maintainable code

### Code Cleanliness Comparison

**models-under-pressure:**
- ❌ "Rough state" per authors
- ❌ Complex infrastructure requirements
- ❌ Tightly coupled to paper experiments
- ✅ Complete and comprehensive
- ✅ All paper experiments reproducible

**TuberLens:**
- ✅ Clean, modular design
- ✅ Well-documented with examples
- ✅ Easy to extend
- ❌ Missing dataset generation
- ❌ Fewer baselines

---

## What You Need for Your Project

### Phase 1: Baseline Reproduction

**Minimum requirements:**
1. **Dataset:** Download from models-under-pressure public links
   - Training: `prompts_4x/train.jsonl` (~4000 examples)
   - Eval: Anthropic HH, ToolACE (2-3 datasets sufficient)

2. **Model:** Llama-3.1-8B-Instruct
   - ~16GB VRAM in fp16
   - ~8GB VRAM in 8-bit quantization

3. **Code approach:** Use TuberLens for cleaner implementation
   - Load their datasets (JSONL format is compatible)
   - Train probe on layer 16 (paper's best layer)
   - Evaluate on their test sets

### Phase 2: Indonesian Translation

**What changes:**
1. **Dataset:** Translate English JSONL to Indonesian
   ```python
   # Pseudocode
   for example in english_dataset:
       indonesian = translate_with_llm(example["input"])
       verify_quality(indonesian)  # Manual sampling
       new_dataset.append(indonesian)
   ```

2. **Probe:** Same probe, different test set
   - Train on English
   - Evaluate on Indonesian
   - Compare AUROC drop

### Phase 3: SAE Failure Analysis

**New components needed:**

1. **Load SAEs:**
   ```python
   from sae_lens import SAE

   # Load Llama Scope SAE for layer 16
   sae = SAE.from_pretrained(
       "fnlp/Llama-Scope",
       hookpoint="blocks.16.hook_resid_post"
   )
   ```

2. **Decompose activations:**
   ```python
   # For each failure case
   raw_activation = model.get_activations(failed_example, layer=16)

   # Decompose into SAE features
   sae_features = sae.encode(raw_activation)
   # Shape: (batch, seq_len, n_features=32768)

   # Find active features
   active_features = sae_features.nonzero()  # Sparse!

   # Interpret features (using auto-interp)
   for feat_idx in active_features:
       description = sae.feature_descriptions[feat_idx]
       activation_strength = sae_features[feat_idx]
       print(f"Feature {feat_idx}: {description} (strength: {activation_strength})")
   ```

3. **Failure analysis:**
   ```python
   # Compare success vs failure cases
   success_features = get_active_features(successful_predictions)
   failure_features = get_active_features(failed_predictions)

   # Find discriminative features
   failure_specific = failure_features - success_features

   # Interpret why probe failed
   for feat in failure_specific:
       print(f"Probe fails when feature '{feat.description}' is active")
   ```

### Infrastructure Requirements

**Compute:**
- GPU: A100 (40GB) ideal, RTX 3090 (24GB) sufficient, Colab Pro acceptable
- Estimated cost: $50-150 total (cloud compute)

**Storage:**
- Datasets: ~500MB
- Model: ~16GB (Llama-8B)
- SAEs: ~2GB per layer (Llama Scope)
- Activations (if cached): ~1GB per dataset

**Software:**
- Python 3.10+
- PyTorch 2.0+
- Transformers
- SAElens
- TuberLens (or models-under-pressure)

### Critical Files to Understand

**From models-under-pressure:**
1. `interfaces/dataset.py` - Dataset format (lines 1-200)
2. `model.py` - Activation extraction (lines 1-150)
3. `probes/base.py` - Probe interface (lines 1-63)
4. `experiments/evaluate_probes.py` - Evaluation metrics (lines 35-150)

**From TuberLens:**
1. `training.py` - Simple probe training (lines 1-100)
2. `evaluation.py` - Evaluation functions
3. `notebooks/examples/high-stakes.ipynb` - End-to-end example

### Recommended Learning Path

1. **Day 1:** Read this document, skim key files listed above
2. **Day 2:** Run TuberLens example notebook on their data
3. **Day 3:** Download models-under-pressure datasets, train a probe
4. **Day 4:** Validate you can reproduce their baseline (~0.91 AUROC)
5. **Day 5+:** Start Indonesian translation and SAE integration

---

## Common Patterns You'll Use

### 1. Loading and Splitting Data
```python
from tuberlens.interfaces.dataset import LabelledDataset

# Load
full_dataset = LabelledDataset.load_from("train.jsonl")

# Split
train_size = int(0.8 * len(full_dataset))
train_data = full_dataset[:train_size]
val_data = full_dataset[train_size:]
```

### 2. Training Multiple Layers
```python
for layer in [8, 12, 16, 20, 24]:
    probe = train_probe(
        train_dataset=train_data,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        layer=layer,
        probe_spec=probe_spec
    )

    results = evaluate_probe(probe, test_data)
    print(f"Layer {layer}: AUROC = {results.metrics['auroc']:.3f}")
```

### 3. Cross-Lingual Comparison
```python
# Train on English
probe_en = train_probe(english_train, ...)

# Evaluate on both
en_results = evaluate_probe(probe_en, english_test)
id_results = evaluate_probe(probe_en, indonesian_test)

print(f"English AUROC: {en_results.metrics['auroc']:.3f}")
print(f"Indonesian AUROC: {id_results.metrics['auroc']:.3f}")
print(f"Generalization gap: {en_results.metrics['auroc'] - id_results.metrics['auroc']:.3f}")
```

---

## Next Steps

Now that you understand the architecture:

1. ✅ **Set up environment:** Install dependencies (PyTorch, Transformers, TuberLens)
2. ✅ **Download datasets:** Get training and eval data from models-under-pressure links
3. ✅ **Run baseline:** Reproduce ~0.91 AUROC on their test sets
4. ⏳ **Plan SAE integration:** Study SAElens documentation, test loading Llama Scope
5. ⏳ **Design translation pipeline:** Choose LLM for translation, create verification workflow

Ready to start implementation when you are!
