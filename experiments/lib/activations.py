"""Activation extraction with forward hooks.

Matches the paper's methodology (v2b):
  - Hook on input_layernorm (residual stream entering the layer)
  - Chat template applied to messages before tokenizing
  - Leading BOS token stripped after tokenization (paper's v[:, 1:])
  - max_length=8192 (paper's 2**13)
  - Mean pooling over sequence length, respecting attention mask
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Example


def mean_pool(activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool activations over sequence length, respecting padding.

    Args:
        activations:    (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)

    Returns:
        pooled: (batch, hidden_dim)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()         # (batch, seq_len, 1)
    masked  = activations * mask_expanded                        # zero out padding
    summed  = masked.sum(dim=1)                                  # (batch, hidden_dim)
    counts  = attention_mask.sum(dim=1, keepdim=True).float()    # (batch, 1)
    counts  = counts.clamp(min=1e-9)
    return summed / counts                                       # (batch, hidden_dim)


def extract_activations_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Example],
    layer_idx: int,
    batch_size: int = 4,
    max_length: int = 2**13,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract mean-pooled activations from input_layernorm at the given layer.

    Hooks into model.model.layers[layer_idx].input_layernorm — this captures
    the residual stream entering the layer (what layers 0..N-1 have built up).

    Returns:
        pooled: (n_examples, hidden_dim) numpy array
    """
    device = next(model.parameters()).device
    all_pooled = []

    hook_target = model.model.layers[layer_idx].input_layernorm

    batches = range(0, len(examples), batch_size)
    if show_progress:
        batches = tqdm(batches, desc=f"Extracting layer {layer_idx} (input_layernorm)")

    for i in batches:
        batch_examples = examples[i : i + batch_size]
        captured = []

        def hook_fn(module, input, output):
            captured.append(output.detach().cpu())

        handle = hook_target.register_forward_hook(hook_fn)

        try:
            # apply chat template to each example's messages
            formatted_texts = [
                tokenizer.apply_chat_template(
                    ex.messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for ex in batch_examples
            ]

            inputs = tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # strip leading BOS token (paper does v[:, 1:])
            inputs["input_ids"]      = inputs["input_ids"][:, 1:]
            inputs["attention_mask"] = inputs["attention_mask"][:, 1:]

            with torch.no_grad():
                _ = model(**inputs)

            # captured[0]: (batch, seq_len, hidden_dim) from layernorm output
            activations    = captured[0]
            attention_mask = inputs["attention_mask"].cpu()

            pooled = mean_pool(activations, attention_mask)
            all_pooled.append(pooled.numpy())

        finally:
            handle.remove()

        # periodic cache clear to avoid VRAM buildup
        if torch.cuda.is_available() and i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    return np.concatenate(all_pooled, axis=0)


def get_activations_cached(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Example],
    layer_idx: int,
    cache_name: str,
    cache_dir: Path,
    cache_prefix: str = "v2b",
    batch_size: int = 4,
    force_recompute: bool = False,
) -> np.ndarray:
    """Extract activations with disk caching.

    Cache files are named: {cache_prefix}_{cache_name}_layer{layer_idx}.npy
    """
    cache_filename = f"{cache_prefix}_{cache_name}_layer{layer_idx}.npy"
    cache_path     = cache_dir / cache_filename

    if cache_path.exists() and not force_recompute:
        print(f"Loading from cache: {cache_path.name}")
        return np.load(cache_path)

    print(f"Computing activations for {len(examples)} examples (batch_size={batch_size})...")
    activations = extract_activations_batched(
        model, tokenizer, examples, layer_idx, batch_size
    )

    np.save(cache_path, activations)
    print(f"Saved to cache: {cache_path.name}")

    return activations
