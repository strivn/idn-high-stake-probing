"""Activation extraction with forward hooks.

Matches the paper's methodology (v2b):
  - Hook on input_layernorm (residual stream entering the layer)
  - Chat template applied to messages before tokenizing
  - Leading BOS token stripped after tokenization (paper's v[:, 1:])
  - max_length=8192 (paper's 2**13)
  - Mean pooling over sequence length, respecting attention mask
  - Layer truncation: only forward through layers 0..layer_idx (paper's HookedModel)
  - Dynamic batching: sort by sequence length to minimize padding (paper's get_batches)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Example
from .model import get_model_layers, set_model_layers


def mean_pool(activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool activations over sequence length, respecting padding.

    Operates on whatever device the tensors are on (GPU or CPU).

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


def _format_and_measure_lengths(
    tokenizer: AutoTokenizer,
    examples: List[Example],
) -> Tuple[List[str], List[int]]:
    """Apply chat template and measure approximate token lengths.

    Returns:
        formatted_texts: list of formatted strings
        lengths: list of token counts (from fast tokenizer, no padding)
    """
    formatted_texts = [
        tokenizer.apply_chat_template(
            ex.messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in examples
    ]

    # tokenize without padding to get true lengths
    encoded = tokenizer(formatted_texts, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]

    return formatted_texts, lengths


def _make_length_sorted_batches(
    lengths: List[int],
    batch_size: int,
) -> List[List[int]]:
    """Sort examples by token length, group into batches of similar length.

    Returns list of batches, where each batch is a list of original indices.
    """
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    batches = []
    for start in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[start : start + batch_size])

    return batches


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

    Optimizations matching the paper (models-under-pressure):
      - Truncates model to layers 0..layer_idx to skip unnecessary computation
      - Sorts examples by length to minimize padding within each batch
      - Pools on GPU, moves only the small (hidden_dim,) result to CPU
      - Clears GPU cache after every batch

    Returns:
        pooled: (n_examples, hidden_dim) numpy array, in original example order
    """
    device = next(model.parameters()).device

    # pre-tokenize to get lengths for dynamic batching
    formatted_texts, lengths = _format_and_measure_lengths(tokenizer, examples)
    batches = _make_length_sorted_batches(lengths, batch_size)

    # allocate output array — fill in original order via index mapping
    hidden_dim      = model.config.hidden_size
    all_pooled      = np.empty((len(examples), hidden_dim), dtype=np.float32)
    all_layers      = get_model_layers(model)
    hook_target     = all_layers[layer_idx].input_layernorm

    # truncate model to avoid computing layers beyond our hook
    original_layers = all_layers
    set_model_layers(model, original_layers[: layer_idx + 1])

    try:
        iterator = tqdm(batches, desc=f"Layer {layer_idx}") if show_progress else batches

        for batch_indices in iterator:
            batch_texts = [formatted_texts[i] for i in batch_indices]
            captured    = []

            def hook_fn(module, input, output):
                captured.append(output.detach())  # stays on GPU

            handle = hook_target.register_forward_hook(hook_fn)

            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                # strip leading BOS token (paper does v[:, 1:])
                inputs["input_ids"]      = inputs["input_ids"][:, 1:]
                inputs["attention_mask"] = inputs["attention_mask"][:, 1:]

                with torch.no_grad():
                    model(**inputs)

                # pool on GPU — captured[0] is (batch, seq_len, hidden_dim) on device
                attention_mask = inputs["attention_mask"]
                pooled = mean_pool(captured[0], attention_mask)  # (batch, hidden_dim)

                # move only the small pooled result to CPU
                pooled_np = pooled.cpu().float().numpy()

                for j, orig_idx in enumerate(batch_indices):
                    all_pooled[orig_idx] = pooled_np[j]

            finally:
                handle.remove()
                # free intermediates
                del captured, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        # restore full model layers
        set_model_layers(model, original_layers)

    return all_pooled


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
