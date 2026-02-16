"""Activation extraction with forward hooks.

Matches the paper's methodology (v2b):
  - Hook on input_layernorm (residual stream entering the layer)
  - Chat template applied to messages before tokenizing
  - Leading BOS token stripped after tokenization (paper's v[:, 1:])
  - max_length=8192 (paper's 2**13)
  - Mean pooling over sequence length, respecting attention mask
  - Layer truncation: only forward through layers 0..layer_idx (paper's HookedModel)
  - Dynamic batching: sort by sequence length to minimize padding (paper's get_batches)

Also supports per-token SAE encoding via extract_sae_features_batched():
  - Encodes each token's activation through an SAE inside the forward hook
  - Max-pools SAE feature activations across tokens per example
  - Avoids storing massive per-token activations (only stores aggregated features)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Example
from .model import get_config_attr, get_model_layers, set_model_layers


# ---------------------------------------------------------------------------
# Gemma-specific message normalization
# ---------------------------------------------------------------------------

TOOL_RESPONSE_PREFIX = "Sure, here's the result of the tool call: "


def _needs_message_normalization(tokenizer: AutoTokenizer) -> bool:
    """Check if tokenizer requires strict user/assistant alternation.

    Gemma 3 rejects tool roles and consecutive same-role messages.
    Llama 3.x handles these natively, so we skip normalization for it.
    """
    name = getattr(tokenizer, "name_or_path", "").lower()
    return "gemma" in name


def _normalize_messages(messages: List[dict]) -> List[dict]:
    """Normalize chat messages for models with strict role requirements.

    Applied only for Gemma-family models. Two transformations:
    1. Convert 'tool' role -> 'user' with a prefix (matching the original paper's
       approach in toolace_dataset.py: tool responses are external input).
    2. Merge consecutive same-role messages (join content with double newline).
       Handles the 5 Anthropic examples with consecutive 'assistant' messages.
    """
    # Step 1: convert tool -> user
    converted = []
    for msg in messages:
        if msg["role"] == "tool":
            converted.append({
                "role":    "user",
                "content": TOOL_RESPONSE_PREFIX + msg.get("content", ""),
            })
        else:
            converted.append(msg)

    # Step 2: merge consecutive same-role messages
    if not converted:
        return converted

    merged = [converted[0].copy()]
    for msg in converted[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + msg.get("content", "")
        else:
            merged.append(msg.copy())

    return merged


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

    For Gemma models, normalizes messages first (tool->user, merge consecutive
    same-role). Llama and other models pass messages through unchanged.

    Returns:
        formatted_texts: list of formatted strings
        lengths: list of token counts (from fast tokenizer, no padding)
    """
    normalize = _needs_message_normalization(tokenizer)

    formatted_texts = []
    for ex in examples:
        msgs = _normalize_messages(ex.messages) if normalize else ex.messages
        formatted_texts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

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
    hidden_dim      = get_config_attr(model, 'hidden_size')
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


# ---------------------------------------------------------------------------
# Per-token SAE feature extraction
# ---------------------------------------------------------------------------

def _sae_encode_tokens(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    sae,
    token_batch_size: int = 512,
) -> torch.Tensor:
    """Encode per-token activations through SAE and max-pool over sequence.

    Processes tokens in sub-batches to avoid OOM on long sequences.

    Args:
        activations:      (batch, seq_len, hidden_dim) on GPU
        attention_mask:    (batch, seq_len)
        sae:              SAE object with .encode() method
        token_batch_size: max tokens to encode at once through SAE

    Returns:
        features: (batch, n_sae_features) max-pooled SAE features
    """
    batch_size, seq_len, hidden_dim = activations.shape
    n_features = sae.cfg.d_sae

    # max-pool accumulator, init to zero (inactive features stay 0)
    max_features = torch.zeros(batch_size, n_features,
                               device=activations.device, dtype=torch.float32)

    # flatten to (batch * seq_len, hidden_dim) for SAE encoding
    flat_acts = activations.reshape(-1, hidden_dim)
    flat_mask = attention_mask.reshape(-1).bool()

    # only encode non-padding tokens
    valid_indices = torch.where(flat_mask)[0]
    n_valid       = valid_indices.shape[0]

    for start in range(0, n_valid, token_batch_size):
        end     = min(start + token_batch_size, n_valid)
        idx     = valid_indices[start:end]
        tok_act = flat_acts[idx]                        # (chunk, hidden_dim)

        with torch.no_grad():
            tok_feat = sae.encode(tok_act)              # (chunk, n_features)

        # map back to (batch, feature) via max
        batch_idx = idx // seq_len                      # which example each token belongs to
        # scatter_reduce with max: update max_features[batch_idx[i]] = max(current, tok_feat[i])
        max_features.scatter_reduce_(
            0,
            batch_idx.unsqueeze(1).expand_as(tok_feat),
            tok_feat,
            reduce="amax",
            include_self=True,
        )

    return max_features


def extract_sae_features_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Example],
    layer_idx: int,
    sae,
    batch_size: int = 2,
    max_length: int = 2**13,
    show_progress: bool = True,
) -> np.ndarray:
    """Extract per-token SAE features, max-pooled over sequence.

    Same forward pass as extract_activations_batched, but instead of mean-pooling
    raw activations, encodes each token through the SAE and max-pools the sparse
    features. This gives proper SAE sparsity (L0 ~ 50) instead of the ~2 you get
    from encoding mean-pooled activations.

    Args:
        model:      Loaded LLM
        tokenizer:  Corresponding tokenizer
        examples:   List of Example objects
        layer_idx:  Which layer to hook
        sae:        SAE object (from sae_lens.SAE.from_pretrained)
        batch_size: Examples per forward pass (use 1-2, SAE encoding is memory-heavy)
        max_length: Max sequence length
        show_progress: Show tqdm bar

    Returns:
        features: (n_examples, n_sae_features) numpy array, max-pooled per example
    """
    device = next(model.parameters()).device

    formatted_texts, lengths = _format_and_measure_lengths(tokenizer, examples)
    batches = _make_length_sorted_batches(lengths, batch_size)

    n_features      = sae.cfg.d_sae
    all_features    = np.empty((len(examples), n_features), dtype=np.float32)
    all_layers      = get_model_layers(model)
    hook_target     = all_layers[layer_idx].input_layernorm

    original_layers = all_layers
    set_model_layers(model, original_layers[: layer_idx + 1])

    try:
        desc     = f"SAE L{layer_idx}"
        iterator = tqdm(batches, desc=desc) if show_progress else batches

        for batch_indices in iterator:
            batch_texts = [formatted_texts[i] for i in batch_indices]
            captured    = []

            def hook_fn(module, input, output):
                captured.append(output.detach())

            handle = hook_target.register_forward_hook(hook_fn)

            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                inputs["input_ids"]      = inputs["input_ids"][:, 1:]
                inputs["attention_mask"] = inputs["attention_mask"][:, 1:]

                with torch.no_grad():
                    model(**inputs)

                # captured[0]: (batch, seq_len, hidden_dim) on GPU
                # encode through SAE per-token, max-pool over sequence
                features = _sae_encode_tokens(
                    captured[0], inputs["attention_mask"], sae
                )
                features_np = features.cpu().float().numpy()

                for j, orig_idx in enumerate(batch_indices):
                    all_features[orig_idx] = features_np[j]

            finally:
                handle.remove()
                del captured, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        set_model_layers(model, original_layers)

    return all_features


def get_sae_features_cached(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Example],
    layer_idx: int,
    sae,
    cache_name: str,
    cache_dir: Path,
    cache_prefix: str = "v2b",
    batch_size: int = 2,
    force_recompute: bool = False,
) -> np.ndarray:
    """Extract per-token SAE features with disk caching.

    Cache files: {cache_prefix}_sae_{cache_name}_layer{layer_idx}.npy
    """
    cache_filename = f"{cache_prefix}_sae_{cache_name}_layer{layer_idx}.npy"
    cache_path     = cache_dir / cache_filename

    if cache_path.exists() and not force_recompute:
        print(f"Loading SAE features from cache: {cache_path.name}")
        return np.load(cache_path)

    print(f"Extracting SAE features for {len(examples)} examples (batch_size={batch_size})...")
    features = extract_sae_features_batched(
        model, tokenizer, examples, layer_idx, sae, batch_size
    )

    np.save(cache_path, features)
    print(f"Saved SAE features to cache: {cache_path.name}")

    return features
