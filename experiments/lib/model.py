"""Model loading with VRAM-aware quantization.

Supports multiple model families:
  - Llama 3.1/3.3 (layers via model.model.layers)
  - Gemma 3 (layers via model.language_model.layers)

Quantization: 8-bit via bitsandbytes on CUDA, fp32 on CPU/MPS.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .env import get_device, get_gpu_vram_gb, should_quantize


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Registry: HF model name -> (short_name for cache, param_size for batch decisions)
# short_name is used in cache file prefixes to avoid collisions
MODEL_REGISTRY = {
    "meta-llama/Llama-3.1-8B-Instruct":  {"short_name": "llama31_8b",  "params_b": 8},
    "meta-llama/Llama-3.3-70B-Instruct": {"short_name": "llama33_70b", "params_b": 70},
    "google/gemma-3-12b-it":             {"short_name": "gemma3_12b",  "params_b": 12},
    "google/gemma-3-27b-it":             {"short_name": "gemma3_27b",  "params_b": 27},
}


def get_model_short_name(model_name: str) -> str:
    """Return a short, filesystem-safe name for cache prefixes."""
    info = MODEL_REGISTRY.get(model_name)
    if info:
        return info["short_name"]
    # fallback: sanitize the HF name
    return model_name.split("/")[-1].lower().replace("-", "_")


def get_config_attr(model, attr_name: str):
    """Get config attribute, handling both flat and nested (text_config) structures.

    Args:
        model: The model instance
        attr_name: Attribute name (e.g., 'num_hidden_layers', 'hidden_size', 'vocab_size')

    Returns:
        The attribute value
    """
    config = model.config
    if hasattr(config, 'text_config'):
        # Multimodal models (Gemma3) nest text params in text_config
        return getattr(config.text_config, attr_name)
    else:
        # Standard models (Llama) have flat config
        return getattr(config, attr_name)


def get_model_layers(model):
    """Return the nn.ModuleList of transformer layers for any supported model.

    Llama:  model.model.layers
    Gemma3: model.language_model.layers
    """
    model_type = model.config.model_type
    if model_type in ("llama",):
        return model.model.layers
    if model_type in ("gemma3",):
        return model.language_model.layers
    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        f"Add layer path to get_model_layers() in lib/model.py"
    )


def set_model_layers(model, layers):
    """Replace the transformer layers list (for layer truncation during extraction)."""
    model_type = model.config.model_type
    if model_type in ("llama",):
        model.model.layers = layers
    elif model_type in ("gemma3",):
        model.language_model.layers = layers
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = None,
    quantize: bool = None,
):
    """Load model and tokenizer with VRAM-aware configuration.

    Args:
        model_name: HuggingFace model identifier
        device:     Override device (auto-detected if None)
        quantize:   Force 8-bit quantization (auto-decided from VRAM if None)

    Returns:
        (model, tokenizer) tuple
    """
    if device is None:
        device = get_device()

    vram_gb = get_gpu_vram_gb()

    if quantize is None:
        quantize = should_quantize()

    if quantize and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map":          "auto",
            "torch_dtype":         torch.bfloat16,  # Gemma 3 overflows in fp16 default (transformers#39972)
        }
        quant_label = "8-bit"
    elif device == "cuda":
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map":  "auto",
        }
        quant_label = "bf16"
    else:
        model_kwargs = {
            "torch_dtype": torch.float32,
            "device_map":  device,
        }
        quant_label = "fp32"

    print(f"Loading {model_name}...")
    print(f"  Device: {device} | VRAM: {vram_gb:.1f} GB | Precision: {quant_label}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use helper to handle both flat (Llama) and nested (Gemma3) config structures
    n_layers   = get_config_attr(model, 'num_hidden_layers')
    hidden_dim = get_config_attr(model, 'hidden_size')
    vocab_size = get_config_attr(model, 'vocab_size')

    print(f"  Layers: {n_layers} | Hidden dim: {hidden_dim} | Vocab: {vocab_size}")

    return model, tokenizer
