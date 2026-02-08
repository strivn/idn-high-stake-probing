"""Model loading with VRAM-aware quantization.

Loads Llama-3.1-8B-Instruct with automatic quantization decisions:
  - VRAM < 20 GB (T4): 8-bit quantization via bitsandbytes
  - VRAM >= 20 GB (A10, A100): fp16, no quantization (cleaner activations)
  - No CUDA (MPS/CPU): fp32
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .env import get_device, get_gpu_vram_gb, should_quantize


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


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
        quantize = should_quantize(vram_gb)

    if quantize and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map":          "auto",
        }
        quant_label = "8-bit"
    elif device == "cuda":
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map":  "auto",
        }
        quant_label = "fp16"
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

    n_layers   = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print(f"  Layers: {n_layers} | Hidden dim: {hidden_dim} | Vocab: {model.config.vocab_size}")

    return model, tokenizer
