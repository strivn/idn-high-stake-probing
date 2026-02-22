# Why bfloat16? Floating Point Formats for Deep Learning

## The Problem We Hit

Running Gemma 3 12B with 8-bit quantization via bitsandbytes, our activation extraction produced NaN values. The culprit: bitsandbytes defaults to **float16** for non-quantized operations, and Gemma 3's RMSNorm layer overflows float16's limited range. This note explains what that actually means and why bfloat16 fixes it.

---

## How Floating Point Numbers Work

A floating point number is stored as three parts: **sign**, **exponent**, and **mantissa** (also called significand or fraction). Think of it like scientific notation:

```
-1.234 × 10^5  =  -123,400

sign = negative
mantissa = 1.234  (the precision — how many significant digits)
exponent = 5      (the range — how big or small the number can get)
```

In binary, the same structure applies. The key tradeoff in any fixed-width format is: **how many bits go to the exponent vs. the mantissa?** More exponent bits = wider range of representable numbers. More mantissa bits = finer precision between consecutive numbers.

---

## The Three Formats, Side by Side

```
Format     Total bits   Sign   Exponent   Mantissa   Max value           Precision
─────────  ──────────   ────   ────────   ────────   ──────────────────  ─────────────
float32       32          1       8          23       ~3.4 × 10^38       ~7 decimal digits
float16       16          1       5          10       ~6.5 × 10^4        ~3.3 decimal digits
bfloat16      16          1       8           7       ~3.4 × 10^38       ~2.4 decimal digits
```

The insight: **bfloat16 is literally float32 with the mantissa truncated from 23 to 7 bits.** The exponent stays identical. That's the whole design.

```
float32:    [1 sign] [8 exponent] [23 mantissa]
bfloat16:   [1 sign] [8 exponent] [ 7 mantissa]    ← just chop off 16 mantissa bits
float16:    [1 sign] [5 exponent] [10 mantissa]     ← different exponent size!
```

Converting float32 to bfloat16 is a single truncation — no rescaling needed. This is why Google designed it this way: fast, cheap hardware conversion.

---

## Range vs. Precision: Why It Matters

### The range problem (exponent)

float16 maxes out at **65,504**. Any intermediate value larger than that becomes `inf`. In a transformer, residual connections repeatedly add activations:

```
hidden_states = residual + hidden_states  # this happens twice per layer
```

After dozens of layers, activation magnitudes can easily exceed 65,504. When that happens in float16:

```python
import torch
torch.tensor(100000, dtype=torch.float16)   # tensor(inf, dtype=torch.float16)
torch.tensor(100000, dtype=torch.bfloat16)  # tensor(100000., dtype=torch.bfloat16)  ← fine
```

bfloat16 has the same range as float32 (~3.4 × 10^38), so this overflow effectively never happens in practice.

### The precision tradeoff (mantissa)

bfloat16 has **less** precision than float16 (7 vs 10 mantissa bits). Consecutive representable numbers are farther apart. For example, near 65,000: three consecutive bfloat16 values are 65,280, 65,536, and 66,048 — gaps of 256 and 512.

But Google's research found that neural networks are **far more sensitive to exponent range than mantissa precision**. Gradients and activations span many orders of magnitude; the exact value at the 4th decimal place matters less than being able to represent it at all without overflowing.

---

## Why Google Designed bfloat16

bfloat16 (Brain Floating Point) was created by Google Brain in 2017 for their Tensor Processing Units (TPUs). The motivation:

1. **Mixed-precision math**: TPUs use bfloat16 for multiply operations but accumulate results in float32. This gives you speed (16-bit multiplies) without sacrificing accumulation accuracy.

2. **Drop-in replacement for float32**: Because bfloat16 shares float32's exponent, it has identical overflow/underflow behavior. You can swap float32 for bfloat16 in training without loss scaling or other fp16 workarounds.

3. **Cheap conversion**: Truncating 16 mantissa bits is trivial in hardware — no normalization or rescaling needed.

The format has since been adopted by nearly every ML hardware vendor: NVIDIA (Ampere+), AMD (Zen), Intel (Xeon), Apple (M2+, A15+), and ARM (v8.6+).

---

## float16 Requires Loss Scaling; bfloat16 Doesn't

When training in float16, gradients for early layers can be tiny (e.g., 10^-8), which **underflows** float16's minimum representable value (~6 × 10^-8). The standard workaround is **loss scaling**: multiply the loss by a large constant before backprop (scaling gradients up into representable range), then divide the optimizer update by the same constant.

bfloat16 doesn't need this because its minimum representable value is ~1.2 × 10^-38 (same as float32). Gradients that would underflow in float16 are representable just fine in bfloat16.

This is why bfloat16 mixed-precision training is simpler to implement and more robust than float16 mixed-precision.

---

## The Gemma 3 Case: What Went Wrong

Gemma 3 was trained in bfloat16 and its architecture assumes bfloat16's dynamic range. Specifically, the `Gemma3RMSNorm` layer produces intermediate values that exceed float16's 65,504 max. When loaded in float16 (or when bitsandbytes defaults to float16 for non-quantized operations):

1. RMSNorm output overflows to `inf`
2. Residual addition propagates `inf` through subsequent layers
3. `inf - inf` or `0 * inf` produces `NaN`
4. Mean pooling of NaN activations produces NaN
5. StandardScaler chokes on NaN input → `ValueError: Input X contains NaN`

The fix: explicitly pass `torch_dtype=torch.bfloat16` when loading. This ensures all non-quantized compute (norms, attention, residuals) stays in bfloat16's safe range.

```python
# Before (broken for Gemma 3)
model_kwargs = {
    "quantization_config": quantization_config,
    "device_map":          "auto",
    # torch_dtype not set → bitsandbytes defaults to fp16
}

# After (works for all models)
model_kwargs = {
    "quantization_config": quantization_config,
    "device_map":          "auto",
    "torch_dtype":         torch.bfloat16,
}
```

This is safe for Llama too — bfloat16 is strictly better than float16 for inference on modern hardware (Ampere GPUs and newer all support bfloat16 natively).

---

## When float16 Is Still Used

float16 isn't obsolete. It's still relevant when:

- **Older GPU hardware** (pre-Ampere NVIDIA, e.g., V100, T4) has optimized float16 tensor cores but no native bfloat16 support
- **Inference-only** on models trained in float16, where activation ranges are known to stay within bounds
- **GGUF/GPTQ quantization** formats that use float16 as the dequantized compute type

But for any model trained in bfloat16 (which includes most modern LLMs: Llama 3.x, Gemma, Mistral, Qwen), always load in bfloat16.

---

## Practical Takeaway

| Scenario | Use |
|---|---|
| Training any new model | bfloat16 mixed-precision (if hardware supports it) |
| Inference with bitsandbytes quantization | Always set `torch_dtype=torch.bfloat16` explicitly |
| Inference on V100/T4 (no bfloat16 hardware) | float16, but watch for overflow on large models |
| You need maximum numeric precision | float32 (but 2x memory) |

The general rule: **if your hardware supports bfloat16, prefer it over float16.** Same memory footprint, same speed, but no overflow risk and no loss scaling needed.

---

## References

- [Google Cloud Blog: BFloat16 — The Secret to High Performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Google's original explanation of bfloat16 design and TPU implementation
- [Nick Higham: What is bfloat16 Arithmetic?](https://nhigham.com/2020/06/02/what-is-bfloat16-arithmetic/) — Rigorous numerical analysis of bfloat16 precision, unit roundoff, and representable number spacing
- [Sebastian Raschka: Accelerating LLMs with Mixed-Precision Techniques](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html) — Practical comparison of float16 vs bfloat16 for LLM training, including concrete overflow examples and DistilBERT benchmarks
- [Wikipedia: bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) — Technical specification, hardware adoption timeline, and Google Brain history
- [Cerebras: To Bfloat or Not to Bfloat?](https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question) — Analysis of when bfloat16 is and isn't appropriate
- [arXiv:1905.12322 — A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322) — Intel/Facebook study validating bfloat16 across CNN and RNN workloads
- [transformers#39972 — Gemma3 fp16 NaN/inf bug](https://github.com/huggingface/transformers/issues/39972) — The specific Gemma 3 overflow issue with RMSNorm in float16
- [bitsandbytes#1572 — Gemma 3 empty output with int8](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1572) — bitsandbytes defaulting to float16 for Gemma 3 quantized models
