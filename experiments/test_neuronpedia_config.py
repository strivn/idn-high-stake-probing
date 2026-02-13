"""Test Neuronpedia configuration mappings.

Run this to verify model/SAE ID mappings are correct before using in notebooks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from lib.neuronpedia_config import (
    get_neuronpedia_ids,
    auto_detect_neuronpedia_ids,
    get_model_neuronpedia_config,
    OUR_MODELS,
)

print("="*80)
print("NEURONPEDIA CONFIGURATION TEST")
print("="*80)

# Test all our models
for model_name, model_config in OUR_MODELS.items():
    print(f"\n{'─'*80}")
    print(f"Model: {model_name}")
    print(f"{'─'*80}")

    layer = model_config["default_layer"]
    release = model_config["sae_release"]

    # Get Neuronpedia IDs
    try:
        model_id, sae_id = get_neuronpedia_ids(release, layer)
        print(f"✅ Neuronpedia Model ID: {model_id}")
        print(f"✅ Neuronpedia SAE ID:   {sae_id}")
    except Exception as e:
        print(f"❌ Error getting Neuronpedia IDs: {e}")
        continue

    # Test auto-detection
    sae_lens_id = model_config["sae_id_template"].format(layer=layer)
    detected_model, detected_sae = auto_detect_neuronpedia_ids(sae_lens_id)

    if detected_model == model_id and detected_sae == sae_id:
        print(f"✅ Auto-detection works: {sae_lens_id}")
    else:
        print(f"⚠️ Auto-detection mismatch:")
        print(f"   SAE Lens ID: {sae_lens_id}")
        print(f"   Detected:    {detected_model}/{detected_sae}")
        print(f"   Expected:    {model_id}/{sae_id}")

    # Test convenience wrapper
    try:
        config = get_model_neuronpedia_config(model_name, layer)
        print(f"✅ Full config: {config}")
    except Exception as e:
        print(f"❌ Error getting full config: {e}")

    # Generate Neuronpedia URL
    url = f"https://www.neuronpedia.org/{model_id}/{sae_id}"
    print(f"📍 URL: {url}")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}\n")

# Additional tests
print("\n" + "="*80)
print("ADDITIONAL LAYER TESTS")
print("="*80)

# Test Llama 3.1 8B different layers
print("\nLlama 3.1 8B - Layer sweep:")
for layer in [8, 12, 16, 20, 26, 31]:
    try:
        model_id, sae_id = get_neuronpedia_ids("llama_scope_lxr_8x", layer)
        print(f"  Layer {layer:2d}: {model_id}/{sae_id}")
    except Exception as e:
        print(f"  Layer {layer:2d}: ❌ {e}")

# Test Gemma 3 12B different layers
print("\nGemma 3 12B IT - Available layers:")
for layer in [12, 24, 31, 41]:
    try:
        model_id, sae_id = get_neuronpedia_ids("gemma_scope_2_12b_it_res", layer)
        print(f"  Layer {layer:2d}: {model_id}/{sae_id}")
    except Exception as e:
        print(f"  Layer {layer:2d}: ❌ {e}")

print("\n✅ All tests complete! Check URLs manually at Neuronpedia.org")
