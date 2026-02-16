"""Neuronpedia configuration for SAE feature lookups.

Maps SAE Lens model/SAE IDs to Neuronpedia model/SAE IDs.
Verified against Neuronpedia.org as of Feb 10, 2026.
"""

from typing import Dict, Tuple, Optional


# Mapping: SAE Lens release -> (Neuronpedia model_id, SAE ID template function)
NEURONPEDIA_MAPPINGS = {
    # ========================================================================
    # Llama 3.1 8B - Llama Scope
    # ========================================================================
    # SAE Lens: fnlp/Llama-Scope (https://huggingface.co/fnlp/Llama-Scope)
    # Neuronpedia: llama3.1-8b (https://www.neuronpedia.org/llama3.1-8b)
    #
    # Note: Llama Scope uses "8x" (32K features) and "32x" (128K features)
    # Neuronpedia uses "32k" and "131k" in SAE IDs
    # ========================================================================
    "llama_scope_lxr_8x": {
        "neuronpedia_model": "llama3.1-8b",
        "sae_id_template": lambda layer: f"{layer}-llamascope-res-32k",
        "description": "Llama 3.1 8B - Residual stream, 32K features (8x expansion)",
        "layers": list(range(32)),  # Layers 0-31
    },
    "llama_scope_lxr_32x": {
        "neuronpedia_model": "llama3.1-8b",
        "sae_id_template": lambda layer: f"{layer}-llamascope-res-131k",
        "description": "Llama 3.1 8B - Residual stream, 128K features (32x expansion)",
        "layers": list(range(32)),
    },

    # ========================================================================
    # Llama 3.3 70B - Goodfire
    # ========================================================================
    # SAE Lens: Not in pretrained_saes.yaml yet (use manual HF path)
    # Neuronpedia: llama3.3-70b-it-gf
    # Only layer 50 available as of Feb 2026
    # ========================================================================
    "llama3.3_70b_goodfire": {
        "neuronpedia_model": "llama3.3-70b-it-gf",
        "sae_id_template": lambda layer: f"{layer}-resid-post-gf",
        "description": "Llama 3.3 70B Instruct - Goodfire SAE (layer 50 only)",
        "layers": [50],  # Only layer 50 available
    },

    # ========================================================================
    # Gemma 3 - Gemma Scope 2
    # ========================================================================
    # HuggingFace: google/gemma-scope-2-{size}-it
    #   resid_post/     -> select layers, widths 16k/65k/262k/1m, l0 small/medium/big
    #   resid_post_all/ -> all layers 0-20 (12B), widths 16k/262k, l0 small/big
    # SAE Lens release: "gemma-scope-2-{size}-it-res", sae_id: "layer_{L}_width_{W}_l0_{sparsity}"
    # Neuronpedia: model "gemma-3-{size}-it", SAE ID "{L}-gemmascope-2-res-{W}"
    # ========================================================================
    "gemma_scope_2_12b_it_res": {
        "neuronpedia_model": "gemma-3-12b-it",
        "sae_id_template": lambda layer, width="16k": f"{layer}-gemmascope-2-res-{width}",
        "description": "Gemma 3 12B IT - Residual stream (resid_post select layers)",
        "layers": [12, 24, 31, 41],
        "widths": ["16k", "65k", "262k", "1m"],
        "l0_levels": ["small", "medium", "big"],
    },
    "gemma_scope_2_27b_it_res": {
        "neuronpedia_model": "gemma-3-27b-it",
        "sae_id_template": lambda layer, width="16k": f"{layer}-gemmascope-2-res-{width}",
        "description": "Gemma 3 27B IT - Residual stream (resid_post select layers)",
        "layers": [16, 31, 40, 53],
        "widths": ["16k", "262k"],
        "l0_levels": ["small", "medium", "big"],
    },
}


def get_neuronpedia_ids(
    sae_lens_release: str,
    layer: int,
    **kwargs
) -> Tuple[str, str]:
    """
    Get Neuronpedia model_id and sae_id from SAE Lens configuration.

    Args:
        sae_lens_release: SAE Lens release name (e.g., "llama_scope_lxr_8x")
        layer: Layer number
        **kwargs: Additional parameters (e.g., width, l0 for Gemma Scope)

    Returns:
        (neuronpedia_model_id, neuronpedia_sae_id)

    Raises:
        ValueError: If release not found or layer not supported

    Example:
        >>> model_id, sae_id = get_neuronpedia_ids("llama_scope_lxr_8x", 12)
        >>> print(model_id, sae_id)
        llama3.1-8b 12-llamascope-res-32k
    """
    if sae_lens_release not in NEURONPEDIA_MAPPINGS:
        raise ValueError(
            f"Unknown SAE Lens release: {sae_lens_release}. "
            f"Available: {list(NEURONPEDIA_MAPPINGS.keys())}"
        )

    config = NEURONPEDIA_MAPPINGS[sae_lens_release]

    # Check layer support
    if layer not in config["layers"]:
        raise ValueError(
            f"Layer {layer} not supported for {sae_lens_release}. "
            f"Available layers: {config['layers']}"
        )

    model_id = config["neuronpedia_model"]
    sae_id = config["sae_id_template"](layer, **kwargs)

    return model_id, sae_id


def detect_sae_release(sae_id: str) -> Optional[str]:
    """
    Auto-detect SAE Lens release from SAE ID string.

    Args:
        sae_id: SAE Lens SAE ID (e.g., "fnlp/Llama3_1-8B-Base-L12R-8x")

    Returns:
        SAE Lens release name or None if not recognized

    Example:
        >>> release = detect_sae_release("fnlp/Llama3_1-8B-Base-L12R-8x")
        >>> print(release)
        llama_scope_lxr_8x
    """
    sae_id = sae_id.lower()

    # Llama Scope patterns
    if "llama" in sae_id and "llamascope" not in sae_id:
        if "8x" in sae_id or "32k" in sae_id:
            if "r-" in sae_id or "resid" in sae_id:
                return "llama_scope_lxr_8x"
        elif "32x" in sae_id or "128k" in sae_id or "131k" in sae_id:
            if "r-" in sae_id or "resid" in sae_id:
                return "llama_scope_lxr_32x"

    # Llama 3.3 70B Goodfire
    if "llama-3.3" in sae_id or "llama3.3" in sae_id:
        if "goodfire" in sae_id or "gf" in sae_id:
            return "llama3.3_70b_goodfire"

    # Gemma Scope 2
    if "gemma" in sae_id and ("scope-2" in sae_id or "gemma-3" in sae_id):
        if "12b" in sae_id and "it" in sae_id:
            return "gemma_scope_2_12b_it_res"
        elif "27b" in sae_id and "it" in sae_id:
            return "gemma_scope_2_27b_it_res"

    return None


def extract_layer_from_sae_id(sae_id: str) -> Optional[int]:
    """
    Extract layer number from SAE ID string.

    Args:
        sae_id: SAE Lens SAE ID

    Returns:
        Layer number or None if not found

    Example:
        >>> layer = extract_layer_from_sae_id("fnlp/Llama3_1-8B-Base-L12R-8x")
        >>> print(layer)
        12
    """
    import re

    # Llama Scope format: L{layer}R-8x or L{layer}R-32x
    match = re.search(r'L(\d+)R-', sae_id)
    if match:
        return int(match.group(1))

    # Gemma Scope format: layer_{layer}_width_...
    match = re.search(r'layer_(\d+)_', sae_id)
    if match:
        return int(match.group(1))

    # Generic layer pattern
    match = re.search(r'[lL](\d+)', sae_id)
    if match:
        return int(match.group(1))

    return None


def auto_detect_neuronpedia_ids(sae_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Automatically detect Neuronpedia IDs from SAE Lens SAE ID.

    Args:
        sae_id: SAE Lens SAE ID

    Returns:
        (neuronpedia_model_id, neuronpedia_sae_id) or (None, None) if not detected

    Example:
        >>> model_id, sae_id = auto_detect_neuronpedia_ids("fnlp/Llama3_1-8B-Base-L12R-8x")
        >>> print(model_id, sae_id)
        llama3.1-8b 12-llamascope-res-32k
    """
    release = detect_sae_release(sae_id)
    if release is None:
        return None, None

    layer = extract_layer_from_sae_id(sae_id)
    if layer is None:
        return None, None

    try:
        return get_neuronpedia_ids(release, layer)
    except ValueError:
        return None, None


# ============================================================================
# Registry for our specific models
# ============================================================================
OUR_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "sae_release": "llama_scope_lxr_8x",
        "default_layer": 12,  # Best from layer sweep
        "sae_id_template": "fnlp/Llama3_1-8B-Base-L{layer}R-8x",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "sae_release": "llama3.3_70b_goodfire",
        "default_layer": 50,  # Only available layer
        "sae_id_template": "Goodfire/Llama-3.3-70B-Instruct-SAE-l{layer}",
    },
    "google/gemma-3-12b-it": {
        "sae_release": "gemma_scope_2_12b_it_res",
        "sae_lens_release": "gemma-scope-2-12b-it-res",
        "default_layer": 24,  # Middle layer
        "sae_id_template": "layer_{layer}_width_16k_l0_medium",
    },
    "google/gemma-3-27b-it": {
        "sae_release": "gemma_scope_2_27b_it_res",
        "sae_lens_release": "gemma-scope-2-27b-it-res",
        "default_layer": 31,  # Middle layer
        "sae_id_template": "layer_{layer}_width_16k_l0_medium",
    },
}


def get_model_neuronpedia_config(
    model_name: str,
    layer: Optional[int] = None
) -> Dict[str, str]:
    """
    Get complete Neuronpedia configuration for a model.

    Args:
        model_name: HuggingFace model name
        layer: Layer number (uses default if None)

    Returns:
        Dict with model_id, sae_id, sae_lens_id

    Example:
        >>> config = get_model_neuronpedia_config("meta-llama/Llama-3.1-8B-Instruct", 12)
        >>> print(config)
        {'model_id': 'llama3.1-8b', 'sae_id': '12-llamascope-res-32k', ...}
    """
    if model_name not in OUR_MODELS:
        raise ValueError(
            f"Model {model_name} not configured. "
            f"Available: {list(OUR_MODELS.keys())}"
        )

    model_config = OUR_MODELS[model_name]
    layer = layer or model_config["default_layer"]

    sae_lens_id = model_config["sae_id_template"].format(layer=layer)
    model_id, sae_id = get_neuronpedia_ids(
        model_config["sae_release"],
        layer
    )

    return {
        "model_id": model_id,
        "sae_id": sae_id,
        "sae_lens_id": sae_lens_id,
        "layer": layer,
    }
