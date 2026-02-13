"""Neuronpedia API integration for fetching feature explanations.

Supports Llama Scope, Gemma Scope 2, and Goodfire SAE feature lookups with caching.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

from .neuronpedia_config import (
    get_neuronpedia_ids,
    auto_detect_neuronpedia_ids,
    get_model_neuronpedia_config,
    OUR_MODELS,
)


def fetch_explanations(
    model_id: str,
    sae_id: str,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False
) -> Dict[int, str]:
    """
    Fetch feature explanations from Neuronpedia API with caching.

    Args:
        model_id: Neuronpedia model ID (e.g., "llama3.1-8b")
        sae_id: Neuronpedia SAE ID (e.g., "12-llamascoperes-8x")
        cache_dir: Directory for caching. If None, uses current dir/.cache/
        force_refresh: Force re-download even if cached

    Returns:
        Dictionary mapping feature_id (int) -> explanation (str)

    Example:
        >>> explanations = fetch_explanations("llama3.1-8b", "12-llamascoperes-8x")
        >>> print(explanations[1234])
        'Emergency and urgent situations'
    """
    # Setup cache
    if cache_dir is None:
        cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{model_id}_{sae_id}_explanations.json"

    # Try cache first
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                # Convert string keys back to int
                explanations = {int(k): v for k, v in cached.items()}
                print(f"📚 Loaded {len(explanations)} explanations from cache: {cache_file.name}")
                return explanations
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Cache corrupted ({e}), re-downloading...")

    # Fetch from API
    url = f"https://www.neuronpedia.org/api/explanation/export"
    params = {"modelId": model_id, "saeId": sae_id}

    print(f"🌐 Fetching explanations from Neuronpedia API...")
    print(f"   URL: {url}?modelId={model_id}&saeId={sae_id}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        explanations = {}

        for entry in data:
            feature_idx = int(entry.get('index', -1))
            if feature_idx >= 0:
                desc = entry.get('description', 'No description available')
                explanations[feature_idx] = desc

        print(f"✅ Downloaded {len(explanations)} feature explanations")

        # Cache results
        with open(cache_file, 'w') as f:
            cache_data = {str(k): v for k, v in explanations.items()}
            json.dump(cache_data, f, indent=2)
        print(f"💾 Cached to {cache_file.name}")

        return explanations

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch explanations: {e}")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Failed to parse response: {e}")
        return {}


def get_explanation(
    feature_id: int,
    explanations: Dict[int, str],
    model_id: str = "",
    sae_id: str = ""
) -> str:
    """
    Get explanation for a specific feature with fallback.

    Args:
        feature_id: Feature index
        explanations: Dictionary of explanations from fetch_explanations()
        model_id: Model ID for generating Neuronpedia link (optional)
        sae_id: SAE ID for generating Neuronpedia link (optional)

    Returns:
        Explanation string or fallback message with link
    """
    if feature_id in explanations:
        return explanations[feature_id]

    # Fallback with link
    if model_id and sae_id:
        link = neuronpedia_url(feature_id, model_id, sae_id)
        return f"No explanation available. Check: {link}"

    return f"No explanation available for feature {feature_id}"


def neuronpedia_url(feature_id: int, model_id: str, sae_id: str) -> str:
    """
    Generate Neuronpedia URL for a feature.

    Args:
        feature_id: Feature index
        model_id: Neuronpedia model ID
        sae_id: Neuronpedia SAE ID

    Returns:
        Full Neuronpedia URL

    Example:
        >>> url = neuronpedia_url(1234, "llama3.1-8b", "12-llamascoperes-8x")
        >>> print(url)
        https://www.neuronpedia.org/llama3.1-8b/12-llamascoperes-8x/1234
    """
    return f"https://www.neuronpedia.org/{model_id}/{sae_id}/{feature_id}"


def format_feature_with_explanation(
    feature_id: int,
    activation: float,
    explanations: Dict[int, str],
    model_id: str = "",
    sae_id: str = "",
    max_length: int = 100
) -> str:
    """
    Format a feature with its explanation for display.

    Args:
        feature_id: Feature index
        activation: Activation strength
        explanations: Dictionary of explanations
        model_id: Model ID for link
        sae_id: SAE ID for link
        max_length: Maximum explanation length before truncation

    Returns:
        Formatted string with feature ID, activation, and explanation
    """
    explanation = get_explanation(feature_id, explanations, model_id, sae_id)

    # Truncate long explanations
    if len(explanation) > max_length:
        explanation = explanation[:max_length-3] + "..."

    url = neuronpedia_url(feature_id, model_id, sae_id) if model_id and sae_id else ""

    if url:
        return f"Feature {feature_id:5d} (act={activation:.3f}): {explanation}\n            {url}"
    else:
        return f"Feature {feature_id:5d} (act={activation:.3f}): {explanation}"


def llama_scope_sae_id(layer: int, expansion: str = "8x") -> str:
    """
    Generate Neuronpedia SAE ID for Llama Scope.

    Args:
        layer: Layer number (0-31)
        expansion: "8x" (32K features) or "32x" (128K features)

    Returns:
        Neuronpedia SAE ID string

    Example:
        >>> sae_id = llama_scope_sae_id(12, "8x")
        >>> print(sae_id)
        12-llamascope-res-32k
    """
    # Map expansion to Neuronpedia format
    width_map = {"8x": "32k", "32x": "131k"}
    width = width_map.get(expansion, "32k")
    return f"{layer}-llamascope-res-{width}"


def get_config_for_model(model_name: str, layer: Optional[int] = None) -> Dict[str, str]:
    """
    Get Neuronpedia configuration for a model (convenience wrapper).

    Args:
        model_name: HuggingFace model name
        layer: Layer number (uses default if None)

    Returns:
        Dict with model_id, sae_id, sae_lens_id

    Example:
        >>> config = get_config_for_model("meta-llama/Llama-3.1-8B-Instruct", 12)
        >>> explanations = fetch_explanations(config["model_id"], config["sae_id"])
    """
    return get_model_neuronpedia_config(model_name, layer)


def batch_lookup(
    feature_ids: List[int],
    explanations: Dict[int, str],
    model_id: str,
    sae_id: str
) -> Dict[int, str]:
    """
    Lookup multiple features at once.

    Args:
        feature_ids: List of feature indices
        explanations: Dictionary of explanations
        model_id: Model ID
        sae_id: SAE ID

    Returns:
        Dictionary mapping feature_id -> explanation (with fallbacks)
    """
    return {
        fid: get_explanation(fid, explanations, model_id, sae_id)
        for fid in feature_ids
    }
