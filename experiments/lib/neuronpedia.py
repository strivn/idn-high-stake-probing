"""Neuronpedia API integration for fetching feature explanations.

Supports Llama Scope, Gemma Scope 2, and Goodfire SAE feature lookups with caching.

Uses the per-feature API: /api/feature/{modelId}/{saeId}/{index}
The bulk export endpoint (/api/explanation/export) was deprecated in late 2025.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any

from .neuronpedia_config import (
    get_neuronpedia_ids,
    auto_detect_neuronpedia_ids,
    get_model_neuronpedia_config,
    OUR_MODELS,
)

FEATURE_API = "https://www.neuronpedia.org/api/feature"


def _extract_description(data: Dict[str, Any]) -> str:
    """Extract description from a Neuronpedia feature API response.

    Checks multiple known locations since the API nests explanations differently.
    """
    if "description" in data and data["description"]:
        return data["description"]

    if "explanations" in data and data["explanations"]:
        first = data["explanations"][0]
        if isinstance(first, dict) and "description" in first:
            return first["description"]

    # Fallback: summarize from top positive tokens
    pos_str = data.get("pos_str", [])
    if pos_str:
        tokens = [t.strip() for t in pos_str[:5] if t.strip()]
        if tokens:
            return f"[auto: top tokens] {', '.join(tokens)}"

    return ""


def _fetch_single(model_id: str, sae_id: str, feature_id: int, timeout: float = 10) -> str:
    """Fetch explanation for one feature from the per-feature API."""
    url = f"{FEATURE_API}/{model_id}/{sae_id}/{feature_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return _extract_description(resp.json())
    except Exception:
        return ""


class LazyExplanations(dict):
    """Dict-like object that fetches feature explanations on demand.

    Keeps the same interface as a plain dict so notebook code like
    `explanations.get(feat_id, "...")` and `explanations[feat_id]` just works.
    Fetched results are cached to disk incrementally.
    """

    def __init__(self, model_id: str, sae_id: str, cache_file: Path, delay: float = 0.05):
        super().__init__()
        self._model_id   = model_id
        self._sae_id     = sae_id
        self._cache_file = cache_file
        self._delay      = delay
        self._fetched     = set()  # track which IDs we already tried (avoid re-fetching empties)

        # Load existing cache from disk
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                for k, v in cached.items():
                    super().__setitem__(int(k), v)
                    self._fetched.add(int(k))
            except (json.JSONDecodeError, ValueError):
                pass

    def _fetch_and_cache(self, feature_id: int) -> str:
        desc = _fetch_single(self._model_id, self._sae_id, feature_id)
        super().__setitem__(feature_id, desc)
        self._fetched.add(feature_id)
        self._save()
        if self._delay > 0:
            time.sleep(self._delay)
        return desc

    def _save(self):
        with open(self._cache_file, "w") as f:
            json.dump({str(k): v for k, v in self.items()}, f, indent=2)

    def __missing__(self, feature_id: int) -> str:
        return self._fetch_and_cache(feature_id)

    def get(self, feature_id, default=None):
        if feature_id in self or feature_id in self._fetched:
            return super().get(feature_id, default)
        desc = self._fetch_and_cache(feature_id)
        return desc if desc else default

    def prefetch(self, feature_ids: List[int]):
        """Batch-fetch a list of feature IDs (skipping already cached)."""
        to_fetch = [fid for fid in feature_ids if fid not in self._fetched]
        if not to_fetch:
            return
        print(f"Fetching {len(to_fetch)} explanations from Neuronpedia ({len(feature_ids) - len(to_fetch)} cached)...")
        for i, fid in enumerate(to_fetch):
            _fetch_single_result = _fetch_single(self._model_id, self._sae_id, fid)
            super().__setitem__(fid, _fetch_single_result)
            self._fetched.add(fid)
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(to_fetch)}")
            if self._delay > 0 and i < len(to_fetch) - 1:
                time.sleep(self._delay)
        self._save()
        print(f"Done. {len(self)} total explanations cached.")


def fetch_explanations(
    model_id: str,
    sae_id: str,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> LazyExplanations:
    """Return a lazy dict that fetches feature explanations on demand.

    Drop-in replacement for the old bulk-export fetch. Same signature, same
    usage (explanations[feat_id], explanations.get(feat_id, "...")), but
    fetches per-feature from /api/feature/ since the bulk export was deprecated.

    Args:
        model_id:      Neuronpedia model ID (e.g., "llama3.1-8b")
        sae_id:        Neuronpedia SAE ID (e.g., "12-llamascope-res-32k")
        cache_dir:     Directory for caching (default: cwd/.cache/)
        force_refresh: Delete cache and start fresh

    Returns:
        LazyExplanations dict — access any feature_id and it auto-fetches.
    """
    if cache_dir is None:
        cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f"{model_id}_{sae_id}_explanations.json"

    if force_refresh and cache_file.exists():
        cache_file.unlink()

    explanations = LazyExplanations(model_id, sae_id, cache_file)
    n_cached = len(explanations)
    if n_cached:
        print(f"Loaded {n_cached} cached explanations from {cache_file.name}")
    else:
        print(f"Explanations will be fetched on demand from Neuronpedia API")
    return explanations


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
    """Lookup multiple features, triggering lazy fetches if needed.

    Args:
        feature_ids:  List of feature indices
        explanations: LazyExplanations dict (or regular dict)
        model_id:     Model ID for fallback URLs
        sae_id:       SAE ID for fallback URLs

    Returns:
        Dict mapping feature_id -> explanation (with URL fallback if missing)
    """
    # Prefetch if the dict supports it (LazyExplanations)
    if hasattr(explanations, "prefetch"):
        explanations.prefetch(feature_ids)

    return {
        fid: get_explanation(fid, explanations, model_id, sae_id)
        for fid in feature_ids
    }
