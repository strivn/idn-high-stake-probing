"""Dataset loading, parsing, and download utilities.

All datasets use the paper's JSONL format with 'inputs' and label fields.
Messages are always in [{role, content}] chat format (v2+ convention).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Dataset registry (paper's R2 bucket)
# ---------------------------------------------------------------------------

DATASET_URLS = {
    "train":          "https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/train.jsonl",
    "test":           "https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/training/prompts_4x/test.jsonl",
    "anthropic_test": "https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/anthropic_test_balanced_apr_23.jsonl",
    "toolace_test":   "https://pub-fd16e959a4f14ca48765b437c9425ba6.r2.dev/evals/test/toolace_test_balanced_apr_22.jsonl",
}


def get_dataset_paths(data_dir: Path) -> Dict[str, Path]:
    """Return standard dataset paths relative to data_dir."""
    return {
        "train":          data_dir / "training" / "prompts_4x" / "train.jsonl",
        "test":           data_dir / "training" / "prompts_4x" / "test.jsonl",
        "anthropic_test": data_dir / "evals" / "test" / "anthropic_test_balanced_apr_23.jsonl",
        "toolace_test":   data_dir / "evals" / "test" / "toolace_test_balanced_apr_22.jsonl",
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, path: Path) -> None:
    """Download a file if it doesn't already exist."""
    if path.exists():
        print(f"  Already exists: {path.name}")
        return
    print(f"  Downloading: {path.name}...")
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (compatible; BluedotProject/1.0)"}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; BluedotProject/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    print(f"  Done: {path.name}")


def ensure_datasets(data_dir: Path) -> Dict[str, Path]:
    """Download all datasets if not present. Returns dataset paths dict."""
    paths = get_dataset_paths(data_dir)
    print("Checking datasets...")
    for name, url in DATASET_URLS.items():
        download_file(url, paths[name])
    print("All datasets ready.")
    return paths


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single example from the dataset."""
    id:       str
    messages: List[Dict]   # [{role, content}] chat format
    label:    int           # 1 = high-stakes, 0 = low-stakes


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_label(row: Dict) -> int:
    # Use 'labels' (the paper's field, clean 50/50 split) over 'high_stakes'
    # (boolean derived from scale_labels with different thresholding)
    if "labels" in row:
        return 1 if row["labels"] == "high-stakes" else 0
    if "high_stakes" in row:
        return 1 if row["high_stakes"] else 0
    raise ValueError(f"Cannot find label in row: {row.keys()}")


def parse_messages(row: Dict) -> List[Dict]:
    """Parse inputs field into chat messages.

    The paper's to_dialogue() wraps plain text as [{role: user, content: text}].
    Dialogue inputs are already in message format.
    """
    inputs = row["inputs"]

    # dialogue format: JSON array of {role, content}
    if inputs.startswith("["):
        try:
            return json.loads(inputs)
        except json.JSONDecodeError:
            pass

    # plain text: wrap as user message (matching paper's to_dialogue)
    return [{"role": "user", "content": inputs}]


def load_dataset(path: Path) -> List[Example]:
    """Load a JSONL dataset into a list of Example objects."""
    rows = load_jsonl(path)
    examples = []
    for row in rows:
        examples.append(Example(
            id       = row.get("ids", str(len(examples))),
            messages = parse_messages(row),
            label    = parse_label(row),
        ))
    return examples
