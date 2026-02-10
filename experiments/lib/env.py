"""Environment detection, path resolution, and device configuration.

Supports three environments:
  - "colab":  Google Colab (google.colab importable)
  - "remote": Remote GPU box (Linux + CUDA, e.g. Lambda Labs, RunPod)
  - "local":  Local development (macOS/Windows, typically no CUDA)
"""

import gc
import os
from pathlib import Path

import torch
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment() -> str:
    """Detect runtime environment: 'colab', 'remote', or 'local'."""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass

    # remote = Linux with CUDA (Lambda Labs, RunPod, etc.)
    if torch.cuda.is_available() and os.name == "posix" and "darwin" not in os.uname().sysname.lower():
        return "remote"

    return "local"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_base_dir() -> Path:
    """Find the repository root by walking up from cwd until pyproject.toml is found.

    Fallback order:
      1. Walk up from cwd looking for pyproject.toml
      2. Colab default: /content/bluedot-project
      3. cwd itself
    """
    env = detect_environment()
    if env == "colab":
        return Path("/content/bluedot-project")

    # walk up from cwd looking for pyproject.toml (the repo root marker)
    candidate = Path.cwd().resolve()
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    return Path.cwd().resolve()


def setup_paths(base_dir: Path) -> dict:
    """Create standard project directories and return them as a dict.

    Returns dict with keys: data_dir, cache_dir, train_dir, eval_dir
    """
    data_dir  = base_dir / "data"
    cache_dir = base_dir / "experiments" / "cache"
    train_dir = data_dir / "training" / "prompts_4x"
    eval_dir  = data_dir / "evals" / "test"

    for d in [data_dir, cache_dir, train_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_dir":  data_dir,
        "cache_dir": cache_dir,
        "train_dir": train_dir,
        "eval_dir":  eval_dir,
    }


# ---------------------------------------------------------------------------
# Device and VRAM
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_vram_gb() -> float:
    """Return total VRAM in GB for the first CUDA device, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def should_quantize() -> bool:
    """Decide whether to use 8-bit quantization.

    2026-02-08: Always quantize to 8-bit. Saves ~8 GB VRAM with negligible
    impact on probe quality (we compare within the same quantization setting).
    Revisit if we need exact fp16 parity with the paper's results.
    """
    return True


def recommend_batch_size(vram_gb: float = None, params_b: float = 8) -> int:
    """Suggest batch size for activation extraction based on VRAM and model size.

    params_b: model size in billions (8 for Llama-3.1-8B, 70 for Llama-3.3-70B, etc.)

    Estimates assume max_length=8192, 8-bit quantization.
    Larger models need proportionally smaller batches.
    """
    if vram_gb is None:
        vram_gb = get_gpu_vram_gb()

    # available VRAM after model is loaded (rough: model takes ~1 GB per B params in 8-bit)
    available = vram_gb - params_b

    if available < 4:
        return 1
    if available < 10:
        return 2
    if available < 20:
        return 4
    if available < 40:
        return 8
    return 16


def free_gpu_memory():
    """Free GPU memory between heavy operations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def setup_hf_auth(token: str = None):
    """Authenticate with HuggingFace.

    Token resolution order:
      1. Explicit token parameter (for Colab where .env isn't accessible via VSCode)
      2. HF_TOKEN environment variable (from .env or shell)
      3. Colab Secrets (google.colab.userdata)

    Returns the resolved HF_TOKEN string, or None if not found.
    """
    hf_token = token

    if not hf_token:
        load_dotenv(override=True)
        hf_token = os.getenv("HF_TOKEN")

    if not hf_token and detect_environment() == "colab":
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token)
        print("Logged in to HuggingFace")
    else:
        print("No HF_TOKEN found. Pass token= directly, set in .env, or use Colab Secrets.")

    # return hf_token


# ---------------------------------------------------------------------------
# Colab file transfer (no-ops outside Colab)
# ---------------------------------------------------------------------------

def download_from_colab(cache_dir: Path, filename: str = None, cache_prefix: str = "v2b"):
    """Download file(s) from Colab to local machine. No-op outside Colab."""
    if detect_environment() != "colab":
        return

    from google.colab import files
    import shutil

    if filename:
        filepath = cache_dir / filename
        if filepath.exists():
            files.download(str(filepath))
            print(f"Downloaded: {filename}")
        else:
            print(f"File not found: {filepath}")
    else:
        import tempfile
        zip_name = f"bluedot_cache_{cache_prefix}"
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in cache_dir.glob(f"{cache_prefix}_*"):
                shutil.copy2(f, Path(tmpdir) / f.name)
            shutil.make_archive(f"/content/{zip_name}", "zip", tmpdir)
        files.download(f"/content/{zip_name}.zip")
        print(f"Downloaded: {zip_name}.zip")


def upload_to_colab(cache_dir: Path):
    """Upload files from local machine to Colab's cache directory. No-op outside Colab."""
    if detect_environment() != "colab":
        return

    from google.colab import files
    print("Select files to upload...")
    uploaded = files.upload()
    for fname, content in uploaded.items():
        dest = cache_dir / fname
        dest.write_bytes(content)
        print(f"Saved: {dest}")


def list_cache(cache_dir: Path, prefix: str = None):
    """List cached files, optionally filtered by prefix."""
    print(f"\nCache directory: {cache_dir}")
    print("-" * 50)
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("*")):
            if prefix and not f.name.startswith(prefix):
                continue
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<45} {size_mb:>8.2f} MB")
    else:
        print("  (empty)")
