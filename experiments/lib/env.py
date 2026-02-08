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
    return props.total_mem / (1024 ** 3)


def should_quantize(vram_gb: float = None) -> bool:
    """Decide whether to use 8-bit quantization.

    Quantize if VRAM < 20 GB (T4 = 16 GB needs it, A10 = 24 GB does not).
    """
    if vram_gb is None:
        vram_gb = get_gpu_vram_gb()
    return 0 < vram_gb < 20


def recommend_batch_size(vram_gb: float = None) -> int:
    """Suggest batch size for activation extraction based on available VRAM.

    Conservative estimates for Llama-3.1-8B with max_length=8192:
      < 20 GB (T4 16GB):   batch_size = 2
      20-30 GB (A10 24GB): batch_size = 4
      30-50 GB (A100 40GB): batch_size = 8
      > 50 GB (A100 80GB): batch_size = 16
    """
    if vram_gb is None:
        vram_gb = get_gpu_vram_gb()
    if vram_gb < 20:
        return 2
    if vram_gb < 30:
        return 4
    if vram_gb < 50:
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

    return hf_token


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
