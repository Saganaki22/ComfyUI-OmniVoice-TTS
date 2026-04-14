"""Model loading and utilities for OmniVoice TTS.

Handles:
  - Model folder registration with ComfyUI
  - Auto-download from HuggingFace
  - Device and precision resolution
  - Model loading with OmniVoice.from_pretrained
  - Audio format conversion for ComfyUI
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("OmniVoice")

# Model folder name in ComfyUI/models/
MODELS_FOLDER_NAME = "omnivoice"


try:
    import comfy.model_patcher as _cmp

    class OmniVoicePatcher(_cmp.ModelPatcher):
        """ModelPatcher subclass with aimdo dynamic VRAM reporting."""

        def is_dynamic(self):
            return True

        def _vbar_get(self):
            vbars = getattr(self.model, "dynamic_vbars", {})
            if vbars:
                return next(iter(vbars.values()))
            return None

    del _cmp
except ImportError:
    # ComfyUI not available (testing outside ComfyUI) — fall back to base
    OmniVoicePatcher = None


def _get_models_base() -> Path:
    """Get or create the models folder path."""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir) / MODELS_FOLDER_NAME
    except ImportError:
        base = Path(__file__).resolve().parent.parent / "checkpoints" / MODELS_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _register_folder() -> None:
    """Register models folder with ComfyUI's folder_paths."""
    try:
        import folder_paths
        base = str(_get_models_base())
        folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, base)
        logger.info(f"Models folder registered: {base}")
    except ImportError:
        pass


# HuggingFace model configuration
# Keys are display names (shown in dropdown), repo_id is the actual HF repo
HF_MODELS = {
    "OmniVoice": {
        "repo_id": "k2-fsa/OmniVoice",
        "url": "https://huggingface.co/k2-fsa/OmniVoice",
        "description": "Full OmniVoice model - 600+ languages (fp32, ~4GB)",
    },
    "OmniVoice-bf16": {
        "repo_id": "drbaph/OmniVoice-bf16",
        "url": "https://huggingface.co/drbaph/OmniVoice-bf16",
        "description": "Bfloat16 quantized OmniVoice - smaller VRAM (~2GB)",
    },
}
HF_DEFAULT_MODEL = "k2-fsa/OmniVoice"
_AUTO_DOWNLOAD_SUFFIX = " (auto download)"


def _auto_download_model(model_name: str = HF_DEFAULT_MODEL) -> bool:
    """Download model from HuggingFace if not already present."""
    if model_name not in HF_MODELS:
        logger.error(f"Unknown model: {model_name}")
        return False

    cfg = HF_MODELS[model_name]
    repo_id = cfg["repo_id"]
    dest = _get_models_base() / model_name.replace("/", "_")

    if _is_model_downloaded(model_name):
        return True

    logger.info(f"Downloading '{model_name}' ({cfg['description']}) from HuggingFace...")
    logger.info(f"Repo: {repo_id}")
    logger.info(f"Destination: {dest}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.h5"],
        )
        logger.info(f"Model downloaded to: {dest}")
        return True
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return False


def _is_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    base = _get_models_base()
    safe_name = model_name.replace("/", "_")
    model_path = base / safe_name
    if not model_path.is_dir():
        return False
    # Check for config.json or any weight files
    has_config = (model_path / "config.json").is_file()
    try:
        has_weights = any(
            f.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
            for f in model_path.iterdir()
            if f.is_file()
        )
    except PermissionError:
        return False
    return has_config or has_weights


def get_model_names() -> list[str]:
    """Get list of available models (downloaded + auto-download options)."""
    base = _get_models_base()
    names = []

    # Build a set of known safe folder names for deduplication.
    # This covers both the display key ("OmniVoice-bf16") and the
    # repo-id-based safe name ("k2-fsa_OmniVoice") so local folders
    # that match any known model are never duplicated in the dropdown.
    _known_folders: set[str] = set()
    for _mn, _cfg in HF_MODELS.items():
        _known_folders.add(_mn)                          # display key
        _known_folders.add(_cfg["repo_id"].replace("/", "_"))  # repo-id safe name

    # Add HF models — always keep the "(auto download)" entry so saved
    # workflow values never break after download (mirrors whisper_loader.py).
    for model_name in HF_MODELS.keys():
        names.append(f"{model_name}{_AUTO_DOWNLOAD_SUFFIX}")
        if _is_model_downloaded(model_name):
            names.append(model_name)

    # Add any local models in the folder
    try:
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            safe_name = entry.name
            # Skip folders that belong to a known HF model (dedup)
            if safe_name in _known_folders:
                continue
            # Check if it has model files
            has_config = (entry / "config.json").is_file()
            try:
                has_weights = any(
                    f.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
                    for f in entry.iterdir()
                    if f.is_file()
                )
            except PermissionError:
                continue
            if has_config or has_weights:
                names.append(safe_name)
    except OSError:
        pass

    return names


def _strip_auto_download_suffix(name: str) -> str:
    """Remove the auto-download suffix from a model name."""
    if name.endswith(_AUTO_DOWNLOAD_SUFFIX):
        return name[: -len(_AUTO_DOWNLOAD_SUFFIX)]
    return name


def _supports_bfloat16() -> bool:
    """Check if the GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def resolve_device(device_choice: str) -> Tuple[str, Optional[torch.dtype]]:
    """Resolve device choice to actual device string."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda", None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", None
        logger.warning("No CUDA or MPS GPU detected — falling back to CPU.")
        return "cpu", None
    return device_choice, None


def resolve_precision(precision_choice: str, device: str) -> torch.dtype:
    """Resolve precision choice to torch dtype."""
    if precision_choice == "auto":
        if device == "cuda":
            return torch.bfloat16 if _supports_bfloat16() else torch.float16
        elif device == "mps":
            return torch.float16
        return torch.float32
    if precision_choice == "bf16":
        if device == "cuda" and not _supports_bfloat16():
            logger.warning(
                "bfloat16 requested but GPU does not support it (compute capability < 8.0). "
                "Consider using 'fp16' instead."
            )
        return torch.bfloat16
    if precision_choice == "fp16":
        return torch.float16
    return torch.float32


def to_numpy_audio(audio) -> np.ndarray:
    """Convert model output to numpy array, handling both tensor and numpy input.

    omnivoice.generate() may return torch tensors or numpy arrays depending on
    version. This normalizes the output to a 1-D numpy array of samples.
    """
    import torch

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    # Squeeze leading batch dim if present (1, T) -> (T,)
    if audio.ndim >= 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    # Ensure 1-D: (C, T) -> flatten to (T,) for mono TTS output
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return audio


def numpy_audio_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    """Convert numpy audio array to ComfyUI AUDIO format.

    Args:
        audio_np: Audio samples as numpy array (samples,) or (channels, samples)
        sample_rate: Sample rate in Hz

    Returns:
        dict with 'waveform' tensor (1, channels, samples) and 'sample_rate'
    """
    import torch

    # Ensure float32 for ComfyUI
    audio_np = audio_np.astype(np.float32)

    # Handle different input shapes
    if audio_np.ndim == 1:
        # (samples,) -> (1, 1, samples)
        audio_np = audio_np[np.newaxis, np.newaxis, :]
    elif audio_np.ndim == 2:
        # (channels, samples) -> (1, channels, samples)
        audio_np = audio_np[np.newaxis, :, :]
    # else: already (batch, channels, samples) — pass through as-is

    waveform = torch.from_numpy(audio_np).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def comfy_audio_to_numpy(audio_dict: dict, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Convert ComfyUI AUDIO format to numpy array.

    Args:
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate'
        target_sr: Optional target sample rate to resample to

    Returns:
        Tuple of (audio_np, sample_rate) where audio_np is (samples,)
    """
    waveform = audio_dict["waveform"]
    source_sr = audio_dict["sample_rate"]

    # ComfyUI AUDIO format: (batch, channels, samples)
    # Ensure we have a tensor, convert from numpy if needed
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(waveform[0]).float()
    else:
        # It's already a tensor
        wav = waveform[0].float()

    # Handle different channel configurations
    if wav.dim() == 1:
        # Already (samples,) - use as-is
        pass
    elif wav.shape[0] > 1:
        # Multi-channel -> mix down to mono: (samples,)
        wav = wav.mean(dim=0)
    elif wav.shape[0] == 1:
        # Mono -> squeeze to (samples,)
        wav = wav.squeeze(0)
    else:
        # Empty or weird shape - flatten to 1D
        wav = wav.flatten()

    # Convert to numpy, ensuring tensor is on CPU first
    audio_np = wav.cpu().numpy() if hasattr(wav, 'cpu') else np.array(wav)

    # Resample if needed
    if target_sr is not None and source_sr != target_sr:
        import soxr
        audio_np = soxr.resample(audio_np, source_sr, target_sr)
        return audio_np, target_sr

    return audio_np, source_sr


def _resolve_attn_implementation(attention: str, device: str) -> str | None:
    """Resolve attention implementation for OmniVoice's Qwen3 LLM backbone.

    OmniVoice only supports "eager" attention through transformers.
    "sage_attention" is handled via post-load monkey-patching.

    Args:
        attention: User's attention choice ("auto", "eager", "sage_attention")
        device: Target device

    Returns:
        attn_implementation value for transformers ("eager" or None)
    """
    if attention == "auto":
        return None

    if attention == "eager":
        return "eager"

    if attention == "sage_attention":
        try:
            from .sage_attention_patch import SAGE_ATTENTION_AVAILABLE
            if SAGE_ATTENTION_AVAILABLE:
                # Load with eager, then monkey-patch Qwen3Attention.forward
                return "eager"
        except ImportError:
            pass

        # V2 not available — try V1 fallback
        try:
            from .sage_attention_v1.sage_attention_v1_patch import SAGE_ATTN_V1_AVAILABLE
            if SAGE_ATTN_V1_AVAILABLE:
                logger.info("SageAttention V2 not available, using V1 (Triton) fallback.")
                return "eager"
        except ImportError:
            pass
        # Neither V2 nor V1 available
        logger.warning(
            "sage_attention requested but sageattention is not installed. "
            "Install with: pip install sageattention\n"
            "Falling back to eager."
        )
        return "eager"

    return None


def load_model(
    model_name: str,
    device: str,
    precision: str,
    attention: str,
):
    """Load OmniVoice model.

    Args:
        model_name: HuggingFace model name or local folder name
        device: Device choice ("auto", "cuda", "cpu", "mps")
        precision: Precision choice ("auto", "bf16", "fp16", "fp32")
        attention: Attention implementation ("auto", "eager", "sage_attention")

    Returns:
        Tuple of (model, None) - no tokenizer needed for OmniVoice
    """
    from omnivoice import OmniVoice

    model_name = _strip_auto_download_suffix(model_name)
    device_str, _ = resolve_device(device)
    dtype = resolve_precision(precision, device_str)

    # Resolve the actual model identifier to pass to OmniVoice.from_pretrained
    # 1. If it's a known HF model (by display name), download to our folder then use local path
    # 2. If it's a local path, use the path
    # 3. Otherwise pass as-is (might be an unknown HF repo)
    model_identifier = model_name

    if model_name in HF_MODELS:
        # Known HF model - auto-download to our models folder first
        if not _is_model_downloaded(model_name):
            logger.info(f"Model '{model_name}' not found locally. Auto-downloading...")
            success = _auto_download_model(model_name)
            if not success:
                raise RuntimeError(f"Failed to download model '{model_name}'")

        # Use local path
        local_path = _get_models_base() / model_name
        model_identifier = str(local_path)
        logger.info(f"Using local model at: {local_path}")
    else:
        # Check if it's a local folder name
        local_path = _get_models_base() / model_name
        if local_path.is_dir():
            model_identifier = str(local_path)
            logger.info(f"Using local model at: {local_path}")

    # Resolve attention implementation
    attn_impl = _resolve_attn_implementation(attention, device_str)

    logger.info(f"Loading OmniVoice: {model_identifier}")
    logger.info(f"Device: {device_str}, Precision: {dtype}, Attention: {attention} -> {attn_impl or 'default'}")

    # Resolve target device string
    if device_str == "cuda":
        target_device = "cuda:0"
    elif device_str == "mps":
        target_device = "mps"
    else:
        target_device = "cpu"

    # Build kwargs — torch_dtype instead of device_map to avoid
    # accelerate dispatch conflicts with cudaMallocAsync
    load_kwargs = {
        "torch_dtype": dtype,
    }
    if attn_impl is not None:
        load_kwargs["attn_implementation"] = attn_impl

    # Load weights in correct dtype, then move to device via native PyTorch
    model = OmniVoice.from_pretrained(model_identifier, **load_kwargs)
    model = model.to(target_device)

    model.eval()

    # Apply SageAttention monkey-patch if requested
    if attention == "sage_attention":
        patched = False
        # Try V2 first (CUDA SM80+)
        if device_str == "cuda":
            try:
                from .sage_attention_patch import set_sage_attention
                set_sage_attention(model)
                patched = True
            except Exception as e:
                logger.warning(f"SageAttention V2 patching failed: {e}.")
        # Try V1 fallback (AMD ROCm / older NVIDIA / CPU)
        if not patched:
            try:
                from .sage_attention_v1.sage_attention_v1_patch import set_sage_attention_v1
                set_sage_attention_v1(model)
                patched = True
            except Exception as e:
                logger.warning(f"SageAttention V1 patching failed: {e}. Using default attention.")
        if not patched:
            logger.info("SageAttention: no compatible version found, using default attention.")

    # Wrap in ComfyUI ModelPatcher for native VRAM management
    patcher = OmniVoicePatcher(
        model,
        load_device=torch.device(target_device),
        offload_device=torch.device("cpu"),
    )

    logger.info("OmniVoice model loaded successfully.")
    return patcher, None  # No tokenizer needed for OmniVoice
