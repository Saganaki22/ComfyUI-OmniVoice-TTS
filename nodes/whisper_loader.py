"""Whisper ASR model loader for OmniVoice TTS.

Loads Whisper ASR models from ComfyUI/models/audio_encoders/
Users can place any HuggingFace Whisper model in that folder.
If no model is present, auto-downloads the default model.

Also sets HuggingFace cache directory globally so OmniVoice's internal
ASR uses ComfyUI's cache folder instead of downloading repeatedly.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Any

import torch

logger = logging.getLogger("OmniVoice")

# Suppress the duplicate logits-processor warning that some transformers
# versions emit during Whisper generation.  The warning is harmless —
# transformers is just informing the user that its internally-created
# SuppressTokensLogitsProcessor takes precedence over one that the
# pipeline also passes.  Only visible on newer transformers releases.
warnings.filterwarnings(
    "ignore",
    message=r"A custom logits processor of type .* has been passed to `.generate\(\)`",
    category=UserWarning,
)


def _get_audio_encoders_dir() -> Path:
    """Get or create the audio_encoders folder path."""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir) / "audio_encoders"
    except ImportError:
        base = Path(__file__).resolve().parent.parent / "checkpoints" / "audio_encoders"
    base.mkdir(parents=True, exist_ok=True)
    return base


# Only set HuggingFace cache environment variables if they are not already
# configured by the user or another plugin. This avoids hijacking cache paths
# for unrelated nodes that also use transformers/huggingface_hub.
_CACHE_DIR = str(_get_audio_encoders_dir())
for _var in ("TRANSFORMERS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
    if _var not in os.environ:
        os.environ[_var] = _CACHE_DIR
logger.info(f"HuggingFace cache set to: {_CACHE_DIR}")

# Default Whisper model for auto-download
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"

# Popular Whisper models for auto-download option
POPULAR_WHISPER_MODELS = {
    "whisper-large-v3-turbo (auto-download)": "openai/whisper-large-v3-turbo",
    "whisper-large-v3 (auto-download)": "openai/whisper-large-v3",
    "whisper-medium (auto-download)": "openai/whisper-medium",
    "whisper-small (auto-download)": "openai/whisper-small",
    "whisper-tiny (auto-download)": "openai/whisper-tiny",
}


def _register_folder() -> None:
    """Register audio_encoders folder with ComfyUI's folder_paths."""
    try:
        import folder_paths
        base = str(_get_audio_encoders_dir())
        # Check if already registered
        if "audio_encoders" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("audio_encoders", base)
            logger.info(f"Audio encoders folder registered: {base}")
    except ImportError:
        pass


def _is_whisper_downloaded(repo_id: str) -> bool:
    """Check if a Whisper model is already downloaded locally."""
    safe_name = repo_id.replace("/", "_")
    model_path = _get_audio_encoders_dir() / safe_name
    if not model_path.is_dir():
        return False
    has_config = (model_path / "config.json").is_file()
    try:
        has_model = any(
            f.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            for f in model_path.iterdir()
            if f.is_file()
        )
    except PermissionError:
        return False
    return has_config or has_model


def get_whisper_model_names() -> list[str]:
    """Get list of available Whisper models from folder + auto-download options."""
    names = []

    # Always include auto-download options (keeps saved workflow values valid
    # even after download — avoids "node not found" on second run)
    for display_name in POPULAR_WHISPER_MODELS:
        names.append(display_name)

    # Add local models from the folder (user-added or previously downloaded)
    base = _get_audio_encoders_dir()
    known_safe_names = {repo_id.replace("/", "_") for repo_id in POPULAR_WHISPER_MODELS.values()}

    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name not in known_safe_names:
                has_config = (entry / "config.json").is_file()
                try:
                    has_model = any(
                        f.suffix in {".safetensors", ".bin", ".pt", ".pth"}
                        for f in entry.iterdir()
                        if f.is_file()
                    )
                except PermissionError:
                    continue
                if has_config or has_model:
                    names.append(entry.name)
    except OSError:
        pass

    return names


def download_whisper_model(repo_id: str) -> Path:
    """Download Whisper model from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID (e.g., "openai/whisper-large-v3-turbo")

    Returns:
        Path to the downloaded model directory
    """
    # Create a safe folder name from repo ID
    safe_name = repo_id.replace("/", "_")
    dest = _get_audio_encoders_dir() / safe_name

    if dest.is_dir() and any(dest.iterdir()):
        logger.info(f"Whisper model already downloaded: {dest}")
        return dest

    logger.info(f"Downloading Whisper model '{repo_id}'...")
    logger.info(f"Destination: {dest}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
        )
        logger.info(f"Whisper model downloaded to: {dest}")
        return dest
    except Exception as e:
        logger.error(f"Whisper model download failed: {e}")
        raise


def _get_repo_id_from_safe_name(safe_name: str) -> Optional[str]:
    """Convert safe folder name back to repo ID if it matches a known model."""
    for display_name, repo_id in POPULAR_WHISPER_MODELS.items():
        if repo_id.replace("/", "_") == safe_name:
            return repo_id
    return None


def load_whisper_pipeline(model_name: str, device: str = "auto", dtype: str = "auto"):
    """Load Whisper ASR pipeline.

    Args:
        model_name: Model name from dropdown (local folder or auto-download)
        device: Device to load on ("auto", "cuda", "cpu")
        dtype: Model precision ("auto", "bf16", "fp16", "fp32")

    Returns:
        HuggingFace ASR pipeline
    """
    from transformers import pipeline as hf_pipeline

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_str = "xpu"
        else:
            device_str = "cpu"
    else:
        device_str = device

    # Resolve dtype
    if dtype == "auto":
        if device_str == "cuda":
            # Use bf16 on Ampere+ (compute capability >= 8.0), fp16 otherwise
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                asr_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                asr_dtype = torch.float16
        elif device_str == "xpu":
            asr_dtype = torch.bfloat16
        else:
            asr_dtype = torch.float32
    elif dtype == "bf16":
        asr_dtype = torch.bfloat16
    elif dtype == "fp16":
        asr_dtype = torch.float16
    else:
        asr_dtype = torch.float32

    # Check if it's an auto-download option (has suffix)
    if model_name in POPULAR_WHISPER_MODELS:
        repo_id = POPULAR_WHISPER_MODELS[model_name]
        model_path = download_whisper_model(repo_id)
        logger.info(f"Loading Whisper ASR from: {model_path}")
    else:
        # It's a local folder name - could be a downloaded HF model or user-added
        model_path = _get_audio_encoders_dir() / model_name
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Whisper model not found: {model_path}\n"
                f"Place your model in: ComfyUI/models/audio_encoders/{model_name}/"
            )
        logger.info(f"Loading Whisper ASR from local folder: {model_path}")

    logger.info(f"Loading Whisper ASR pipeline on {device_str} with dtype {asr_dtype}...")

    # Load pipeline from local path
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        torch_dtype=asr_dtype,
        device=device_str,
    )

    logger.info(f"Whisper ASR model loaded on {device_str} ({asr_dtype}).")
    return pipe


def find_local_whisper_model() -> Optional[str]:
    """Find the best locally available Whisper model folder name.

    Returns the safe folder name (e.g. ``"openai_whisper-large-v3-turbo"``)
    suitable for passing to :func:`load_whisper_pipeline`, or ``None`` if no
    model is present.
    """
    # Check default model first
    if _is_whisper_downloaded(DEFAULT_WHISPER_MODEL):
        return DEFAULT_WHISPER_MODEL.replace("/", "_")

    # Check popular models
    for _display, repo_id in POPULAR_WHISPER_MODELS.items():
        if _is_whisper_downloaded(repo_id):
            return repo_id.replace("/", "_")

    # Check for any user-added model folder
    base = _get_audio_encoders_dir()
    known = {rid.replace("/", "_") for rid in POPULAR_WHISPER_MODELS.values()}
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name not in known:
                has_config = (entry / "config.json").is_file()
                try:
                    has_model = any(
                        f.suffix in {".safetensors", ".bin", ".pt", ".pth"}
                        for f in entry.iterdir()
                        if f.is_file()
                    )
                except PermissionError:
                    continue
                if has_config or has_model:
                    return entry.name
    except OSError:
        pass

    return None


class OmniVoiceWhisperLoader:
    """Load Whisper ASR model for OmniVoice auto-transcription.

    Models are loaded from ComfyUI/models/audio_encoders/
    You can place any HuggingFace-compatible Whisper model there.

    If you select an "auto-download" option, the model will be downloaded
    to that folder automatically on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_whisper_model_names(),
                    {
                        "tooltip": (
                            "Whisper model for auto-transcription. "
                            "Select an 'auto-download' option to download on first use, "
                            "or place your own model in ComfyUI/models/audio_encoders/"
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Device to load the ASR model on.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model precision. 'auto' uses bf16 on Ampere+ GPUs, "
                            "fp16 on older GPUs, fp32 on CPU."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WHISPER_ASR",)
    RETURN_NAMES = ("whisper_model",)
    FUNCTION = "load"
    CATEGORY = "OmniVoice"
    DESCRIPTION = (
        "Load Whisper ASR model for OmniVoice auto-transcription. "
        "Models are stored in ComfyUI/models/audio_encoders/"
    )

    def load(self, model: str, device: str, dtype: str):
        # Load the pipeline
        pipe = load_whisper_pipeline(model, device, dtype)

        # Return as a dict with metadata
        return ({
            "pipeline": pipe,
            "model_name": model,
            "device": device,
            "dtype": dtype,
        },)


# Register the folder on import
_register_folder()
