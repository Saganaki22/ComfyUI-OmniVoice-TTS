"""ComfyUI custom nodes for OmniVoice TTS.

Provides nodes:
  - OmniVoiceTTS            -- text -> speech, auto voice selection (random)
  - OmniVoiceVoiceCloneTTS  -- reference audio + text -> cloned-voice speech
  - OmniVoiceVoiceDesignTTS -- text + voice description -> designed voice speech
  - OmniVoiceWhisperLoader  -- load Whisper ASR for auto-transcription

Dependencies are installed via install.py (run by ComfyUI-Manager).
Model weights are auto-downloaded from HuggingFace on first inference.

Supports 600+ languages with zero-shot voice cloning and voice design.
"""

__version__ = "0.2.7"

import logging
import sys
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).parent.resolve()

# Add this folder to sys.path for local imports
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger("OmniVoice")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[OmniVoice] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _check_dependencies() -> bool:
    """Check if omnivoice is installed. Returns True if ready."""
    try:
        import omnivoice
        return True
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f" omnivoice import failed: {e}")
        logger.error(" Run: pip install --no-deps omnivoice")
        logger.error(" Or restart ComfyUI to trigger install.py")
        logger.error("=" * 60)
        return False


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Always register the Whisper loader (doesn't depend on omnivoice)
try:
    from .nodes.whisper_loader import OmniVoiceWhisperLoader
    NODE_CLASS_MAPPINGS["OmniVoiceWhisperLoader"] = OmniVoiceWhisperLoader
    NODE_DISPLAY_NAME_MAPPINGS["OmniVoiceWhisperLoader"] = "OmniVoice Whisper Loader"
except Exception as e:
    logger.warning(f"Failed to register Whisper loader: {e}")

if _check_dependencies():
    try:
        from .nodes.loader import _register_folder
        _register_folder()

        from .nodes.omnivoice_tts import OmniVoiceLongformTTS
        from .nodes.voice_clone_node import OmniVoiceVoiceCloneTTS
        from .nodes.voice_design_node import OmniVoiceVoiceDesignTTS
        from .nodes.multi_speaker_node import OmniVoiceMultiSpeakerTTS

        NODE_CLASS_MAPPINGS.update({
            "OmniVoiceLongformTTS": OmniVoiceLongformTTS,
            "OmniVoiceVoiceCloneTTS": OmniVoiceVoiceCloneTTS,
            "OmniVoiceVoiceDesignTTS": OmniVoiceVoiceDesignTTS,
            "OmniVoiceMultiSpeakerTTS": OmniVoiceMultiSpeakerTTS,
        })

        NODE_DISPLAY_NAME_MAPPINGS.update({
            "OmniVoiceLongformTTS": "OmniVoice Longform TTS",
            "OmniVoiceVoiceCloneTTS": "OmniVoice Voice Clone TTS",
            "OmniVoiceVoiceDesignTTS": "OmniVoice Voice Design TTS",
            "OmniVoiceMultiSpeakerTTS": "OmniVoice Multi-Speaker TTS",
        })

        logger.info(
            f"Registered {len(NODE_CLASS_MAPPINGS)} nodes "
            f"(v{__version__}): {', '.join(NODE_DISPLAY_NAME_MAPPINGS.values())}"
        )

    except Exception as e:
        logger.error(f"Failed to register nodes: {e}", exc_info=True)
else:
    # Fallback: try to install omnivoice with --no-deps to avoid
    # clobbering the user's torch/torchvision/torchaudio stack.
    try:
        import subprocess

        logger.warning("omnivoice not found — attempting to install with --no-deps ...")
        pip_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "omnivoice"]
        result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.warning("OmniVoice installed — RESTART ComfyUI to complete setup.")
        else:
            logger.error(f"Failed to install omnivoice: {result.stderr}")
    except Exception as e:
        logger.error(f"Failed to install omnivoice: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
