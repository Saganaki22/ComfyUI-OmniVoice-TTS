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

__version__ = "0.2.10"

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


def _check_dependencies() -> tuple[bool, list[tuple[str, list[str]]]]:
    """Check if omnivoice and its critical dependencies are importable.

    Returns:
        (ready, missing) where *missing* is a list of (package_name, extra_args)
        tuples.  *extra_args* are pip flags like ``["--upgrade"]`` needed
        beyond a plain ``pip install <pkg>``.
    """
    try:
        import omnivoice
    except Exception as e:
        logger.error(f"Failed to import omnivoice: {e}")
        return False, [("omnivoice", ["--no-deps"])]

    # Sub-dependencies that ``pip install omnivoice --no-deps`` skips.
    missing: list[tuple[str, list[str]]] = []

    try:
        import soxr
    except ImportError:
        missing.append(("soxr", []))

    try:
        import transformers
    except ImportError:
        missing.append(("transformers", ["--upgrade"]))
    else:
        # transformers is installed but may be too old — check version.
        # OmniVoice needs transformers >= 4.57 (HiggsAudio tokenizer support).
        try:
            current = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            if current < (4, 57):
                missing.append(("transformers", ["--upgrade"]))
        except (ValueError, AttributeError):
            pass

    return (len(missing) == 0), missing


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

_deps_ready, _deps_missing = _check_dependencies()

if _deps_ready:
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
    # Fallback: install exactly what's missing.
    # omnivoice itself missing  -> pip install omnivoice --no-deps
    # sub-dep missing/too-old   -> pip install <pkg> [--upgrade]
    _missing_names = [pkg for pkg, _ in _deps_missing]
    logger.error("=" * 60)
    logger.error(f" Missing packages: {', '.join(_missing_names)}")
    logger.error("=" * 60)
    try:
        import subprocess

        for _pkg, _extra_args in _deps_missing:
            _cmd = [sys.executable, "-m", "pip", "install"] + _extra_args + [_pkg]
            logger.warning(f"Installing {_pkg} ...")
            _result = subprocess.run(_cmd, capture_output=True, text=True, timeout=120)
            if _result.returncode == 0:
                logger.warning(f"  {_pkg} installed successfully.")
            else:
                logger.error(f"  Failed to install {_pkg}: {_result.stderr}")

        # Re-check to report final status
        _deps_ready_now, _deps_still_missing = _check_dependencies()
        if _deps_ready_now:
            logger.warning("All dependencies installed — RESTART ComfyUI to load nodes.")
        else:
            _still = [pkg for pkg, _ in _deps_still_missing]
            logger.error(
                f"Dependencies still missing after install: {', '.join(_still)}. "
                f"Try manually: pip install {' '.join(_still)}"
            )
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
