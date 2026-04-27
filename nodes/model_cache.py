"""Model caching and VRAM management for OmniVoice TTS.

Supports:
  - Model caching with configurable keep_loaded
  - CPU offload to free VRAM between runs
  - Native ComfyUI ModelPatcher integration for VRAM management
  - aimdo dynamic VBar for viz panel residency tracking
  - Interrupt handling for cancelled generations
"""

import gc
import logging
import math
import threading
from typing import Any

import torch

logger = logging.getLogger("OmniVoice")

# Global cache state protected by lock.
# _cached_model stores a comfy.model_patcher.ModelPatcher wrapping the raw
# OmniVoice nn.Module.  All public API returns the raw model for inference
# (nodes call .generate() on it), while the patcher is used internally for
# ComfyUI's VRAM tracking.
_cache_lock = threading.Lock()
_cached_model: Any = None          # ModelPatcher or None
_cached_key: tuple = ()
_keep_loaded: bool = False
_offloaded: bool = False

# Whisper ASR cache — kept separate from the OmniVoice model since
# it's loaded by a different node and has its own lifecycle.
_whisper_lock = threading.Lock()
_cached_whisper: Any = None
_cached_whisper_key: tuple = ()

# Cancellation event for interrupt handling
cancel_event: threading.Event = threading.Event()


def get_raw_model(patcher_or_model: Any) -> Any:
    """Unwrap a ComfyUI ModelPatcher to get the raw nn.Module.

    Returns the inner model if wrapped in a ModelPatcher, otherwise
    returns the object unchanged.
    """
    try:
        import comfy.model_patcher
        if isinstance(patcher_or_model, comfy.model_patcher.ModelPatcher):
            return patcher_or_model.model
    except ImportError:
        pass
    return patcher_or_model


def get_cache_key(model_path: str, device: str, dtype: str, attention: str) -> tuple:
    """Generate a cache key from model configuration."""
    return (model_path, device, dtype, attention)


def get_cached_model() -> tuple[Any, tuple]:
    """Get the cached model and its key."""
    with _cache_lock:
        return _cached_model, _cached_key


def set_cached_model(model: Any, key: tuple, keep_loaded: bool = False) -> None:
    """Cache a model with its configuration key."""
    global _cached_model, _cached_key, _keep_loaded, _offloaded
    with _cache_lock:
        _cached_model = model
        _cached_key = key
        _keep_loaded = keep_loaded
        _offloaded = False


def set_keep_loaded(keep_loaded: bool) -> None:
    """Update the keep_loaded flag for the cached model."""
    global _keep_loaded
    with _cache_lock:
        _keep_loaded = keep_loaded


def is_offloaded() -> bool:
    """Check if the model is currently offloaded to CPU."""
    with _cache_lock:
        return _offloaded


# ---------------------------------------------------------------------------
# Internal helpers (callers must already hold _cache_lock)
# ---------------------------------------------------------------------------

def _unregister_from_comfy() -> None:
    """Remove the cached model from ComfyUI's current_loaded_models.

    Safe to call even if comfy.model_management is unavailable or the
    model was never registered.
    """
    if _cached_model is None:
        return
    try:
        import comfy.model_management as mm
        mm.current_loaded_models[:] = [
            lm for lm in mm.current_loaded_models
            if lm.model is not _cached_model
        ]
    except Exception:
        pass


def _was_evicted_by_comfy() -> bool:
    """Check if ComfyUI evicted our model to CPU without us knowing.

    Compares the device of the raw model's first parameter against the
    ModelPatcher's load_device.  Returns True if the model was moved
    to a different device (e.g. by ComfyUI's VRAM eviction).
    """
    if _cached_model is None:
        return False
    try:
        raw = get_raw_model(_cached_model)
        param = next(raw.parameters(), None)
        if param is None:
            return False
        return param.device != _cached_model.load_device
    except Exception:
        return False


class OmniVoiceVBar:
    """VBar implementation for aimdo dynamic VRAM visualization.

    Reports per-page residency for the OmniVoice model.  Since OmniVoice
    is always fully on GPU or fully on CPU, all pages move together.
    """

    page_size: int = 32 * 1024 * 1024  # 32 MB per page
    offset: int = 0

    def __init__(self, model: Any, device: torch.device):
        self._model = model
        # Normalize device type so torch.device("mps") and
        # torch.device("mps:0") compare equal
        self._device = torch.device(device.type)
        self._total_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        self._total_pages = max(1, math.ceil(self._total_size / self.page_size))
        self._watermark: int = 0

    def loaded_size(self) -> int:
        """Bytes currently in VRAM."""
        try:
            param = next(self._model.parameters(), None)
            if param is None:
                return 0
            if torch.device(param.device.type) == self._device:
                return self._total_size
        except Exception:
            pass
        return 0

    def get_residency(self) -> list[int]:
        """Per-page flags.  bit 1 = resident, bit 2 = pinned."""
        loaded = self.loaded_size()
        resident_pages = min(
            int(loaded // self.page_size), self._total_pages
        )
        return [1 if i < resident_pages else 0 for i in range(self._total_pages)]

    def get_watermark(self) -> int:
        """Current high-watermark."""
        current = self.loaded_size()
        self._watermark = max(self._watermark, current)
        return self._watermark

    def prioritize(self):
        """Reset watermark (triggered by wm button in viz panel)."""
        self._watermark = self.loaded_size()


def _register_with_comfy() -> None:
    """Register the cached ModelPatcher with ComfyUI's VRAM tracking.

    Creates a LoadedModel and adds it to current_loaded_models WITHOUT
    calling load_model_gpu, which would patch model weights (designed
    for ComfyUI diffusion models, not HuggingFace models like OmniVoice).
    Also attaches aimdo dynamic VBar for viz panel residency tracking.
    Skips registration for CPU-only models (no VRAM to manage).
    """
    if _cached_model is None:
        return

    load_device = _cached_model.load_device

    # Skip registration for CPU models — no VRAM to manage
    if load_device.type == "cpu":
        return

    try:
        import comfy.model_management as mm
        import weakref

        # Avoid double registration
        if any(lm.model is _cached_model for lm in mm.current_loaded_models):
            return

        # Create LoadedModel manually — no weight patching
        loaded = mm.LoadedModel(_cached_model)
        raw = get_raw_model(_cached_model)

        # Tell ComfyUI the full model is loaded (set on the inner model
        # since ModelPatcher.loaded_size() reads from self.model)
        model_size = _cached_model.model_size()
        raw.model_loaded_weight_memory = model_size

        # Attach aimdo dynamic VBar for viz panel
        raw.dynamic_vbars = {
            load_device: OmniVoiceVBar(raw, load_device),
        }

        loaded.real_model = weakref.ref(raw)
        loaded.model_finalizer = weakref.finalize(raw, mm.cleanup_models)
        loaded.model_finalizer.atexit = False
        loaded.currently_used = True

        mm.current_loaded_models.insert(0, loaded)
        logger.info(
            f"Model registered with ComfyUI VRAM management "
            f"({model_size / (1024 * 1024):.1f} MB)."
        )
    except Exception as e:
        logger.warning(f"Could not register with ComfyUI VRAM management: {e}")


def _do_unload() -> None:
    """Unload logic without lock acquisition. Caller must hold _cache_lock."""
    global _cached_model, _cached_key, _keep_loaded, _offloaded
    if _cached_model is None:
        return
    logger.info("Unloading OmniVoice model from memory...")
    _unregister_from_comfy()
    try:
        get_raw_model(_cached_model).to("cpu")
    except Exception:
        pass
    del _cached_model
    _cached_model = None
    _cached_key = ()
    _keep_loaded = False
    _offloaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()
    logger.info("Model unloaded and VRAM freed.")


def _do_resume(device: str) -> None:
    """Resume an offloaded or evicted model to device.

    Handles both explicit offload (via offload_model_to_cpu) and
    implicit eviction (by ComfyUI's VRAM management).  Caller must
    hold _cache_lock.
    """
    global _offloaded
    if _cached_model is None:
        return
    if not _offloaded and not _was_evicted_by_comfy():
        return
    get_raw_model(_cached_model).to(device)
    _offloaded = False
    # Re-register with ComfyUI (removed during eviction or never registered)
    _register_with_comfy()
    # Restore internal ASR pipeline device if attached
    raw = get_raw_model(_cached_model)
    asr = getattr(raw, "_asr_pipe", None)
    if asr is not None:
        _whisper_to_device(asr, device)
    logger.info(f"Model resumed to {device}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def offload_model_to_cpu() -> None:
    """Offload the cached model to CPU to free VRAM."""
    global _offloaded
    with _cache_lock:
        if _cached_model is None:
            return
        if _offloaded:
            return

        # Unregister from ComfyUI so it stops tracking our VRAM usage
        _unregister_from_comfy()

        try:
            get_raw_model(_cached_model).to("cpu")
            _offloaded = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            gc.collect()
            logger.info("Model offloaded to CPU. VRAM freed.")
        except Exception as e:
            logger.warning(f"Failed to offload model: {e}")


def resume_model_to_device(device: str = "cuda") -> None:
    """Resume an offloaded model back to the target device.

    Also restores the device of any internal ASR pipeline (``_asr_pipe``)
    attached to the model so that input tensors and weights stay on the
    same device.
    """
    with _cache_lock:
        _do_resume(device)


# Backward-compatible alias for any external callers
resume_model_to_cuda = resume_model_to_device


def unload_model() -> None:
    """Fully unload the model from memory."""
    with _cache_lock:
        _do_unload()


# ---------------------------------------------------------------------------
# Shared model loader (single point of truth for all nodes)
# ---------------------------------------------------------------------------

def get_or_load_model(
    model_name: str,
    device: str,
    dtype: str,
    attention: str,
    keep_loaded: bool = False,
) -> tuple[Any, None]:
    """Get or load the OmniVoice model with thread-safe caching.

    Atomically checks the cache key and unloads the old model if settings
    changed — no TOCTOU gap between check and unload.  The heavy
    ``load_model()`` call runs outside the lock so concurrent requests
    aren't blocked during download / GPU allocation.

    Returns:
        (raw_model, None) — the raw OmniVoice nn.Module ready for
        inference.  The ModelPatcher wrapper is stored internally.
        The second element is kept for API compatibility.
    """
    global _cached_model, _cached_key, _keep_loaded, _offloaded

    from .loader import load_model, resolve_device

    key = get_cache_key(model_name, device, dtype, attention)
    device_str, _ = resolve_device(device)

    # ---- Fast path / unload-under-lock ----
    with _cache_lock:
        if _cached_model is not None and _cached_key == key:
            _keep_loaded = keep_loaded
            if _offloaded or _was_evicted_by_comfy():
                _do_resume(device_str)
                logger.info(f"Resumed offloaded/evicted model to {device_str}.")
            else:
                logger.info("Reusing cached OmniVoice model.")
            return get_raw_model(_cached_model), None

        if _cached_model is not None:
            logger.info(
                "Settings changed (model/device/dtype/attention) — "
                "unloading cached model."
            )
            _do_unload()

    # ---- Slow path: load outside lock ----
    omnivoice_patcher, _ = load_model(model_name, device, dtype, attention)

    # ---- Store result under lock ----
    with _cache_lock:
        # Another thread may have loaded the same key while we waited
        if _cached_model is not None and _cached_key == key:
            logger.info("Another thread loaded the same model — using cached version.")
            _keep_loaded = keep_loaded
            # Discard the duplicate we just loaded
            try:
                get_raw_model(omnivoice_patcher).to("cpu")
            except Exception:
                pass
            del omnivoice_patcher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            gc.collect()
            return get_raw_model(_cached_model), None

        if _cached_model is not None:
            _do_unload()

        _cached_model = omnivoice_patcher
        _cached_key = key
        _keep_loaded = keep_loaded
        _offloaded = False

    # Register with ComfyUI model management so VRAM is tracked and
    # other models can evict ours when memory is tight.
    _register_with_comfy()

    return get_raw_model(omnivoice_patcher), None


# ---------------------------------------------------------------------------
# Whisper helpers
# ---------------------------------------------------------------------------

def _whisper_to_device(pipe: Any, device: str) -> None:
    """Move a HuggingFace pipeline to the target device.

    Strategy: move the underlying model directly (most reliable across
    transformers versions), then fall back to pipeline-level .to().
    """
    for attr in ("model", "_model"):
        m = getattr(pipe, attr, None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
                return
            except Exception:
                pass
    # Fallback: pipeline-level .to()
    try:
        pipe.to(device)
    except Exception:
        pass


def _whisper_to_cpu(pipe: Any) -> None:
    """Move a HuggingFace pipeline's model to CPU."""
    _whisper_to_device(pipe, "cpu")


def unload_whisper() -> None:
    """Fully unload the cached Whisper model from memory."""
    global _cached_whisper, _cached_whisper_key
    with _whisper_lock:
        if _cached_whisper is not None:
            logger.info("Unloading Whisper ASR model from memory...")
            _whisper_to_cpu(_cached_whisper)
            del _cached_whisper
            _cached_whisper = None
            _cached_whisper_key = ()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            gc.collect()
            logger.info("Whisper ASR model unloaded and VRAM freed.")


def offload_whisper_to_cpu() -> None:
    """Offload the cached Whisper model to CPU to free VRAM."""
    with _whisper_lock:
        if _cached_whisper is None:
            return
        try:
            _whisper_to_cpu(_cached_whisper)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            gc.collect()
            logger.info("Whisper ASR model offloaded to CPU. VRAM freed.")
        except Exception as e:
            logger.warning(f"Failed to offload Whisper model: {e}")


def get_or_cache_whisper(whisper_input: dict | None, model_name: str, device: str, dtype: str) -> Any:
    """Get or load a Whisper pipeline with caching.

    Args:
        whisper_input: Whisper dict from node input (None if not connected).
            Must have "pipeline" and "model_name" keys.
        model_name: Model name from dropdown (for cache key when no input).
        device: Device to load on.
        dtype: Model precision.

    Returns:
        Whisper pipeline or None.
    """
    if whisper_input is None:
        return None

    key = (whisper_input.get("model_name", model_name), device, dtype)

    # Resolve target device so we can restore Whisper to it after CPU offload
    from .loader import resolve_device
    device_str, _ = resolve_device(device)

    global _cached_whisper, _cached_whisper_key
    with _whisper_lock:
        if _cached_whisper is not None and _cached_whisper_key == key:
            # Ensure Whisper is on the target device (may have been offloaded to CPU)
            _whisper_to_device(_cached_whisper, device_str)
            logger.info(f"Reusing cached Whisper ASR model (on {device_str}).")
            return _cached_whisper

        # Settings changed — unload old
        if _cached_whisper is not None:
            logger.info("Whisper settings changed — unloading old Whisper model.")
            _whisper_to_cpu(_cached_whisper)
            del _cached_whisper
            _cached_whisper = None
            _cached_whisper_key = ()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

        # Use the pipeline from the input (already loaded by Whisper Loader node).
        # IMPORTANT: The same pipeline object is shared with ComfyUI's node
        # output cache, so it may have been moved to CPU by a previous
        # unload_whisper() call.  Restore it to the target device now.
        pipe = whisper_input["pipeline"]
        _whisper_to_device(pipe, device_str)
        _cached_whisper = pipe
        _cached_whisper_key = key
        logger.info(f"Whisper ASR model cached (on {device_str}).")
        return _cached_whisper
