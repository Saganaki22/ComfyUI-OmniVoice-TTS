"""LoRA loading utilities for OmniVoice inference nodes.

Applies trained LoRA adapters to the OmniVoice model at inference time.
LoRA weights are loaded from models/loras/ directory.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("OmniVoice")


def get_available_loras() -> list[str]:
    """List available OmniVoice LoRA adapters from models/loras/.

    Returns entries like:
      - "jimbo_lora" (top-level = latest checkpoint)
      - "jimbo_lora/step_250" (specific per-step checkpoint)
      - "jimbo_lora/step_500"
    """
    lora_entries = []
    try:
        import folder_paths
        lora_base = folder_paths.get_folder_paths("loras")[0]
    except (ImportError, IndexError):
        return lora_entries

    if not os.path.isdir(lora_base):
        return lora_entries

    for entry in sorted(os.listdir(lora_base)):
        full = os.path.join(lora_base, entry)
        if not os.path.isdir(full):
            continue

        # Top-level adapter (latest)
        if os.path.isfile(os.path.join(full, "adapter_config.json")):
            lora_entries.append(entry)

            # Scan for per-step checkpoints inside checkpoints/
            ckpt_dir = os.path.join(full, "checkpoints")
            if os.path.isdir(ckpt_dir):
                for step_dir in sorted(os.listdir(ckpt_dir)):
                    step_path = os.path.join(ckpt_dir, step_dir)
                    if os.path.isdir(step_path) and os.path.isfile(os.path.join(step_path, "adapter_config.json")):
                        lora_entries.append(f"{entry}/{step_dir}")

    return lora_entries


def resolve_lora_path(lora_name: str) -> Optional[str]:
    """Resolve a LoRA name to its directory path.

    Handles:
      - "jimbo_lora" -> models/loras/jimbo_lora/
      - "jimbo_lora/step_250" -> models/loras/jimbo_lora/checkpoints/step_250/
    """
    if not lora_name or lora_name == "None":
        return None

    try:
        import folder_paths
        lora_base = folder_paths.get_folder_paths("loras")[0]
    except (ImportError, IndexError):
        return None

    # Check if it's a "name/step_N" format
    if "/" in lora_name:
        parts = lora_name.split("/", 1)
        lora_dir = os.path.join(lora_base, parts[0], "checkpoints", parts[1])
    else:
        lora_dir = os.path.join(lora_base, lora_name)

    if os.path.isdir(lora_dir) and os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
        return lora_dir

    return None


def apply_lora(model, lora_path: str):
    """Apply a LoRA adapter to the OmniVoice model.

    Wraps model.llm with PEFT and loads audio layer weights if present.
    Does NOT merge weights — keeps LoRA as a wrapper so the base model
    can be restored after inference via remove_lora().

    Returns the original audio layer weights for restoration.

    Args:
        model: Raw OmniVoice model instance.
        lora_path: Path to the LoRA directory (containing adapter_config.json).

    Returns:
        Dict of original audio layer weights (for remove_lora), or None.
    """
    from peft import PeftModel

    # Apply PEFT LoRA to the LLM backbone
    logger.info(f"Loading LoRA adapter from {lora_path}...")
    model.llm = PeftModel.from_pretrained(model.llm, lora_path)

    # Save original audio layer weights before modifying them
    original_audio = {}
    audio_path = os.path.join(lora_path, "audio_layers.safetensors")
    if not os.path.isfile(audio_path):
        audio_path = os.path.join(lora_path, "audio_layers.bin")

    if os.path.isfile(audio_path):
        # Save originals first
        for name, param in model.named_parameters():
            if "llm" not in name:
                original_audio[name] = param.data.clone()

        logger.info("Loading audio layer weights...")
        if audio_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            audio_state = load_file(audio_path)
        else:
            audio_state = torch.load(audio_path, map_location="cpu")
            if "state_dict" in audio_state:
                audio_state = audio_state["state_dict"]

        # Apply audio layer weights
        model_state = model.state_dict()
        loaded = 0
        for key, value in audio_state.items():
            if key in model_state:
                model_state[key].copy_(value.to(model_state[key].device))
                loaded += 1
        logger.info(f"Loaded {loaded} audio layer parameters.")

    logger.info("LoRA adapter applied.")
    return original_audio if original_audio else None


def remove_lora(model, original_audio=None):
    """Remove the LoRA adapter and restore the base model.

    Undoes apply_lora() — restores model.llm to the original base LLM
    and restores original audio layer weights.

    Args:
        model: OmniVoice model with LoRA applied.
        original_audio: Dict of original audio weights from apply_lora().
    """
    try:
        from peft import PeftModel
        if isinstance(model.llm, PeftModel):
            base_model = model.llm.unload()
            # Clean up residual PEFT state that unload() may leave behind
            for attr in ("peft_config", "base_model"):
                if hasattr(base_model, attr):
                    try:
                        delattr(base_model, attr)
                    except (AttributeError, TypeError):
                        pass
            model.llm = base_model
    except ImportError:
        pass

    # Restore original audio layer weights
    if original_audio:
        model_state = model.state_dict()
        for key, value in original_audio.items():
            if key in model_state:
                model_state[key].copy_(value.to(model_state[key].device))
        # Free cloned tensors
        original_audio.clear()

    logger.info("LoRA adapter removed, base model restored.")
