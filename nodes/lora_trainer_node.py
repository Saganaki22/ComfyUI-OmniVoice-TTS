"""OmniVoice LoRA Trainer node.

Runs the full LoRA fine-tuning pipeline:
  - Audio tokenization (HiggsAudio v2)
  - Model loading + PEFT LoRA application
  - Training loop with data augmentation
  - Checkpoint saving

This node blocks the ComfyUI UI during training. Use the ComfyUI
interrupt button to cancel.
"""

import logging
import os

logger = logging.getLogger("OmniVoice")

# Lazy import — training deps may not be installed
TRAINING_IMPORT_ERROR = None
run_lora_training = None

try:
    from ..modules.omni_trainer import run_lora_training
except ImportError as e:
    run_lora_training = None
    TRAINING_IMPORT_ERROR = str(e)
    # Check specific missing packages
    missing = []
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import safetensors
    except ImportError:
        missing.append("safetensors")
    if missing:
        TRAINING_IMPORT_ERROR = (
            f"Missing packages for training: {', '.join(missing)}. "
            f"Run: pip install {' '.join(missing)}"
        )


class OmniVoice_LoRATrainer:
    """Trains a LoRA adapter for OmniVoice TTS.

    WARNING: This process takes time and blocks the ComfyUI UI.
    Use the interrupt button to cancel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["OmniVoice", "OmniVoice-bf16"],
                    {
                        "tooltip": "Base OmniVoice model to fine-tune.",
                    },
                ),
                "train_config": ("OMNIVOICE_TRAIN_CONFIG", {
                    "tooltip": "Configuration from OmniVoice Train Config node.",
                }),
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the JSONL manifest from OmniVoice Dataset Maker.",
                }),
                "output_name": ("STRING", {
                    "default": "omnivoice_lora_v1",
                    "tooltip": "Name of the subfolder in models/loras to save LoRA outputs.",
                }),
                "max_steps": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 100000,
                    "tooltip": "Total number of training steps.",
                }),
                "save_every_steps": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 5000,
                    "tooltip": "Save a checkpoint every N steps.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "execute"
    CATEGORY = "audio/tts/training"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Train a LoRA adapter for OmniVoice. "
        "WARNING: blocks the UI during training."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, model_name, **kwargs):
        # Populate model dropdown dynamically
        return True

    def execute(
        self,
        model_name,
        train_config,
        dataset_path,
        output_name,
        max_steps,
        save_every_steps,
    ):
        if run_lora_training is None:
            raise RuntimeError(
                f"Training unavailable. {TRAINING_IMPORT_ERROR}"
            )

        if not dataset_path or not os.path.exists(dataset_path):
            raise ValueError(
                f"Dataset path not found: {dataset_path}. "
                f"Use the Dataset Maker node to create a manifest first."
            )

        # Resolve output directory
        try:
            import folder_paths
            lora_base = folder_paths.get_folder_paths("loras")[0]
        except (ImportError, IndexError):
            lora_base = os.path.join(os.path.dirname(__file__), "..", "..", "models", "loras")

        output_dir = os.path.join(lora_base, output_name)

        logger.info(f"Starting OmniVoice LoRA training...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Dataset: {dataset_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Steps: {max_steps}")

        final_dir = run_lora_training(
            model_name=model_name,
            train_config=train_config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            max_steps=max_steps,
            save_every_steps=save_every_steps,
        )

        return (final_dir,)
