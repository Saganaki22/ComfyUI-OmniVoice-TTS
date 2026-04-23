"""OmniVoice LoRA Training Configuration node.

Pure configuration node — returns a dict of hyperparameters.
No computation. Passes output to the LoRA Trainer node.
"""

import logging

logger = logging.getLogger("OmniVoice")


class OmniVoice_TrainConfig:
    """Configuration parameters for OmniVoice LoRA fine-tuning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 5e-5,
                        "min": 1e-6,
                        "max": 1e-2,
                        "step": 1e-5,
                        "tooltip": "Learning rate for the optimizer.",
                    },
                ),
                "lora_rank": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Rank (dimension) of the LoRA adapter.",
                    },
                ),
                "lora_alpha": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Alpha scaling factor for LoRA.",
                    },
                ),
                "lora_dropout": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": "Dropout probability for LoRA layers.",
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Number of warmup steps for the LR scheduler.",
                    },
                ),
                "grad_accum_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Gradient accumulation steps before weight update.",
                    },
                ),
                "weight_decay": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": "Weight decay for regularization.",
                    },
                ),
                "target_modules": (
                    "STRING",
                    {
                        "default": "q_proj,k_proj,v_proj,o_proj",
                        "tooltip": "Comma-separated list of target modules in the Qwen3 backbone for LoRA. "
                        "Options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
                    },
                ),
                "sequence_packing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Pack multiple short samples into one fixed-length sequence (up to batch_tokens). "
                        "Uses block attention masking so samples don't attend across boundaries. "
                        "Most efficient use of GPU compute. Recommended for best training results.",
                    },
                ),
                "batch_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 512,
                        "max": 16384,
                        "step": 512,
                        "tooltip": "Maximum total token length per packed sequence when Sequence Packing is enabled. "
                        "Short samples get concatenated up to this length. "
                        "4096 uses ~24GB VRAM. Lower to 2048 for less VRAM, raise to 8192 if you have headroom. "
                        "Ignored when Sequence Packing is disabled.",
                    },
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Compile the model with torch.compile() for faster training. "
                        "Strongly recommended when using Sequence Packing — without compile, "
                        "flex_attention materializes the full attention matrix which is very slow. "
                        "Adds a one-time compile overhead at training start (~30-60s). "
                        "Also speeds up non-packed training by fusing optimizer ops.",
                    },
                ),
                "train_audio_layers": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Also train audio_embeddings and audio_heads (recommended for voice cloning). "
                        "These are small layers that handle the audio↔LLM interface.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("OMNIVOICE_TRAIN_CONFIG",)
    RETURN_NAMES = ("train_config",)
    FUNCTION = "execute"
    CATEGORY = "audio/tts/training"
    DESCRIPTION = "Configuration parameters for OmniVoice LoRA fine-tuning."

    def execute(self, **kwargs):
        return (kwargs,)
