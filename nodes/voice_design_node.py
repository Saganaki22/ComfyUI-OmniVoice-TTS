"""OmniVoice Voice Design TTS - Text-to-speech with voice design from text description.

This node creates synthetic voices from text descriptions (voice attributes)
and synthesizes speech in that designed voice. No reference audio needed.
"""

import logging
from typing import Tuple

import numpy as np
import torch

from .loader import (
    get_model_names,
    numpy_audio_to_comfy,
    to_numpy_audio,
    manual_seed_all,
)
from .model_cache import (
    cancel_event,
    get_or_load_model,
    offload_model_to_cpu,
    unload_model,
)

try:
    from comfy.utils import ProgressBar
    _PBAR = True
except ImportError:
    _PBAR = False

try:
    import comfy.model_management as mm
    _MM = True
except ImportError:
    _MM = False

logger = logging.getLogger("OmniVoice")

# OmniVoice outputs at 24kHz
OMNIVOICE_SAMPLE_RATE = 24000

# Voice design attribute hints
VOICE_DESIGN_HINT = (
    "Voice attributes (comma-separated): "
    "gender (male/female), age (child/young/elderly), "
    "pitch (very low/low/medium/high/very high), "
    "style (whisper), "
    "accent (american/british/australian/sichuan/shaanxi/etc.). "
    "Example: 'female, low pitch, british accent'"
)


class OmniVoiceVoiceDesignTTS:
    """OmniVoice Voice Design TTS node."""

    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names()
        return {
            "required": {
                "model": (
                    model_names,
                    {
                        "tooltip": (
                            "OmniVoice model checkpoint. "
                            "Models are stored in ComfyUI/models/omnivoice/"
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is a test of voice design with OmniVoice.",
                        "tooltip": (
                            "Text to synthesize in the designed voice. "
                            "Supports inline non-verbal tags like [laughter], [sigh], etc."
                        ),
                    },
                ),
                "voice_instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "female, low pitch, british accent",
                        "tooltip": VOICE_DESIGN_HINT,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of diffusion steps. "
                            "16 = faster, 32 = balanced, 64 = best quality."
                        ),
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale. Higher = more aligned with text.",
                    },
                ),
                "t_shift": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Time-step shift for noise schedule. Smaller = emphasis on earlier steps.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Speaking speed factor. >1.0 = faster, <1.0 = slower.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": (
                            "Fixed output duration in seconds. "
                            "0 = automatic. Overrides speed if set."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps", "xpu"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device. 'auto' picks CUDA > MPS > XPU > CPU.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model precision. 'auto' picks bf16 for CUDA (Ampere+), "
                            "fp16 for older CUDA/MPS, fp32 for CPU."
                        ),
                    },
                ),
                "attention": (
                    ["auto", "eager", "sage_attention"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention implementation. "
                            "'auto' uses model default (eager). "
                            "'sage_attention' uses SageAttention CUDA kernels (requires SM80+ GPU)."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    },
                ),
                "position_temperature": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Temperature for mask-position selection. 0 = greedy, higher = more random.",
                    },
                ),
                "class_temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Temperature for token sampling. 0 = greedy, higher = more random.",
                    },
                ),
                "layer_penalty_factor": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Penalty on deeper codebook layers, encouraging lower layers to unmask first.",
                    },
                ),
                "denoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prepend denoise token to input for cleaner output.",
                    },
                ),
                "postprocess_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Post-process generated audio (remove long silences).",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice"
    DESCRIPTION = (
        "OmniVoice Voice Design TTS - Design synthetic voices from text descriptions. "
        "Control gender, age, pitch, accent, and more. No reference audio needed."
    )

    def generate(
        self,
        model: str,
        text: str,
        voice_instruct: str,
        steps: int,
        guidance_scale: float,
        t_shift: float,
        speed: float,
        duration: float,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        position_temperature: float,
        class_temperature: float,
        layer_penalty_factor: float,
        denoise: bool,
        postprocess_output: bool,
        keep_model_loaded: bool,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        if not voice_instruct.strip():
            logger.warning(
                "No voice instruction provided. A random voice will be used. "
                "Consider adding attributes like 'female, low pitch, british accent'."
            )

        # Load or get cached model
        omnivoice_model, _ = get_or_load_model(
            model, device, dtype, attention, keep_model_loaded
        )

        pbar = ProgressBar(3) if _PBAR else None

        # Log what we're generating
        logger.info(f"Voice Design TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        logger.info(f"Voice attributes: {voice_instruct}")

        if pbar:
            pbar.update_absolute(1, 3)

        # Set random seed
        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        manual_seed_all(actual_seed)

        self._check_interrupt()

        result = None
        try:
            # Build kwargs for generate
            gen_kwargs = {
                "text": text,
                "instruct": voice_instruct,
                "num_step": steps,
                "guidance_scale": guidance_scale,
                "t_shift": t_shift,
                "speed": speed,
                "position_temperature": position_temperature,
                "class_temperature": class_temperature,
                "layer_penalty_factor": layer_penalty_factor,
                "denoise": denoise,
                "postprocess_output": postprocess_output,
            }
            if duration > 0:
                gen_kwargs["duration"] = duration

            # Generate audio with voice design
            with torch.inference_mode():
                audio_list = omnivoice_model.generate(**gen_kwargs)

            if pbar:
                pbar.update_absolute(2, 3)

            # Convert to ComfyUI format
            audio_np = to_numpy_audio(audio_list[0])

            result = numpy_audio_to_comfy(audio_np, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_np) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz with designed voice"
            )

            if pbar:
                pbar.update_absolute(3, 3)

        finally:
            if not keep_model_loaded:
                unload_model()
            else:
                offload_model_to_cpu()

        if result is None:
            raise RuntimeError("Generation failed — see logs above.")
        return (result,)

    def _check_interrupt(self):
        """Check if processing was interrupted."""
        if _MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                cancel_event.set()
                raise


# Voice attribute reference for documentation
VOICE_ATTRIBUTES = {
    "gender": ["male", "female"],
    "age": ["child", "young", "middle-aged", "elderly"],
    "pitch": ["very low", "low", "medium", "high", "very high"],
    "style": ["whisper"],
    "english_accents": [
        "american accent", "british accent", "australian accent",
        "canadian accent", "indian accent", "irish accent",
        "scottish accent", "south african accent",
    ],
    "chinese_dialects": [
        "四川话", "陕西话", "广东话", "东北话", "山东话",
        "河南话", "上海话", "闽南话", "客家话",
    ],
}

NON_VERBAL_TAGS = [
    "[laughter]",
    "[confirmation-en]",
    "[question-en]", "[question-ah]", "[question-oh]",
    "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
    "[sniff]",
    "[sigh]",
]
