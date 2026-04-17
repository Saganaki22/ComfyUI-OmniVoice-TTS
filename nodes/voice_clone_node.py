"""OmniVoice Voice Clone TTS - Text-to-speech with voice cloning from reference audio.

This node clones a voice from a reference audio sample and synthesizes new speech
in that voice. Supports 600+ languages with high-quality zero-shot voice cloning.
"""

import logging
from typing import Tuple

import torch

from .loader import (
    get_model_names,
    numpy_audio_to_comfy,
    to_numpy_audio,
    comfy_audio_to_numpy,
    manual_seed_all,
)
from .model_cache import (
    cancel_event,
    get_or_cache_whisper,
    get_or_load_model,
    offload_model_to_cpu,
    offload_whisper_to_cpu,
    unload_model,
    unload_whisper,
)
from .whisper_loader import find_local_whisper_model, load_whisper_pipeline

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


class OmniVoiceVoiceCloneTTS:
    """OmniVoice Voice Clone TTS node."""

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
                        "default": "Hello! This is a test of voice cloning with OmniVoice.",
                        "tooltip": (
                            "Text to synthesize in the cloned voice. "
                            "Supports inline non-verbal tags like [laughter], [sigh], etc."
                        ),
                    },
                ),
                "ref_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Reference audio to clone voice from. "
                            "3-15 seconds of clear speech works best. "
                            "Will be resampled to 24kHz if needed."
                        ),
                    },
                ),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Transcript of the reference audio. "
                            "Leave empty to auto-transcribe with Whisper ASR. "
                            "Providing the transcript improves quality."
                        ),
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
                            "0 = automatic (uses speed). Overrides speed if set."
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
                "preprocess_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Preprocess voice-clone prompt audio (remove silences, add punctuation).",
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
                "instruct": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": (
                            "Dialect/style instruction from the model's supported values. "
                            "English: american/british/australian/canadian/chinese/indian/"
                            "japanese/korean/portuguese/russian accent, "
                            "male/female, child/young adult/teenager/middle-aged/elderly, "
                            "very low pitch/low pitch/moderate pitch/high pitch/very high pitch, whisper. "
                            "Chinese: 四川话/东北话/陕西话/河南话/云南话/贵州话/甘肃话/"
                            "宁夏话/石家庄话/济南话/青岛话/桂林话, "
                            "男/女, 儿童/少年/青年/中年/老年, "
                            "极低音调/低音调/中音调/高音调/极高音调, 耳语. "
                            "Use comma-separated (English) or full-width comma (Chinese). "
                            "Leave empty for default."
                        ),
                    },
                ),
            },
            "optional": {
                "whisper_model": (
                    "WHISPER_ASR",
                    {
                        "tooltip": (
                            "Optional pre-loaded Whisper ASR model. "
                            "Connect from OmniVoice Whisper Loader to avoid "
                            "re-downloading on each run."
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
        "OmniVoice Voice Clone TTS - Clone a voice from reference audio and "
        "synthesize new speech. High-quality zero-shot voice cloning for 600+ languages."
    )

    def generate(
        self,
        model: str,
        text: str,
        ref_audio: dict,
        ref_text: str,
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
        preprocess_prompt: bool,
        postprocess_output: bool,
        keep_model_loaded: bool,
        instruct: str,
        whisper_model: dict = None,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        # Load or get cached model
        omnivoice_model, _ = get_or_load_model(
            model, device, dtype, attention, keep_model_loaded
        )

        pbar = ProgressBar(4) if _PBAR else None

        # Convert reference audio from ComfyUI format to numpy at 24kHz
        logger.info("Processing reference audio...")
        ref_audio_np, ref_sr = comfy_audio_to_numpy(ref_audio, target_sr=OMNIVOICE_SAMPLE_RATE)
        ref_audio_tensor = torch.from_numpy(ref_audio_np).float()
        ref_duration = len(ref_audio_np) / OMNIVOICE_SAMPLE_RATE

        # Warn about reference audio length
        if ref_duration < 1:
            logger.warning(
                f"Reference audio is only {ref_duration:.1f}s — "
                "recommend 3-15s for best quality."
            )
        elif ref_duration > 30:
            logger.warning(
                f"Reference audio is {ref_duration:.1f}s — "
                "longer than recommended 15s may cause issues."
            )

        if pbar:
            pbar.update_absolute(1, 4)

        # Log what we're generating
        logger.info(f"Voice Clone TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        if ref_text.strip():
            logger.info(f"Reference transcript provided — bypassing Whisper ASR")
        elif whisper_model is not None:
            whisper_pipe = get_or_cache_whisper(whisper_model, model, device, dtype)
            if whisper_pipe is not None:
                logger.info("No reference transcript — using pre-loaded Whisper ASR")
                omnivoice_model._asr_pipe = whisper_pipe
        else:
            # No Whisper node connected — check for a locally downloaded model
            # before letting OmniVoice trigger its own download
            local_name = find_local_whisper_model()
            if local_name is not None:
                logger.info(
                    f"No reference transcript — auto-detected local Whisper "
                    f"({local_name}) for transcription"
                )
                try:
                    pipe = load_whisper_pipeline(local_name, device, dtype)
                    get_or_cache_whisper(
                        {"pipeline": pipe, "model_name": local_name},
                        model, device, dtype,
                    )
                    omnivoice_model._asr_pipe = pipe
                except Exception as e:
                    logger.warning(f"Failed to load local Whisper: {e}")
                    logger.info("No reference transcript — Whisper will auto-transcribe (will download if not cached)")
            else:
                logger.info("No reference transcript — Whisper will auto-transcribe (will download if not cached)")

        # Set random seed
        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        manual_seed_all(actual_seed)

        if pbar:
            pbar.update_absolute(2, 4)

        self._check_interrupt()

        result = None
        try:
            gen_kwargs = {
                "text": text,
                "num_step": steps,
                "guidance_scale": guidance_scale,
                "t_shift": t_shift,
                "speed": speed,
                "ref_audio": (ref_audio_tensor, OMNIVOICE_SAMPLE_RATE),
                "position_temperature": position_temperature,
                "class_temperature": class_temperature,
                "layer_penalty_factor": layer_penalty_factor,
                "denoise": denoise,
                "preprocess_prompt": preprocess_prompt,
                "postprocess_output": postprocess_output,
            }
            if ref_text.strip():
                gen_kwargs["ref_text"] = ref_text.strip()
            if instruct and instruct.strip():
                gen_kwargs["instruct"] = instruct.strip()
            if duration > 0:
                gen_kwargs["duration"] = duration

            with torch.no_grad():
                try:
                    audio_list = omnivoice_model.generate(**gen_kwargs)
                except ValueError as e:
                    if "instruct" in str(e).lower() or "unsupported" in str(e).lower():
                        raise RuntimeError(
                            f"Invalid instruct value '{gen_kwargs.get('instruct')}'. "
                            f"The model only accepts specific values. Original error:\n{e}"
                        ) from e
                    raise

            audio_np = to_numpy_audio(audio_list[0])
            result = numpy_audio_to_comfy(audio_np, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_np) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz in cloned voice"
            )

        finally:
            if not keep_model_loaded:
                unload_model()
                unload_whisper()
            else:
                offload_model_to_cpu()
                offload_whisper_to_cpu()

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
