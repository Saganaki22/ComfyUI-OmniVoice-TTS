"""OmniVoice Multi-Speaker TTS - Generate dialogue with multiple cloned voices.

Uses [Speaker_N]: tags in text to assign lines to different speakers.
Each speaker needs reference audio connected.
Reference audio + text -> speech with voice cloning for multiple speakers.
Supports 600+ languages with zero-shot voice cloning.
"""

import logging
import re
from typing import Tuple

import numpy as np
import torch

from .loader import (
    get_model_names,
    numpy_audio_to_comfy,
    comfy_audio_to_numpy,
    to_numpy_audio,
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

try:
    from comfy_api.latest import IO
    _V3 = True
except ImportError:
    _V3 = False

logger = logging.getLogger("OmniVoice")

OMNIVOICE_SAMPLE_RATE = 24000
MAX_SPEAKERS = 10


def _parse_dialogue_lines(text: str):
    """Parse multi-speaker text into (speaker_idx, line_text) tuples.

    Recognizes:
        [Speaker_1]: Hello world
        [Speaker_2]: Hi there
        [speaker_3]: Nice to meet you

    Lines without speaker tags are dropped.
    """
    tag_re = re.compile(r'\[speaker_(\d+)\]:\s*(.*)', re.IGNORECASE)
    lines = text.strip().splitlines()
    turns = []
    current_speaker = None
    current_parts = []

    for raw in lines:
        m = tag_re.match(raw.strip())
        if m:
            # Flush previous turn
            if current_speaker is not None and current_parts:
                turns.append((current_speaker, " ".join(current_parts).strip()))
            # Start new turn - convert to 0-based index
            current_speaker = int(m.group(1)) - 1  # 0-based
            current_parts = [m.group(2)] if m.group(2).strip() else []
        else:
            stripped = raw.strip()
            if stripped and current_speaker is not None:
                current_parts.append(stripped)

    # Flush last turn
    if current_speaker is not None and current_parts:
        turns.append((current_speaker, " ".join(current_parts).strip()))
    return turns


# ---------------------------------------------------------------------------
# Helper — build the per-option input list for a given speaker count
# ---------------------------------------------------------------------------

def _speaker_inputs(count: int) -> list:
    """Return IO input descriptors for `count` speakers (1-indexed for UI)."""
    inputs = []
    for i in range(1, count + 1):
        inputs.append(
            IO.Audio.Input(
                f"speaker_{i}_audio",
                optional=True,
                tooltip=(
                    f"Reference audio for speaker {i}. "
                    f"Use [speaker_{i}]: in your text for this voice."
                ),
            )
        )
        inputs.append(
            IO.String.Input(
                f"speaker_{i}_ref_text",
                multiline=False,
                default="",
                optional=True,
                tooltip=(
                    f"Transcript of speaker {i}'s reference audio. "
                    "Leave empty to auto-transcribe with Whisper."
                ),
            )
        )
        inputs.append(
            IO.String.Input(
                f"speaker_{i}_instruct",
                multiline=False,
                default="",
                optional=True,
                tooltip=(
                    f"Dialect/style instruction for speaker {i} from the model's supported values. "
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
            )
        )
    return inputs


def _auto_load_whisper(omnivoice_model, model_name: str, device: str, dtype: str) -> None:
    """Load a locally available Whisper model if the model doesn't already
    have an ASR pipeline set.  Avoids OmniVoice triggering its own download.
    """
    if getattr(omnivoice_model, "_asr_pipe", None) is not None:
        return  # Already has a pipeline

    local_name = find_local_whisper_model()
    if local_name is None:
        return

    logger.info(
        f"Auto-detected local Whisper model ({local_name}) — "
        "loading for auto-transcription"
    )
    try:
        pipe = load_whisper_pipeline(local_name, device, dtype)
        # Cache it so offload/resume lifecycle is handled properly
        get_or_cache_whisper(
            {"pipeline": pipe, "model_name": local_name},
            model_name, device, dtype,
        )
        omnivoice_model._asr_pipe = pipe
    except Exception as e:
        logger.warning(f"Failed to auto-load local Whisper: {e}")


# ---------------------------------------------------------------------------
# V3 node (DynamicCombo — inputs update when num_speakers changes)
# ---------------------------------------------------------------------------

if _V3:
    class OmniVoiceMultiSpeakerTTS(IO.ComfyNode):
        """
        OmniVoice Multi-Speaker TTS.
        Generates a conversation with multiple cloned voices in one pass.
        Change num_speakers to show/hide speaker reference audio inputs.
        Use [Speaker_N]: tags in text.
        """

        @classmethod
        def define_schema(cls) -> IO.Schema:
            model_names = get_model_names()

            # One DynamicCombo option per speaker count (2..MAX_SPEAKERS)
            speaker_options = [
                IO.DynamicCombo.Option(
                    key=str(n),
                    inputs=_speaker_inputs(n),
                )
                for n in range(2, MAX_SPEAKERS + 1)
            ]

            return IO.Schema(
                node_id="OmniVoiceMultiSpeakerTTS",
                display_name="OmniVoice Multi-Speaker TTS",
                category="OmniVoice",
                description=(
                    "OmniVoice Multi-Speaker TTS. Generates a "
                    "conversation between multiple cloned voices. "
                    "Connect reference audio clips and use "
                    "[Speaker_N]: tags in text."
                ),
                inputs=[
                    IO.Combo.Input(
                        "model",
                        options=model_names,
                        tooltip=(
                            "OmniVoice model checkpoint. "
                            "Models are stored in ComfyUI/models/omnivoice/"
                        ),
                    ),
                    IO.String.Input(
                        "text",
                        multiline=True,
                        default=(
                            "[Speaker_1]: Hello, I'm speaker one.\n"
                            "[Speaker_2]: And I'm speaker two!"
                        ),
                        tooltip=(
                            "Multi-speaker text. Use [Speaker_N]: to assign lines to "
                            "each connected speaker. Supports inline tags: "
                            "[laughter], [sigh], etc."
                        ),
                    ),
                    IO.Int.Input(
                        "steps",
                        default=32, min=4, max=64, step=1,
                        tooltip="Diffusion steps per speaker.",
                    ),
                    IO.Float.Input(
                        "guidance_scale",
                        default=2.0, min=0.0, max=10.0, step=0.1,
                        tooltip="Classifier-free guidance scale.",
                    ),
                    IO.Float.Input(
                        "t_shift",
                        default=0.1, min=0.0, max=1.0, step=0.01,
                        tooltip="Time-step shift for noise schedule.",
                    ),
                    IO.Float.Input(
                        "speed",
                        default=1.0, min=0.5, max=2.0, step=0.1,
                        tooltip="Speaking speed for all speakers.",
                    ),
                    IO.Float.Input(
                        "pause_between_speakers",
                        default=0.3, min=0.0, max=2.0, step=0.1,
                        tooltip="Seconds of silence between speakers.",
                    ),
                    IO.Combo.Input(
                        "device",
                        options=["auto", "cuda", "cpu", "mps"],
                        tooltip="Compute device.",
                    ),
                    IO.Combo.Input(
                        "dtype",
                        options=["auto", "bf16", "fp16", "fp32"],
                        tooltip="Model precision.",
                    ),
                    IO.Combo.Input(
                        "attention",
                        options=["auto", "eager", "sage_attention"],
                        tooltip="Attention implementation.",
                    ),
                    IO.Float.Input(
                        "position_temperature",
                        default=5.0, min=0.0, max=20.0, step=0.5,
                        tooltip="Temperature for mask-position selection. 0 = greedy.",
                    ),
                    IO.Float.Input(
                        "class_temperature",
                        default=0.0, min=0.0, max=5.0, step=0.1,
                        tooltip="Temperature for token sampling. 0 = greedy.",
                    ),
                    IO.Float.Input(
                        "layer_penalty_factor",
                        default=5.0, min=0.0, max=20.0, step=0.5,
                        tooltip="Penalty on deeper codebook layers.",
                    ),
                    IO.Boolean.Input(
                        "denoise",
                        default=True,
                        tooltip="Prepend denoise token for cleaner output.",
                    ),
                    IO.Boolean.Input(
                        "preprocess_prompt",
                        default=True,
                        tooltip="Preprocess reference audio (remove silences).",
                    ),
                    IO.Boolean.Input(
                        "postprocess_output",
                        default=True,
                        tooltip="Post-process audio (remove long silences).",
                    ),
                    IO.Int.Input(
                        "seed",
                        default=0, min=0, max=2**31 - 1,
                        tooltip="Random seed. 0 = random.",
                    ),
                    IO.Boolean.Input(
                        "keep_model_loaded",
                        default=True,
                        tooltip="Keep model loaded between runs.",
                    ),
                    IO.DynamicCombo.Input(
                        "num_speakers",
                        options=speaker_options,
                        display_name="Number of Speakers",
                        tooltip=(
                            f"How many speakers (2-{MAX_SPEAKERS}). "
                            "Changing this shows/hides speaker audio inputs."
                        ),
                    ),
                ],
                outputs=[
                    IO.Audio.Output(display_name="audio"),
                ],
            )

        @classmethod
        def execute(
            cls,
            model: str,
            text: str,
            steps: int,
            guidance_scale: float,
            t_shift: float,
            speed: float,
            pause_between_speakers: float,
            device: str,
            dtype: str,
            attention: str,
            position_temperature: float,
            class_temperature: float,
            layer_penalty_factor: float,
            denoise: bool,
            preprocess_prompt: bool,
            postprocess_output: bool,
            seed: int,
            keep_model_loaded: bool,
            num_speakers: dict,
        ) -> IO.NodeOutput:
            cancel_event.clear()
            cls._check_interrupt()

            if not text.strip():
                raise ValueError("Text cannot be empty.")

            # num_speakers is a dict from DynamicCombo:
            n = int(num_speakers["num_speakers"])

            # Validate speakers have reference audio
            missing = []
            for i in range(1, n + 1):
                speaker_audio = num_speakers.get(f"speaker_{i}_audio")
                if speaker_audio is None:
                    missing.append(i)

            if missing:
                raise ValueError(
                    f"Missing reference audio for speakers: {missing}. "
                    "Please connect audio to each speaker input."
                )

            # Load model
            omnivoice_model, _ = get_or_load_model(
                model, device, dtype, attention, keep_model_loaded
            )

            # Parse dialogue
            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError(
                    "No speaker lines found. Use [Speaker_N]: text format"
                )

            logger.info(
                f"Multi-Speaker TTS ({n} speakers, {len(dialogue_lines)} lines)"
            )

            # Auto-detect local Whisper if any speaker needs transcription
            any_without_ref = any(
                not num_speakers.get(f"speaker_{i + 1}_ref_text", "").strip()
                for i in range(n)
            )
            if any_without_ref:
                _auto_load_whisper(omnivoice_model, model, device, dtype)

            # Set random seed
            actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
            manual_seed_all(actual_seed)

            total_steps = len(dialogue_lines) + 1
            pbar = ProgressBar(total_steps) if _PBAR else None
            audio_turns = []
            sample_rate = OMNIVOICE_SAMPLE_RATE
            result = None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    cls._check_interrupt()

                    # Check speaker index is valid
                    if speaker_idx < 0 or speaker_idx >= n:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker index {speaker_idx + 1} "
                            f"but only {n} speakers are connected."
                        )

                    # Get reference audio for this speaker
                    speaker_audio = num_speakers.get(f"speaker_{speaker_idx + 1}_audio")
                    speaker_ref_text = num_speakers.get(f"speaker_{speaker_idx + 1}_ref_text", "")

                    if speaker_audio is None:
                        raise ValueError(
                            f"No reference audio for speaker {speaker_idx + 1}"
                        )

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[Speaker_{speaker_idx + 1}]: {line_text[:50]}{'...' if len(line_text) > 50 else ''}"
                    )

                    # Convert reference audio to numpy at 24kHz
                    ref_audio_np, _ = comfy_audio_to_numpy(
                        speaker_audio,
                        target_sr=OMNIVOICE_SAMPLE_RATE
                    )

                    # Build kwargs for generate
                    ref_audio_tensor = torch.from_numpy(ref_audio_np).float()
                    gen_kwargs = {
                        "text": line_text,
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

                    # Only add ref_text if provided - otherwise OmniVoice uses its own Whisper
                    if speaker_ref_text.strip():
                        gen_kwargs["ref_text"] = speaker_ref_text.strip()

                    speaker_instruct = num_speakers.get(f"speaker_{speaker_idx + 1}_instruct", "")
                    if speaker_instruct and speaker_instruct.strip():
                        gen_kwargs["instruct"] = speaker_instruct.strip()

                    # Generate audio for this line
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
                    audio_turns.append(audio_np)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                # Concatenate all turns with optional silence
                if pause_between_speakers > 0:
                    silence_samples = int(pause_between_speakers * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                logger.info(
                    f"Generated {len(audio_out) / OMNIVOICE_SAMPLE_RATE:.2f}s of multi-speaker audio "
                    f"({n} speakers, {len(dialogue_lines)} lines)"
                )

                result = numpy_audio_to_comfy(audio_out, sample_rate)

            finally:
                if not keep_model_loaded:
                    unload_model()
                    unload_whisper()
                else:
                    offload_model_to_cpu()
                    offload_whisper_to_cpu()

            if result is None:
                raise RuntimeError("Generation failed — see logs above.")
            return IO.NodeOutput(result)

        @classmethod
        def _check_interrupt(cls):
            """Check if processing was interrupted."""
            if _MM:
                try:
                    mm.throw_exception_if_processing_interrupted()
                except Exception:
                    cancel_event.set()
                    raise


# ---------------------------------------------------------------------------
# V2 fallback (old INPUT_TYPES API) — used if ComfyUI < 0.8.1
# Keeps all 10 speaker slots always visible (original behaviour).
# ---------------------------------------------------------------------------

else:
    class OmniVoiceMultiSpeakerTTS:  # type: ignore[no-redef]
        """
        OmniVoice Multi-Speaker TTS (legacy fallback — upgrade ComfyUI
        to 0.8.1+ for dynamic speaker inputs).
        """

        @classmethod
        def INPUT_TYPES(cls):
            model_names = get_model_names()
            optional_inputs = {
                "whisper_model": ("WHISPER_ASR", {
                    "tooltip": (
                        "Optional pre-loaded Whisper ASR model for auto-transcription. "
                        "Only used if ref_text is empty for a speaker."
                    ),
                }),
            }
            for i in range(1, MAX_SPEAKERS + 1):
                optional_inputs[f"speaker_{i}_audio"] = ("AUDIO", {
                    "tooltip": (
                        f"Reference audio for speaker {i}. "
                        f"Use [speaker_{i}]: in text."
                    ),
                })
                optional_inputs[f"speaker_{i}_ref_text"] = ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": f"Transcript of speaker {i}'s reference audio. Leave empty to auto-transcribe.",
                })
                optional_inputs[f"speaker_{i}_instruct"] = (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": (
                            f"Dialect/style instruction for speaker {i} from the model's supported values. "
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
                )

            return {
                "required": {
                    "model": (model_names, {
                        "tooltip": (
                            "OmniVoice model checkpoint. "
                            "Models are stored in ComfyUI/models/omnivoice/"
                        ),
                    }),
                    "text": ("STRING", {
                        "multiline": True,
                        "default": (
                            "[Speaker_1]: Hello, I'm speaker one.\n"
                            "[Speaker_2]: And I'm speaker two!"
                        ),
                    }),
                    "num_speakers": ("INT", {
                        "default": 2, "min": 2, "max": MAX_SPEAKERS, "step": 1,
                        "tooltip": f"Number of active speakers (2-{MAX_SPEAKERS}).",
                    }),
                    "steps": ("INT", {
                        "default": 32, "min": 4, "max": 64, "step": 1,
                        "tooltip": "Diffusion steps per speaker.",
                    }),
                    "guidance_scale": ("FLOAT", {
                        "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                        "tooltip": "Classifier-free guidance scale.",
                    }),
                    "t_shift": ("FLOAT", {
                        "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Time-step shift for noise schedule.",
                    }),
                    "speed": ("FLOAT", {
                        "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                        "tooltip": "Speaking speed for all speakers.",
                    }),
                    "pause_between_speakers": ("FLOAT", {
                        "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1,
                        "tooltip": "Seconds of silence between speakers.",
                    }),
                    "device": (["auto", "cuda", "cpu", "mps", "xpu"], {"default": "auto"}),
                    "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                    "attention": (["auto", "eager", "sage_attention"], {"default": "auto"}),
                    "position_temperature": ("FLOAT", {
                        "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                        "tooltip": "Temperature for mask-position selection. 0 = greedy.",
                    }),
                    "class_temperature": ("FLOAT", {
                        "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                        "tooltip": "Temperature for token sampling. 0 = greedy.",
                    }),
                    "layer_penalty_factor": ("FLOAT", {
                        "default": 5.0, "min": 0.0, "max": 20.0, "step": 0.5,
                        "tooltip": "Penalty on deeper codebook layers.",
                    }),
                    "denoise": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Prepend denoise token for cleaner output.",
                    }),
                    "preprocess_prompt": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Preprocess reference audio.",
                    }),
                    "postprocess_output": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Post-process generated audio.",
                    }),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                    "keep_model_loaded": ("BOOLEAN", {"default": True}),
                },
                "optional": optional_inputs,
            }

        RETURN_TYPES = ("AUDIO",)
        RETURN_NAMES = ("audio",)
        FUNCTION = "generate"
        CATEGORY = "OmniVoice"
        DESCRIPTION = "OmniVoice Multi-Speaker TTS (legacy mode — upgrade ComfyUI for dynamic inputs). Non-verbal expression tags. Multi-speaker dialogue with cloned voices using [Speaker_N]: format."

        def generate(
            self,
            model, text, num_speakers, steps, guidance_scale, t_shift, speed,
            pause_between_speakers, device, dtype, attention,
            position_temperature, class_temperature, layer_penalty_factor,
            denoise, preprocess_prompt, postprocess_output,
            seed, keep_model_loaded, **kwargs
        ):
            cancel_event.clear()
            self._check_interrupt()

            whisper_model = kwargs.get("whisper_model")

            if not text.strip():
                raise ValueError("Text cannot be empty.")

            # Validate speakers have reference audio
            missing = []
            for i in range(1, num_speakers + 1):
                speaker_audio = kwargs.get(f"speaker_{i}_audio")
                if speaker_audio is None:
                    missing.append(i)

            if missing:
                raise ValueError(
                    f"Missing reference audio for speakers: {missing}. "
                    "Please connect audio in each speaker input."
                )

            # Load model
            omnivoice_model, _ = get_or_load_model(
                model, device, dtype, attention, keep_model_loaded
            )

            # Parse dialogue
            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError(
                    "No speaker lines found. Use [Speaker_N]: text format"
                )

            logger.info(
                f"Multi-Speaker TTS ({num_speakers} speakers, {len(dialogue_lines)} lines)"
            )

            # Auto-detect local Whisper if no Whisper node is connected
            # and any speaker needs auto-transcription
            any_without_ref = any(
                not kwargs.get(f"speaker_{i}_ref_text", "").strip()
                for i in range(1, num_speakers + 1)
            )
            if any_without_ref and whisper_model is None:
                _auto_load_whisper(omnivoice_model, model, device, dtype)

            # Set random seed
            actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
            manual_seed_all(actual_seed)

            total_steps = len(dialogue_lines) + 1
            pbar = ProgressBar(total_steps) if _PBAR else None
            audio_turns = []
            sample_rate = OMNIVOICE_SAMPLE_RATE

            # Track which speakers need Whisper ( no ref_text provided)
            speakers_need_whisper = set()
            result = None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    self._check_interrupt()

                    # Check speaker index is valid
                    if speaker_idx < 0 or speaker_idx >= num_speakers:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker index {speaker_idx + 1} "
                            f"but only {num_speakers} speakers are connected."
                        )

                    # Get reference audio for this speaker
                    speaker_audio = kwargs.get(f"speaker_{speaker_idx + 1}_audio")
                    speaker_ref_text = kwargs.get(f"speaker_{speaker_idx + 1}_ref_text", "")

                    if speaker_audio is None:
                        raise ValueError(
                            f"No reference audio for speaker {speaker_idx + 1}"
                        )

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[Speaker_{speaker_idx + 1}]: {line_text[:50]}{'...' if len(line_text) > 50 else ''}"
                    )

                    # Convert reference audio to numpy at 24kHz
                    ref_audio_np, _ = comfy_audio_to_numpy(
                        speaker_audio,
                        target_sr=OMNIVOICE_SAMPLE_RATE
                    )

                    # Build kwargs for generate
                    ref_audio_tensor = torch.from_numpy(ref_audio_np).float()
                    gen_kwargs = {
                        "text": line_text,
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

                    # Only add ref_text if provided - otherwise let OmniVoice use Whisper
                    if speaker_ref_text.strip():
                        gen_kwargs["ref_text"] = speaker_ref_text.strip()

                    speaker_instruct = kwargs.get(f"speaker_{speaker_idx + 1}_instruct", "")
                    if speaker_instruct and speaker_instruct.strip():
                        gen_kwargs["instruct"] = speaker_instruct.strip()

                    if not speaker_ref_text.strip() and whisper_model is not None:
                        # Pre-loaded Whisper available - inject for this speaker
                        whisper_pipe = get_or_cache_whisper(whisper_model, model, device, dtype)
                        if whisper_pipe is not None:
                            omnivoice_model._asr_pipe = whisper_pipe
                            speakers_need_whisper.add(speaker_idx + 1)

                    # Generate audio for this line
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
                    audio_turns.append(audio_np)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                # Log Whisper usage
                if speakers_need_whisper:
                    logger.info(f"Used pre-loaded Whisper for speakers: {sorted(speakers_need_whisper)}")

                # Concatenate all turns with optional silence
                if pause_between_speakers > 0:
                    silence_samples = int(pause_between_speakers * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                logger.info(
                    f"Generated {len(audio_out) / OMNIVOICE_SAMPLE_RATE:.2f}s of multi-speaker audio "
                    f"({num_speakers} speakers, {len(dialogue_lines)} lines)"
                )

                result = numpy_audio_to_comfy(audio_out, sample_rate)

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
