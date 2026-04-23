"""OmniVoice TTS nodes for ComfyUI."""

from .omnivoice_tts import OmniVoiceLongformTTS
from .voice_clone_node import OmniVoiceVoiceCloneTTS
from .voice_design_node import OmniVoiceVoiceDesignTTS
from .multi_speaker_node import OmniVoiceMultiSpeakerTTS

# Training nodes — lazy loaded, optional deps
OmniVoice_TrainConfig = None
OmniVoice_DatasetMaker = None
OmniVoice_LoRATrainer = None

try:
    from .train_config_node import OmniVoice_TrainConfig
    from .dataset_maker_node import OmniVoice_DatasetMaker
    from .lora_trainer_node import OmniVoice_LoRATrainer
except Exception as e:
    import logging
    logging.getLogger("OmniVoice").warning(f"Training nodes not loaded: {e}")

__all__ = [
    "OmniVoiceLongformTTS",
    "OmniVoiceVoiceCloneTTS",
    "OmniVoiceVoiceDesignTTS",
    "OmniVoiceMultiSpeakerTTS",
    "OmniVoice_TrainConfig",
    "OmniVoice_DatasetMaker",
    "OmniVoice_LoRATrainer",
]
