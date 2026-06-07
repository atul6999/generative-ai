import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

_MODEL_ID = "openai/whisper-small"
_processor = None
_model = None


def _load():
    global _processor, _model
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(_MODEL_ID)


def transcribe(audio: tuple[int, np.ndarray]) -> str:
    """Accept (sample_rate, audio_array) as returned by gr.Audio(type='numpy')."""
    _load()
    sample_rate, audio_array = audio
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0:
        audio_array /= 32768.0

    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    inputs = _processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = _model.generate(inputs["input_features"])
    return _processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


