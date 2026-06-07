import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

_MODEL_ID = "Salesforce/blip-image-captioning-base"
_processor = None
_model = None


def _load():
    global _processor, _model
    if _processor is None:
        _processor = BlipProcessor.from_pretrained(_MODEL_ID)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_ID)


def caption(image: Image.Image) -> str:
    _load()
    inputs = _processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=50)
    return _processor.decode(out[0], skip_special_tokens=True)
