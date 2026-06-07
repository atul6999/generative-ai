from PIL import Image
from transformers import pipeline

_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("visual-question-answering", model="Salesforce/blip-vqa-capfilt-large")


def answer(image: Image.Image, question: str, top_k: int = 3) -> list[dict]:
    _load()
    return _pipe(image.convert("RGB"), question, top_k=top_k)
