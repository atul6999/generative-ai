from transformers import pipeline

_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("sentiment-analysis")


def analyze(texts: list[str]) -> list[dict]:
    _load()
    return _pipe(texts)
