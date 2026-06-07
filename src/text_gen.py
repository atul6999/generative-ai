import os

from transformers import pipeline
from openai import OpenAI

_local_pipe = None


def _load_local(model="gpt2"):
    global _local_pipe
    if _local_pipe is None:
        _local_pipe = pipeline("text-generation", model=model)


def generate_local(prompt: str, max_new_tokens=60, temperature=0.7) -> str:
    _load_local()
    result = _local_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
    )
    return result[0]["generated_text"]


def generate_hf_endpoint(
    prompt: str,
    model="meta-llama/Llama-3.1-8B-Instruct:novita",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    token = os.getenv("HF_TOKEN", "")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content
