import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32

_txt2img = None
_img2img = None


def _load_txt2img():
    global _txt2img
    if _txt2img is None:
        _txt2img = StableDiffusionPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=_dtype
        ).to(_device)


def _load_img2img():
    global _img2img
    if _img2img is None:
        _img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=_dtype
        ).to(_device)


def text_to_image(
    prompt: str, steps: int = 30, guidance_scale: float = 7.5
) -> Image.Image:
    _load_txt2img()
    return _txt2img(
        prompt, num_inference_steps=steps, guidance_scale=guidance_scale
    ).images[0]


def image_to_image(
    image: Image.Image,
    prompt: str,
    strength: float = 0.7,
    steps: int = 40,
    guidance_scale: float = 7.5,
) -> Image.Image:
    _load_img2img()
    init = image.convert("RGB").resize((512, 512))
    return _img2img(
        prompt=prompt,
        image=init,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
