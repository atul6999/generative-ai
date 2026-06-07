import sys
from pathlib import Path

# Allow imports from project root when running app/main.py directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def run_sentiment(text: str) -> str:
    from src.sentiment import analyze

    if not text.strip():
        return "Please enter some text."
    results = analyze([text])
    r = results[0]
    return f"{r['label']}  (confidence: {r['score']:.1%})"


def build_sentiment_tab():
    with gr.Tab("Sentiment Analysis"):
        gr.Markdown("Classify text as **POSITIVE** or **NEGATIVE**.")
        text_in = gr.Textbox(label="Text", placeholder="Enter text here…", lines=3)
        out = gr.Textbox(label="Result")
        gr.Button("Analyze").click(run_sentiment, inputs=text_in, outputs=out)


# ── Local Text Generation ─────────────────────────────────────────────────────

def run_text_gen(prompt: str, max_tokens: int, temperature: float) -> str:
    from src.text_gen import generate_local

    if not prompt.strip():
        return "Please enter a prompt."
    return generate_local(prompt, max_new_tokens=int(max_tokens), temperature=temperature)


def build_text_gen_tab():
    with gr.Tab("Text Generation (GPT-2)"):
        gr.Markdown("Generate text locally with GPT-2.")
        prompt = gr.Textbox(label="Prompt", placeholder="C++ is ", lines=2)
        with gr.Row():
            max_tokens = gr.Slider(10, 200, value=60, step=10, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
        out = gr.Textbox(label="Generated text", lines=5)
        gr.Button("Generate").click(run_text_gen, inputs=[prompt, max_tokens, temperature], outputs=out)


# ── HF Chat (Inference Endpoint) ─────────────────────────────────────────────

def run_hf_chat(prompt: str, system_prompt: str) -> str:
    from src.text_gen import generate_hf_endpoint

    if not prompt.strip():
        return "Please enter a message."
    try:
        return generate_hf_endpoint(prompt, system_prompt=system_prompt)
    except Exception as e:
        return f"Error: {e}\n\nMake sure HF_TOKEN is set in your .env file."


def build_hf_chat_tab():
    with gr.Tab("HF Chat (Llama-3.1)"):
        gr.Markdown("Chat via the **HuggingFace Inference Router** (requires `HF_TOKEN` in `.env`).")
        system = gr.Textbox(label="System prompt", value="You are a helpful assistant.", lines=2)
        prompt = gr.Textbox(label="Your message", placeholder="Ask me anything…", lines=3)
        out = gr.Textbox(label="Response", lines=8)
        gr.Button("Send").click(run_hf_chat, inputs=[prompt, system], outputs=out)


# ── Image Captioning ──────────────────────────────────────────────────────────

def run_captioning(image) -> str:
    from src.captioning import caption

    if image is None:
        return "Please upload an image."
    return caption(image)


def build_captioning_tab():
    with gr.Tab("Image Captioning"):
        gr.Markdown("Generate a caption for any image using **BLIP**.")
        img = gr.Image(type="pil", label="Upload image")
        out = gr.Textbox(label="Caption")
        gr.Button("Caption").click(run_captioning, inputs=img, outputs=out)


# ── Visual Question Answering ─────────────────────────────────────────────────

def run_vqa(image, question: str) -> str:
    from src.vqa import answer

    if image is None or not question.strip():
        return "Please upload an image and enter a question."
    results = answer(image, question, top_k=3)
    lines = []
    for i, r in enumerate(results):
        ans = r.get("answer", "")
        score = r.get("score")
        if score is not None:
            lines.append(f"{i+1}. {ans} ({score:.1%})")
        else:
            lines.append(f"{i+1}. {ans}")
    return "\n".join(lines)



def build_vqa_tab():
    with gr.Tab("Visual QA"):
        gr.Markdown("Ask a question about an image using **BLIP VQA**.")
        with gr.Row():
            img = gr.Image(type="pil", label="Upload image")
            with gr.Column():
                question = gr.Textbox(label="Question", placeholder="How many objects are there?")
                out = gr.Textbox(label="Top answers", lines=5)
                gr.Button("Ask").click(run_vqa, inputs=[img, question], outputs=out)


# ── Speech to Text ────────────────────────────────────────────────────────────

def run_asr(audio) -> str:
    from src.asr import transcribe

    if audio is None:
        return "Please record or upload audio."
    return transcribe(audio)


def build_asr_tab():
    with gr.Tab("Speech to Text"):
        gr.Markdown("Transcribe audio using **Whisper Small**.")
        audio = gr.Audio(label="Record or upload audio", type="numpy")
        out = gr.Textbox(label="Transcription", lines=4)
        gr.Button("Transcribe").click(run_asr, inputs=audio, outputs=out)


# ── Image Generation ──────────────────────────────────────────────────────────

def run_txt2img(prompt: str, steps: int, guidance: float):
    from src.image_gen import text_to_image

    if not prompt.strip():
        return None
    return text_to_image(prompt, steps=int(steps), guidance_scale=guidance)


def run_img2img(image, prompt: str, strength: float, steps: int, guidance: float):
    from src.image_gen import image_to_image

    if image is None or not prompt.strip():
        return None
    return image_to_image(image, prompt, strength=strength, steps=int(steps), guidance_scale=guidance)


def build_image_gen_tab():
    with gr.Tab("Image Generation"):
        gr.Markdown(
            "Generate images with **Stable Diffusion v1.5**.  \n"
            "> ⚠️ First run downloads ~4 GB. GPU strongly recommended."
        )
        with gr.Tabs():
            with gr.Tab("Text → Image"):
                prompt = gr.Textbox(label="Prompt", placeholder="A majestic lion wearing a crown…", lines=2)
                with gr.Row():
                    steps = gr.Slider(10, 50, value=30, step=5, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                out_img = gr.Image(label="Generated image")
                gr.Button("Generate").click(run_txt2img, inputs=[prompt, steps, guidance], outputs=out_img)

            with gr.Tab("Image → Image"):
                with gr.Row():
                    init_img = gr.Image(type="pil", label="Input image")
                    out_img2 = gr.Image(label="Output image")
                prompt2 = gr.Textbox(label="Prompt", placeholder="A futuristic city skyline…", lines=2)
                with gr.Row():
                    strength = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength")
                    steps2 = gr.Slider(10, 50, value=40, step=5, label="Steps")
                    guidance2 = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                gr.Button("Transform").click(
                    run_img2img,
                    inputs=[init_img, prompt2, strength, steps2, guidance2],
                    outputs=out_img2,
                )


# ── App entry point ───────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="HuggingFace AI Lab") as demo:
        gr.Markdown("# HuggingFace AI Lab\nAll models load lazily on first use.")
        build_sentiment_tab()
        build_text_gen_tab()
        build_hf_chat_tab()
        build_captioning_tab()
        build_vqa_tab()
        build_asr_tab()
        build_image_gen_tab()
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
