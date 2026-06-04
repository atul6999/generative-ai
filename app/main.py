from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Generative AI Starter",
    description="A minimal FastAPI scaffold for generative AI applications.",
    version="0.1.0",
)

class PingResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=PingResponse)
def read_root():
    return {"status": "ok", "message": "Generative AI starter is running."}

@app.post("/generate")
def generate_text(input: dict):
    prompt = input.get("prompt", "")
    return {"prompt": prompt, "output": f"Echo: {prompt}"}
