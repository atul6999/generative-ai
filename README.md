# generative-ai

A minimal Python FastAPI scaffold for generative AI projects.

## Getting started

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Endpoints

- `GET /` - health check endpoint
- `POST /generate` - sample text generation endpoint

## Notes

This scaffold includes a starter FastAPI app and a simple test case.
