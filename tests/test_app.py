from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Generative AI starter is running."}


def test_generate_text():
    response = client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"prompt": "Hello", "output": "Echo: Hello"}
