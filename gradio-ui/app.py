import os
import httpx
import httpx
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://edge-nginx:8088").rstrip("/")
EDGE_API_KEY = os.getenv("EDGE_API_KEY", "")

# Long timeout because first LLM call can be slow (model load, cold start)
HTTP_TIMEOUT_S = float(os.getenv("GRADIO_HTTP_TIMEOUT_S", "600"))

def _timeout_s(default: float = 600.0) -> float:
    raw = (os.getenv("GRADIO_HTTP_TIMEOUT_S") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def call_api(path: str, payload: dict):
    if not EDGE_API_KEY:
        return {"error": "EDGE_API_KEY is not set for the UI container."}

    headers = {"X-API-Key": EDGE_API_KEY}

    timeout = httpx.Timeout(_timeout_s(), connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{API_BASE_URL}{path}", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

def chat_fn(message, history):
    data = call_api("/chat", {"message": message})
    if "error" in data:
        return f"Error: {data['error']}"
    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []

    if sources:
        answer += "\n\n---\n**Sources (retrieved from Qdrant):**\n"
        for i, s in enumerate(sources, start=1):
            title = s.get("title") or "Untitled"
            url = s.get("url") or ""
            dist = s.get("distance")
            answer += f"{i}. {title} — {url} (distance={dist})\n"
    return answer

def health_text():
    if not EDGE_API_KEY:
        return "UI misconfigured: EDGE_API_KEY is missing."
    headers = {"X-API-Key": EDGE_API_KEY}
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE_URL}/health", headers=headers)
            return f"{r.status_code}: {r.text}"
    except Exception as e:
        return f"Health check failed: {e}"

with gr.Blocks(title="Lab 2 — Chat with Qdrant (RAG)") as demo:
    gr.Markdown(
        "# Lab 2 — Chat with Your Dataset (RAG)\n"
        "This UI chats with your dataset using:\n"
        "- Retrieval from Qdrant\n"
        "- A local LLM via Ollama\n"
        "- An API layer behind an authenticated NGINX proxy\n"
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.ChatInterface(chat_fn)
        with gr.Column(scale=1):
            gr.Markdown("## System status")
            btn = gr.Button("Refresh health")
            out = gr.Textbox(label="API /health output", lines=10)
            btn.click(fn=health_text, outputs=out)

demo.launch(server_name="0.0.0.0", server_port=7860)
