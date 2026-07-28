import os
import gradio as gr
from server import app as fastapi_app

# Create a clean Gradio landing page for Hugging Face Space UI
demo = gr.Blocks(title="Advanced Cognitive RAG Hub API")
with demo:
    gr.Markdown(
        """
        # 🚀 Advanced Cognitive RAG Hub Backend API
        
        Your production FastAPI backend is live and healthy!
        
        ### 📌 Available API Endpoints:
        * **`GET /`**: Server status & endpoints overview
        * **`GET /health`**: Health check status (`200 OK`)
        * **`POST /upload`**: Document ingestion pipeline (PDF, TXT, DOCX, CSV, etc.)
        * **`POST /chat`**: Streaming RAG response endpoint (`text/event-stream`)
        * **`GET /warmup`**: Model pre-warming trigger
        """
    )

# Expose `app` at module level so Hugging Face Space automatically serves FastAPI & Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


