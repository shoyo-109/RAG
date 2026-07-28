import os
import gradio as gr
from server import app as fastapi_app

# ZeroGPU decorator hook for Hugging Face Spaces
try:
    import spaces
    @spaces.GPU
    def zero_gpu_pipeline_hook():
        """Hook to register ZeroGPU hardware allocation on startup."""
        return True
    
    # Execute immediately on import to register with ZeroGPU supervisor
    zero_gpu_pipeline_hook()
except (ImportError, Exception):
    def zero_gpu_pipeline_hook():
        return True

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

# Expose `app` at module level for Hugging Face Space runner
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")



