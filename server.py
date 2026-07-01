import os
import uuid
import shutil
import tempfile
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import RAG pipeline functions
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Advance-Rag"))
from advance_rag import AdvancedRAGPipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader

app = FastAPI(title="Dynamic Session-based Advanced RAG API")

# Allow CORS for Next.js frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store advanced RAG pipelines in-memory keyed by session id
sessions_db: Dict[str, AdvancedRAGPipeline] = {}

# Keep track of active temp files to clean up later
sessions_temp_files: Dict[str, str] = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Create session id
    session_id = str(uuid.uuid4())

    # Write upload to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Route loaders based on file extension
        if ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif ext == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        docs = loader.load()

        # Build Advanced RAG pipeline for this session
        pipeline = AdvancedRAGPipeline()
        pipeline.add_documents(docs)
        
        sessions_db[session_id] = pipeline
        sessions_temp_files[session_id] = temp_file_path
        
        # Get chunk projections for 3D layout
        projections = pipeline.get_chunk_projections()
        
        return {
            "session_id": session_id,
            "status": "indexed",
            "filename": filename,
            "projections": projections
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@app.post("/chat")
async def chat_session(session_id: str = Form(...), question: str = Form(...)):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please upload document again.")

    pipeline = sessions_db[session_id]

    async def event_generator():
        try:
            async for chunk in pipeline.astream_query(question):
                yield chunk
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.on_event("shutdown")
def shutdown_event():
    # Cleanup all temp files on server close
    for path in sessions_temp_files.values():
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
