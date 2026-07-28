import os
import uuid
import shutil
import tempfile
import asyncio
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from datetime import datetime, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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

# Security Middleware for OWASP HTTP Headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Store advanced RAG pipelines in-memory keyed by session id
sessions_db: Dict[str, AdvancedRAGPipeline] = {}

# Keep track of active temp files to clean up later
sessions_temp_files: Dict[str, str] = {}

# In-memory daily upload records: tracks client IP -> list of upload datetimes
upload_records: Dict[str, List[datetime]] = defaultdict(list)

def check_upload_limit(client_ip: str) -> Tuple[bool, str]:
    now = datetime.now()
    # Get midnight today local time
    midnight = datetime.combine(now.date(), time.min)
    
    # Filter out uploads from before midnight today
    upload_records[client_ip] = [t for t in upload_records[client_ip] if t >= midnight]
    
    if len(upload_records[client_ip]) >= 10:
        return False, "Daily upload limit of 10 documents exceeded. Resets at midnight."
    
    return True, ""

def record_upload(client_ip: str):
    upload_records[client_ip].append(datetime.now())

import re

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            return loader.load()
        except Exception:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            return loader.load()
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()



SUPPORTED_EXTENSIONS = [
    ".pdf", ".txt", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".html", ".md", ".csv", ".json"
]

DANGEROUS_EXTENSIONS = [
    ".exe", ".py", ".sh", ".php", ".js", ".bat", ".vbs", ".dll", ".cmd", ".ps1", ".jar"
]

@app.on_event("startup")
async def startup_event():
    logger = sys.modules.get("Advance-Rag.logger") or sys.modules.get("logger")
    if logger and hasattr(logger, "info"):
        logger.info("Server startup: Initiating background pipeline warmup...")
    # Pre-warm AdvancedRAGPipeline (triggering HuggingFaceEmbeddings model load)
    asyncio.create_task(asyncio.to_thread(AdvancedRAGPipeline))

@app.get("/")
@app.get("/api")
async def root():
    return {
        "status": "online",
        "service": "Advanced Cognitive RAG Hub API",
        "version": "1.0.0",
        "endpoints": ["/api/upload", "/api/chat", "/api/warmup", "/api/health"]
    }

@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/warmup")
@app.get("/api/warmup")
async def warmup_pipeline():
    # Trigger model load in background if not already loaded
    asyncio.create_task(asyncio.to_thread(AdvancedRAGPipeline))
    return {"status": "warming_up"}

@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    # Enforce upload limits (10 documents per 24 hours from midnight)
    client_ip = request.client.host if request.client else "unknown"
    is_allowed, limit_msg = check_upload_limit(client_ip)
    if not is_allowed:
        raise HTTPException(status_code=429, detail=limit_msg)

    # Read uploaded bytes directly
    content = await file.read()
    size = len(content)

    # Enforce file size limit (50MB)
    if size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 50MB.")

    # 1. Path Traversal Defense: Sanitize filename
    raw_filename = os.path.basename(file.filename or "uploaded_document")
    filename = re.sub(r"[^\w\s.-]", "_", raw_filename).strip()
    ext = os.path.splitext(filename)[-1].lower()

    if not ext or ext in DANGEROUS_EXTENSIONS or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Security Error: Prohibited or unsupported file format '{ext}'.")

    # 2. Magic Bytes Inspection (MIME Spoofing Defense)
    header_bytes = content[:512]

    if ext == ".pdf" and not header_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Security Error: File header does not match valid PDF magic bytes.")
    if ext in [".docx", ".xlsx", ".pptx"] and not header_bytes.startswith(b"PK\x03\x04"):
        raise HTTPException(status_code=400, detail=f"Security Error: File content does not match binary format for '{ext}'.")

    # Check if appending to an existing session
    is_append = False
    if session_id and session_id in sessions_db:
        pipeline = sessions_db[session_id]
        is_append = True
    else:
        session_id = str(uuid.uuid4())
        pipeline = AdvancedRAGPipeline()

    # Write upload to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name


    try:
        # Load document using production-grade PyPDFLoader / TextLoader
        docs = load_document(temp_file_path)


        # Build or add to Advanced RAG pipeline for this session
        pipeline.add_documents(docs)
        
        sessions_db[session_id] = pipeline
        sessions_temp_files[session_id] = temp_file_path
        
        # Get chunk projections for 3D layout
        projections = pipeline.get_chunk_projections()
        
        # Log successful upload
        record_upload(client_ip)
        
        return {
            "session_id": session_id,
            "status": "indexed",
            "filename": filename,
            "projections": projections,
            "is_append": is_append
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@app.post("/api/chat")
async def chat_session(


    session_id: str = Form(...), 
    question: str = Form(...),
    top_k: int = Form(10)
):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please upload document again.")

    pipeline = sessions_db[session_id]
    pipeline.top_k = top_k

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




'''
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

'''