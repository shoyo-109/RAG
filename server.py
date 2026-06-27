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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Basic RAG"))
from basic_rag import create_knowledgebase

app = FastAPI(title="Dynamic Session-based RAG API")

# Allow CORS for Next.js frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store vector databases in-memory keyed by session id
sessions_db: Dict[str, any] = {}

# Keep track of active temp files to clean up later
sessions_temp_files: Dict[str, str] = {}

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

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
        # Build Chroma vector store for this session
        vector_store = create_knowledgebase(temp_file_path)
        sessions_db[session_id] = vector_store
        sessions_temp_files[session_id] = temp_file_path
        
        return {"session_id": session_id, "status": "indexed", "filename": filename}
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@app.post("/chat")
async def chat_session(session_id: str = Form(...), question: str = Form(...)):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please upload document again.")

    vector_store = sessions_db[session_id]
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.5,
        streaming=True,
        model_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384
            }
        }
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context provided:
        
        Context:
        {context}
        
        Question:
        {question}

        If you don't have enough context, respond with "I don't have enough context"
        """
    )

    # Format documents context
    docs = retriever.invoke(question)
    context_str = format_docs(docs)

    # Prepare chain
    chain = prompt | llm | StrOutputParser()

    async def event_generator():
        try:
            # First, stream reasoning stages to client to make it feel interactive
            stages = [
                "🔍 Consulting session knowledge base...",
                "🧠 Analyzing context chunks...",
                "📝 Formulating final response..."
            ]
            for stage in stages:
                yield f"data: {stage}\n\n"
                await asyncio.sleep(0.8)

            # Stream tokens
            async for chunk in chain.astream({"context": context_str, "question": question}):
                yield f"data: {chunk}\n\n"
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
