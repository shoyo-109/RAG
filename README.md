# Advanced Cognitive RAG Hub

An enterprise-ready, production-grade Retrieval-Augmented Generation (RAG) platform. Features semantic percentile chunking, Chroma HNSW vector database,Reciprocal Rank Fusion (RRF) hybrid search (lexical + vector), dynamic semantic cache, Nvidia Nemotron primary LLM reasoning + GPT-4o-Mini fallback controls, and a fully interactive 3D embedding space visualizer.

---

## 📸 Platform Screenshots
<img width="1913" height="1078" alt="image" src="https://github.com/user-attachments/assets/03bdec73-3d61-4a2b-a5e7-22ac740771a8" />


| Conversation Dashboard | 3D Embedding Space Projection |
| :---: | :---: |
| <img width="1373" height="652" alt="image" src="https://github.com/user-attachments/assets/df90ca50-8efe-44d6-87b7-239a08ea4fa4" />
 | <img width="1793" height="906" alt="image" src="https://github.com/user-attachments/assets/85a53c80-8a0e-47e8-9f53-c204f6b8bbb0" />
 |

---

## 🛠️ Architecture & Technical Features

1. **Precision Retrieval**: Combines semantic text splitting with a 90th percentile breakpoint, HNSW graph vector indices ($M=20$, $ef=100$), and lexical BM25 matching fused using equal weight Reciprocal Rank Fusion ($0.5$ vector / $0.5$ keyword).
2. **Instability Shields**: Exponential backoff retries on LLM endpoints, circuit breakers to prevent cascaded downstream timeouts, and automatic failovers from Nvidia Nemotron models to GPT-4o-Mini.
3. **Defense & PII Protection**: Real-time prompt injection filtering and regex-based PII masking (SSN, credit card, email, IP, phone) on query inputs and LLM outputs.
4. **Structured Observability**: Formats all logs to JSON format for aggregation and records telemetry (error rates, processing latency, input/output tokens, and cache hits).
5. **Real Space Visualizer**: 3D PCA dimensional reduction engine projecting chunk embeddings into an interactive space background filled with stars and drifting meteors.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.10+ and Node.js 18+ installed on your system.

### 🔑 Environment Variables
Create a `.env` file in the root directory:
```env
# Primary LLM API Key
NVIDIA_API_KEY=your_nvidia_api_key_here

# Fallback LLM & Security Guards Key
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Tracing Telemetry (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

---

## 💻 Commands Guide

### 1. Backend Setup & Run
Open a terminal in the root folder:

```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows Powershell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r Requirements.py

# Install PDF spacing parsing optimizer
pip install pymupdf

# Launch FastAPI backend server
python server.py
```

### 2. Frontend Setup & Run
Open a separate terminal in the `frontend` folder:

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

### 🧪 Run Pipeline Diagnostics Test
```bash
.\venv\Scripts\python.exe Advance-Rag/advance_rag.py
```
