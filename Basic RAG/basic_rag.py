from openai.types import vector_store
import os
from langchain_openai import ChatOpenAI
import tempfile
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
load_dotenv()

import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def create_knowledgebase(file_path: str):
    # Route loaders based on file extension
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    docs = loader.load()

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    

    # Vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp()
    )

    return vector_store


def demo_basic_rag():
    vector_store = create_knowledgebase("Documents/Bhubhurv_Resume.pdf")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.5,
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
    
    # format retrived documents
    def format_docs(docs):
        return "\n\n".join([document.page_content for document in docs])
    
    # Rag chain

    rag_chain = (
        {
            "context" : retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )



    # Test Rag Chain


    question = [
        "WHat Bhubhurv Hates ?",
        "What projects did Bhubhurv worked on ?"
    ]

    for q in question :
        print(f"Q : {q}")
        response=rag_chain.invoke(q)
        print(f"A : {response}")
        print("\n")

if __name__ == "__main__":
    demo_basic_rag()   