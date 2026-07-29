import os
import sys
import numpy as np
from langchain_core.documents import Document

sys.path.append(os.path.join(os.getcwd(), "Advance-Rag"))

from advance_rag import AdvancedRAGPipeline
from cache import RAGCache

def test_session_isolation():
    print("Testing Multi-Tenant Session Isolation...")

    # Session 1: Resume document
    session_1_id = "sess_resume_1111"
    pipeline_1 = AdvancedRAGPipeline(session_id=session_1_id)
    doc_1 = Document(
        page_content="Bhubhurv's resume: Software Engineer specialized in React, Python, RAG, and AI Systems.",
        metadata={"source": "resume.pdf"}
    )
    pipeline_1.add_documents([doc_1])

    # Session 2: Unit V textbook document
    session_2_id = "sess_unit5_2222"
    pipeline_2 = AdvancedRAGPipeline(session_id=session_2_id)
    doc_2 = Document(
        page_content="Chapter 5: Advanced Cognitive Architectures and Neural Network Optimization. Page 102.",
        metadata={"source": "unit_v.pdf"}
    )
    pipeline_2.add_documents([doc_2])

    # Query Session 2 using hybrid search
    dummy_query = "What is Name of Chapter?"
    dummy_emb = np.array(pipeline_2.embeddings.embed_query(dummy_query))
    
    docs_2, max_score, top_vec_score = pipeline_2.custom_hybrid_search(dummy_query, dummy_emb, top_k=5)

    print(f"Session 2 retrieved {len(docs_2)} documents for query '{dummy_query}':")
    for d in docs_2:
        print(f"  - [{d.metadata.get('session_id')}] {d.page_content}")
        assert d.metadata.get("session_id") == session_2_id, f"Cross-session bleed detected! Retrieved chunk from session {d.metadata.get('session_id')}"

    print("[SUCCESS] 0 chunks from Session 1 bled into Session 2!")

    # Test Session Cache Isolation
    pipeline_1.cache.set("What is Name of Chapter?", "Chapter 1: Resume Introduction", query_emb=dummy_emb)
    
    # Query Cache in Session 2
    cached_res_2 = pipeline_2.cache.get("What is Name of Chapter?", query_emb=dummy_emb)
    assert cached_res_2 is None, f"Cross-session cache bleed detected! Session 2 got cached response: {cached_res_2}"

    print("[SUCCESS] Cache in Session 2 is completely isolated from Session 1!")

    # Test Session Reset
    pipeline_1.cache.clear_session(session_1_id)
    print("[SUCCESS] Session cache clear executed cleanly!")


if __name__ == "__main__":
    test_session_isolation()
