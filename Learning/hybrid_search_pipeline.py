from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    Document(
        page_content="For network connectivity issues, first check the "
                     "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"}
    ),

    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
                     "rejected the connection. Check firewall settings.",
        metadata={"type": "error"}
    ),

    Document(
        page_content="The authentication process requires valid credentials. "
                     "Use OAuth2 for secure API access.",
        metadata={"type": "auth"}
    ),

    Document(
        page_content="Router configuration guide: Access the admin panel "
                     "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"}
    ),
]



vector_store = Chroma.from_documents(documents, embedding=embeddings, collection_name="Testing_Hybrid")
print("Vector Store Ready ...")

vector_retriver = vector_store.as_retriever(search_kwargs={"k":3})
print("Vector Retriever Ready ...")

bm25_retriver = BM25Retriever.from_documents(documents, k=3)
print("BM25 Retriever Ready ...")


def hybrid_search(query, retrievers, weight, k=3, rrf_k=60):
    rrf_scores = {}
    doc_map = {}
    
    for r_idx, retriever in enumerate(retrievers):
        # Get documents from the current retriever
        docs = retriever.invoke(query)
        w = weight[r_idx] if r_idx < len(weight) else 1.0
        
        for rank_idx, doc in enumerate(docs):
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            
            # Calculate RRF score: weight * (1 / (rrf_k + rank))
            # rank is 1-based (rank_idx + 1)
            score = w * (1.0 / (rrf_k + (rank_idx + 1)))
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + score
            
    # Sort documents based on their combined RRF score in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_key] for doc_key, _ in sorted_docs[:k]]


# Run manual test comparison
test_queries = [
    'E_CONN_REFUSED error',
    'OAuth2 API',
    'Router Configuration'
]

def print_results(query, name, docs):
    print(f"\n--- {name} Results for: '{query}' ---")
    for doc in docs:
        print(f"- {doc.page_content} (Metadata: {doc.metadata})")

for query in test_queries:
    vector_result = vector_retriver.invoke(query)
    bm25_result = bm25_retriver.invoke(query)
    
    # Custom hybrid search using our new function
    hybrid_result = hybrid_search(
        query=query,
        retrievers=[vector_retriver, bm25_retriver],
        weight=[0.5, 0.5],
        k=3,
        rrf_k=60
    )
    
    print_results(query, "VECTOR", vector_result)
    print_results(query, "BM25", bm25_result)
    print_results(query, "CUSTOM HYBRID (RRF)", hybrid_result)