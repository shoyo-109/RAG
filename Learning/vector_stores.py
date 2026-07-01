# import chromadb

# chroma_client= chromadb.Client()

# collection = chroma_client.get_or_create_collection("resume_parser")



from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

doc = [
    {"id" : 1, "text" : "Bhubhurv has 5 years of experience as a software engineer. He has worked on various projects in the IT industry. He is a skilled programmer and has experience in various programming languages."},
    {"id" : 2, "text" : "Bhubhurv silently enjoys time with his friends. But prefers to stay alone."},
    {"id" : 3, "text" : "Bhubhurv once used to strict exercise, achieving a peak of performingg 100 ushups, 100 squats, 100 crunches "}
]



def similarity_search_by_scores():

    query = 'Bhubhurv'
    # response_with_scores = collection.query(
    #     query_texts=query,
    #     n_results=2,
    #     include=["documents", "distances"]
    # )

    # return response_with_scores

    filter_criteria = {"id": 2}
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(query)

    # 1. Filter documents first based on criteria
    filtered_docs = [
        d for d in doc 
        if all(d.get(k) == v for k, v in filter_criteria.items())
    ]

    if not filtered_docs:
        return []

    # 2. Extract only the text value from the filtered dictionary list
    doc_texts = [d["text"] for d in filtered_docs]
    doc_embeddings = model.encode(doc_texts)

    # 3. Calculate similarity scores between the query and filtered documents
    scores = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    # Return filtered documents paired with their similarity score
    results = []
    for i, score in enumerate(scores):
        results.append({
            "id": filtered_docs[i]["id"], 
            "text": filtered_docs[i]["text"], 
            "similarity_score": float(score)
        })

    return results



# def metadata_filtering():
    filter = {"id":2}

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(query)

    # Extract only the text value from the dictionary list
    doc_texts = [d["text"] for d in doc]
    doc_embeddings = model.encode(doc_texts)

    # Calculate similarity scores between the query and all documents
    scores = cosine_similarity(
        [query_embedding],
        doc_embeddings,
        filter=filter
    )[0]

    # Return documents paired with their similarity score
    results = []
    for i, score in enumerate(scores):
        results.append({"id": doc[i]["id"], "text": doc[i]["text"], "similarity_score": float(score)})

    return results 


if __name__ == "__main__":
    print(similarity_search_by_scores())
    # print(metadata_filtering())
    