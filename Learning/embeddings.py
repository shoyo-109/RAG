# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

import numpy as np


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def basic_embeddings():
    text = "Hey, Keep it up ... "

    embedded = embeddings.encode(text)

    print(f"The text : {text}")
    print(f"The embedded vector : {embedded}")


def batch_embed():
    text = [
        "Bhubhurv's mother always said:",
        "Bhubhurv ..... Not sure",
        "The movie was absolutely amazing! I highly recommend it."
    ]
    embedded = embeddings.embed_documents(text)
    print(f"The embedded vectors : {embedded[0]}")



def similarity_Search():
    docs = [
        "Bhubhurv has 5 years of experience as a software engineer. He has worked on various projects in the IT industry. He is a skilled programmer and has experience in various programming languages.",
        "Bhubhurv silently enjoys time with his friends. But prefers to stay alone.",
        "Bhubhurv once used to strict exercise, achieving a peak of performingg 100 ushups, 100 squats, 100 crunches "
    ]

    query = "BHubhurv 5 year experience"
    
    doc_embedded = embeddings.embed_documents(docs)
    query_embedded = embeddings.embed_query(query)


    def cosine_similarity(vec1, vec2):
        return (np.dot(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    

    
    similarities = [cosine_similarity(query_embedded, doc) for doc in doc_embedded]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    
    print(f"Querry : {query}")
    print("Ranked Documents")
    for doc,score in ranked_docs:
        print(f"{score:.4f} : {doc}")


    



if __name__ == "__main__":
    # basic_embeddings()
    # batch_embed()
    similarity_Search()