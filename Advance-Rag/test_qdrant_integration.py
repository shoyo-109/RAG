import os
import sys
from dotenv import load_dotenv

# Load environment variables from root directory
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pipeline import AdvancedRAGPipeline

def test_qdrant():
    print("Initializing AdvancedRAGPipeline with Qdrant Cloud backend...")
    pipeline = AdvancedRAGPipeline()
    print("Pipeline initialized successfully!")
    
    test_text = "Qdrant Cloud is a vector search engine designed for high performance and scalability."
    print("Adding text document to Qdrant Cloud...")
    pipeline.add_text_document(test_text, metadata={"source": "qdrant_test"})
    print("Text indexed successfully!")
    
    query = "What is Qdrant Cloud?"
    print(f"Querying: '{query}'")
    response = pipeline.query(query)
    print(f"\nResponse:\n{response}")
    
    print("\nQdrant Cloud Integration Test Completed Successfully!")

if __name__ == "__main__":
    test_qdrant()
