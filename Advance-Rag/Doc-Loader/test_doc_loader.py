import os
import sys
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Ensure Advance-Rag and Doc-Loader are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
advance_rag_dir = os.path.dirname(current_dir)
if advance_rag_dir not in sys.path:
    sys.path.insert(0, advance_rag_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from capability_registry import CapabilityRegistry
from pipeline import IngestionPipeline


def run_tests():
    print("=== Step 1: Testing Capability Registry Startup Scan ===")
    reg = CapabilityRegistry()
    summary = reg.get_summary()
    print("Capability Scan Summary:")
    for cat, caps in summary.items():
        print(f"  {cat}: {caps}")

    print("\n=== Step 2: Testing Text Document Ingestion ===")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write("# Production RAG Ingestion Pipeline\n\nThis is a test document paragraph for validating canonical element normalization.\n\nAnother paragraph containing test data.")
        temp_txt_path = f.name

    try:
        docs = IngestionPipeline.load_document(temp_txt_path, tenant_id="test_tenant_123")
        print(f"Successfully loaded {len(docs)} document chunks.")
        for idx, d in enumerate(docs):
            print(f" Document {idx+1}:")
            print(f"   Content: {d.page_content}")
            print(f"   Metadata: {d.metadata}")
        assert len(docs) >= 2, "Expected at least 2 document chunks"
    finally:
        if os.path.exists(temp_txt_path):
            os.remove(temp_txt_path)

    print("\n=== Step 3: Testing Markdown Document Ingestion ===")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as f:
        f.write("# Section 1: Introduction\n\nWelcome to Extreme Condition RAG Ingestion.\n\n# Section 2: Architecture\n\nBuilt with Capability Registry & Dynamic Fallbacks.")
        temp_md_path = f.name

    try:
        docs = IngestionPipeline.load_document(temp_md_path, tenant_id="test_tenant_md")
        print(f"Successfully loaded {len(docs)} Markdown chunks.")
        for idx, d in enumerate(docs):
            print(f"  Chunk {idx+1} [{d.metadata.get('element_type')}]: {d.page_content[:50]}...")
        assert len(docs) >= 4, "Expected at least 4 chunks"
    finally:
        if os.path.exists(temp_md_path):
            os.remove(temp_md_path)

    print("\n✅ ALL DOC-LOADER INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
