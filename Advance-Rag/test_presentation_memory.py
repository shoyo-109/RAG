import time
import logging
import sys
import os

# Ensure paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory import MultiLayerMemoryManager, ContextAllocator, TopicTracker
from presentation import PresentationBuilder, PresentationTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPresentationMemory")

def test_cpu_speed_benchmark():
    logger.info("--- Test 1: CPU Speed Benchmark (< 1.0ms target) ---")
    raw_markdown = """
    ChromaDB.### Education#### Bachelor of Technology
    Computer Science Engineering - 8.5 CGPA
    
    ### Technical Skills
    Java, Python, React.js, Express.js, RAG, ChromaDB, Docker, Git, LangChain
    
    - Core Python programming with high concurrency
    - Built RAG pipelines with semantic chunking
    - Fast sub-millisecond retrieval on HNSW indices
    - Implemented PII sanitization and hallucination guardrails
    - Optimized token allocation across context windows
    - Integrated multi-model failovers with backoff
    - Deployed microservices on Docker containers
    """
    
    iterations = 50
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = PresentationBuilder.transform_to_presentation(
            raw_markdown,
            active_topic="Technical Stack",
            should_render_header=True
        )
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000.0
    print(f"Average CPU Presentation Engine Latency: {avg_latency_ms:.3f} ms / execution")
    assert avg_latency_ms < 1.0, f"CPU latency benchmark failed: {avg_latency_ms:.3f} ms exceeds 1.0ms limit!"
    print("[OK] CPU Speed Benchmark PASSED!")

def test_data_grouping():
    logger.info("\n--- Test 2: Data Grouping & Badges Taxonomy ---")
    raw_skills = ["Python", "Java", "React.js", "RAG", "ChromaDB", "Docker", "Git"]
    grouped = PresentationTuner.group_raw_list_items(raw_skills)
    
    group_map = {g.category_name: g.items for g in grouped}
    print("Grouped Categories:", group_map)
    
    assert "Languages" in group_map, "Languages category missing in taxonomy matching"
    assert "Python" in group_map["Languages"], "Python should be under Languages"
    assert "AI & RAG Stack" in group_map, "AI & RAG Stack category missing"
    assert "ChromaDB" in group_map["AI & RAG Stack"], "ChromaDB should be under AI & RAG Stack"
    print("[OK] Data Grouping Test PASSED!")

def test_multi_turn_topic_tracker():
    logger.info("\n--- Test 3: Multi-Turn Topic Tracker & Header Control ---")
    manager = MultiLayerMemoryManager(capacity=5)
    
    # Turn 1: Topic shift to Work Experience
    state1 = manager.process_incoming_query("Tell me about Bhubhurv's work experience")
    assert state1["topic_analysis"]["is_topic_shift"] == True, "Turn 1 should be a topic shift"
    assert state1["topic_analysis"]["should_render_top_header"] == True, "Turn 1 should render top header"
    
    manager.record_completed_turn(
        "Tell me about Bhubhurv's work experience",
        "### Work Experience\nBhubhurv worked as a Software Engineer."
    )
    
    # Turn 2: Continuation query "What skills does he have?"
    state2 = manager.process_incoming_query("What skills does he have?")
    print("Turn 2 Topic Analysis:", state2["topic_analysis"])
    assert state2["topic_analysis"]["is_continuation"] == True, "Turn 2 should be detected as a continuation"
    assert state2["topic_analysis"]["should_render_top_header"] == False, "Turn 2 should NOT render top header"
    
    # Transform turn 2 response without top header
    transformed = PresentationBuilder.transform_to_presentation(
        "### Skills Overview\n- Python\n- React",
        active_topic=state2["topic_analysis"]["active_topic"],
        should_render_header=state2["topic_analysis"]["should_render_top_header"]
    )
    print("Transformed Continuation Output:\n", transformed)
    assert not transformed.startswith("###"), "Top header should be stripped on continuation turn"
    print("[OK] Multi-Turn Topic Tracker PASSED!")

def test_context_allocator():
    logger.info("\n--- Test 4: Context Allocator Token Budgeting ---")
    allocator = ContextAllocator(max_context_tokens=1000)
    
    rag_text = "RAG " * 2000  # Excessively large RAG text
    working_text = "User: Hello\nAssistant: Hi"
    summary_text = "Discussed greeting"
    
    result = allocator.allocate(
        rag_text=rag_text,
        working_memory_text=working_text,
        episodic_summary_text=summary_text
    )
    
    assert len(result["rag_context"]) <= (allocator.rag_budget * 4), "RAG context breached allocated character limit"
    assert len(result["memory_context"]) <= (allocator.memory_budget * 4), "Memory context breached allocated character limit"
    print("[OK] Context Allocator PASSED!")

if __name__ == "__main__":
    test_cpu_speed_benchmark()
    test_data_grouping()
    test_multi_turn_topic_tracker()
    test_context_allocator()
    print("\nALL PRESENTATION & MEMORY TESTS PASSED SUCCESSFULLY!")

