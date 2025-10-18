#!/usr/bin/env python3
"""
Debug test to see exactly where the list index error occurs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_summarization_directly():
    """Test summarization directly to see the error"""
    print("Testing summarization directly...")
    
    try:
        from backend.app.agents.summarizer import abstractive_summarize, extractive_filter
        
        # Test with a simple text
        test_text = "This is a test document about management principles. It contains information about planning, organizing, leading, and controlling. These are the four main functions of management that every manager should understand and apply in their daily work."
        
        print(f"Testing with text: {test_text}")
        print(f"Text length: {len(test_text)}")
        
        # Test extractive filter first
        print("\n=== Testing Extractive Filter ===")
        extractive_result = extractive_filter(test_text, top_k=3)
        print(f"Extractive result: {extractive_result}")
        
        # Test abstractive summarization
        print("\n=== Testing Abstractive Summarization ===")
        abstractive_result = abstractive_summarize(test_text, max_length=100, min_length=30)
        print(f"Abstractive result: {abstractive_result}")
        
        return True
        
    except Exception as e:
        print(f"Error in direct test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunk_summarization():
    """Test chunk summarization"""
    print("\n=== Testing Chunk Summarization ===")
    
    try:
        from backend.app.agents.summarizer import chunk_and_summarize_chunks
        
        chunks = [
            {"text": "Management is the process of planning, organizing, leading, and controlling resources to achieve organizational goals.", "page": 1},
            {"text": "Planning involves setting objectives and determining the best course of action to achieve them.", "page": 2},
            {"text": "Organizing involves arranging resources and tasks to implement the plan effectively.", "page": 3}
        ]
        
        result = chunk_and_summarize_chunks(chunks)
        print(f"Chunk summarization result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error in chunk test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_agent():
    """Test QAAgent directly"""
    print("\n=== Testing QAAgent ===")
    
    try:
        from backend.app.crew_orchestrator import CrewOrchestrator
        
        # Initialize orchestrator
        orchestrator = CrewOrchestrator(use_crew_sdk=False)
        
        # Test query
        query = "What are the main management principles?"
        print(f"Testing query: {query}")
        
        result = orchestrator.run_query(query, top_k=3)
        print(f"QAAgent result: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error in QAAgent test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("Starting debug tests...")
    print("=" * 50)
    
    tests = [
        test_summarization_directly,
        test_chunk_summarization,
        test_qa_agent
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test failed with error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    main()
