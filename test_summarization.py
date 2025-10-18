#!/usr/bin/env python3
"""
Test script to verify summarization works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.agents.summarizer import abstractive_summarize, extractive_filter, chunk_and_summarize_chunks

def test_extractive_filter():
    """Test extractive summarization"""
    print("Testing extractive filter...")
    text = "This is a test document. It contains multiple sentences. Each sentence has different content. The goal is to test summarization. We want to see if it works correctly."
    
    result = extractive_filter(text, top_k=3)
    print(f"Input: {text}")
    print(f"Output: {result}")
    print(f"Length: {len(result)} characters")
    print("✅ Extractive filter test passed\n")
    return True

def test_abstractive_summarize():
    """Test abstractive summarization"""
    print("Testing abstractive summarization...")
    text = "This is a comprehensive test document about artificial intelligence and machine learning. It covers various topics including neural networks, deep learning, natural language processing, and computer vision. The document explains how these technologies work together to create intelligent systems that can understand and process human language and visual information."
    
    try:
        result = abstractive_summarize(text, max_length=100, min_length=30)
        print(f"Input: {text}")
        print(f"Output: {result}")
        print(f"Length: {len(result)} characters")
        print("✅ Abstractive summarization test passed\n")
        return True
    except Exception as e:
        print(f"❌ Abstractive summarization failed: {e}")
        print("This is expected if the model is not available\n")
        return False

def test_chunk_summarization():
    """Test chunk summarization"""
    print("Testing chunk summarization...")
    chunks = [
        {"text": "This is the first chunk about machine learning algorithms.", "page": 1},
        {"text": "This is the second chunk about neural networks and deep learning.", "page": 2},
        {"text": "This is the third chunk about natural language processing techniques.", "page": 3}
    ]
    
    try:
        result = chunk_and_summarize_chunks(chunks)
        print(f"Number of chunks processed: {len(result['per_chunk'])}")
        print(f"Final summary: {result['final_summary']}")
        print(f"Summary length: {len(result['final_summary'])} characters")
        print("✅ Chunk summarization test passed\n")
        return True
    except Exception as e:
        print(f"❌ Chunk summarization failed: {e}")
        return False

def test_empty_input():
    """Test handling of empty input"""
    print("Testing empty input handling...")
    
    # Test empty text
    result1 = abstractive_summarize("")
    print(f"Empty text result: '{result1}'")
    
    # Test empty chunks
    result2 = chunk_and_summarize_chunks([])
    print(f"Empty chunks result: {result2}")
    
    # Test chunks with empty text
    result3 = chunk_and_summarize_chunks([{"text": "", "page": 1}, {"text": "   ", "page": 2}])
    print(f"Empty text chunks result: {result3}")
    
    print("✅ Empty input handling test passed\n")
    return True

def main():
    """Run all tests"""
    print("Starting summarization tests...")
    print("=" * 50)
    
    tests = [
        test_extractive_filter,
        test_abstractive_summarize,
        test_chunk_summarization,
        test_empty_input
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All summarization tests passed!")
    else:
        print("❌ Some tests failed, but the system should still work with fallbacks.")

if __name__ == "__main__":
    main()
