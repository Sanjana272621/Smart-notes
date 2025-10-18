#!/usr/bin/env python3
"""
Test script to verify the summarization pipeline works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_pipeline_directly():
    """Test the pipeline directly to see what it returns"""
    print("Testing summarization pipeline directly...")
    
    try:
        from transformers import pipeline
        from backend.app.agents.config import SUMMARIZER_MODEL
        
        print(f"Initializing pipeline with model: {SUMMARIZER_MODEL}")
        
        # Initialize pipeline
        summarizer = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            tokenizer=SUMMARIZER_MODEL,
            device=-1
        )
        
        # Test with a simple text
        test_text = "This is a test document about artificial intelligence. It contains multiple sentences that should be summarized. The goal is to test if the pipeline returns a proper summary. This text is long enough to be summarized properly."
        
        print(f"Input text: {test_text}")
        print(f"Input length: {len(test_text)} characters")
        
        # Call the pipeline
        result = summarizer(test_text, max_length=100, min_length=30)
        
        print(f"Pipeline result type: {type(result)}")
        print(f"Pipeline result: {result}")
        
        if isinstance(result, list) and len(result) > 0:
            print(f"First element: {result[0]}")
            if isinstance(result[0], dict):
                print(f"Keys in first element: {result[0].keys()}")
                if "summary_text" in result[0]:
                    print(f"Summary text: {result[0]['summary_text']}")
                else:
                    print("No 'summary_text' key found!")
            else:
                print(f"First element is not a dict: {type(result[0])}")
        else:
            print("Pipeline returned empty list or non-list result!")
            
        return True
        
    except Exception as e:
        print(f"Error testing pipeline: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_texts():
    """Test with different text lengths"""
    print("\nTesting with different text lengths...")
    
    try:
        from transformers import pipeline
        from backend.app.agents.config import SUMMARIZER_MODEL
        
        summarizer = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            tokenizer=SUMMARIZER_MODEL,
            device=-1
        )
        
        test_cases = [
            "Short text.",
            "This is a medium length text that should work for summarization.",
            "This is a longer text that contains multiple sentences and should definitely work for summarization. It has enough content to be processed by the T5 model and should return a proper summary.",
            "This is a very long text that exceeds the typical token limits for T5-small. " * 20
        ]
        
        for i, text in enumerate(test_cases):
            print(f"\nTest case {i+1}:")
            print(f"Text length: {len(text)} characters")
            print(f"Text: {text[:100]}...")
            
            try:
                result = summarizer(text, max_length=50, min_length=10)
                print(f"Result: {result}")
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "summary_text" in result[0]:
                    print(f"Summary: {result[0]['summary_text']}")
                else:
                    print("Invalid result format!")
            except Exception as e:
                print(f"Error with this text: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in text length tests: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing summarization pipeline...")
    print("=" * 50)
    
    tests = [
        test_pipeline_directly,
        test_with_different_texts
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"Results: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
