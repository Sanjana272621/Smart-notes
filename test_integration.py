#!/usr/bin/env python3
"""
Simple integration test script to verify the backend API endpoints work correctly.
Run this after starting the backend server.
"""

import requests
import json

# Test configuration
BASE_URL = "http://127.0.0.1:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_summary_endpoint():
    """Test the summary endpoint"""
    print("\nTesting summary endpoint...")
    try:
        payload = {
            "query": "What are the main topics discussed?",
            "top_k": 3
        }
        response = requests.post(f"{BASE_URL}/summary", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Summary points: {len(data.get('summaryPoints', []))}")
            print(f"Q&A pairs: {len(data.get('qa', []))}")
            print(f"Sample summary point: {data.get('summaryPoints', [''])[0][:100]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_flashcards_endpoint():
    """Test the flashcards endpoint"""
    print("\nTesting flashcards endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/flashcards")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Flashcards: {len(data.get('flashcards', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_query_endpoint():
    """Test the query endpoint"""
    print("\nTesting query endpoint...")
    try:
        payload = {
            "query": "What is this document about?",
            "top_k": 3
        }
        response = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Answer length: {len(data.get('answer', ''))}")
            print(f"Sources: {len(data.get('sources', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting integration tests...")
    print("=" * 50)
    
    tests = [
        test_root_endpoint,
        test_query_endpoint,
        test_summary_endpoint,
        test_flashcards_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Integration is working correctly.")
    else:
        print("❌ Some tests failed. Check the backend server and try again.")

if __name__ == "__main__":
    main()
