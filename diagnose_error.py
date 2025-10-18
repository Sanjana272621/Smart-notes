#!/usr/bin/env python3
"""
Comprehensive diagnostic script to find the exact source of the list index out of range error.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_embeddings():
    """Test embeddings service"""
    print("=== Testing Embeddings Service ===")
    try:
        from backend.app.agents.embeddings import EmbeddingService
        
        service = EmbeddingService()
        test_texts = ["This is a test", "Another test sentence"]
        
        embeddings = service.embed(test_texts)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings type: {type(embeddings)}")
        print("✅ Embeddings service working")
        return True
    except Exception as e:
        print(f"❌ Embeddings service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_index():
    """Test FAISS index"""
    print("\n=== Testing FAISS Index ===")
    try:
        from backend.app.agents.faiss_index import FaissIndex
        
        # Create a test index
        dim = 384  # all-MiniLM-L6-v2 dimension
        index = FaissIndex(dim)
        
        # Test with dummy data
        import numpy as np
        test_vectors = np.random.rand(2, dim).astype(np.float32)
        test_metadata = [
            {"text": "Test document 1", "page": 1},
            {"text": "Test document 2", "page": 2}
        ]
        
        index.add(test_vectors, test_metadata)
        print(f"Added {len(test_metadata)} vectors to index")
        
        # Test search
        query_vector = np.random.rand(dim).astype(np.float32)
        hits = index.search(query_vector, top_k=2)
        print(f"Search returned {len(hits)} hits")
        print(f"Hits: {hits}")
        
        print("✅ FAISS index working")
        return True
    except Exception as e:
        print(f"❌ FAISS index failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_agent_directly():
    """Test QAAgent directly"""
    print("\n=== Testing QAAgent Directly ===")
    try:
        from backend.app.agents.embeddings import EmbeddingService
        from backend.app.agents.faiss_index import FaissIndex
        from backend.app.agents.agents import QAAgent, SummarizerAgent
        
        # Initialize components
        embedding_service = EmbeddingService()
        faiss_agent = FAISSAgent(dim=384)
        summarizer_agent = SummarizerAgent()
        qa_agent = QAAgent(embedding_service, faiss_agent, summarizer_agent)
        
        # Test query
        query = "What is management?"
        print(f"Testing query: {query}")
        
        result = qa_agent.run(query, top_k=3)
        print(f"QAAgent result: {result}")
        print(f"Answer: {result.payload.get('answer', 'No answer')}")
        print(f"Sources: {len(result.payload.get('sources', []))}")
        
        print("✅ QAAgent working")
        return True
    except Exception as e:
        print(f"❌ QAAgent failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator():
    """Test orchestrator"""
    print("\n=== Testing Orchestrator ===")
    try:
        from backend.app.crew_orchestrator import CrewOrchestrator
        
        orchestrator = CrewOrchestrator(use_crew_sdk=False)
        
        # Test query
        query = "What is management?"
        print(f"Testing query: {query}")
        
        result = orchestrator.run_query(query, top_k=3)
        print(f"Orchestrator result keys: {result.keys()}")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        print("✅ Orchestrator working")
        return True
    except Exception as e:
        print(f"❌ Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_summary_endpoint():
    """Test summary endpoint"""
    print("\n=== Testing Summary Endpoint ===")
    try:
        import requests
        
        payload = {
            "query": "What is management?",
            "top_k": 3
        }
        
        response = requests.post("http://127.0.0.1:8000/summary", json=payload)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {data.keys()}")
            print(f"Summary points: {data.get('summaryPoints', [])}")
            print(f"Q&A pairs: {data.get('qa', [])}")
            print("✅ Summary endpoint working")
            return True
        else:
            print(f"❌ Summary endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Summary endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_faiss_data():
    """Check if FAISS index has data"""
    print("\n=== Checking FAISS Data ===")
    try:
        from backend.app.agents.config import FAISS_INDEX_PATH, METADATA_DB
        import json
        
        print(f"FAISS index path: {FAISS_INDEX_PATH}")
        print(f"Metadata path: {METADATA_DB}")
        
        if FAISS_INDEX_PATH.exists():
            print(f"✅ FAISS index file exists: {FAISS_INDEX_PATH.stat().st_size} bytes")
        else:
            print("❌ FAISS index file does not exist")
            
        if METADATA_DB.exists():
            with open(METADATA_DB, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata file exists with {len(metadata)} entries")
        else:
            print("❌ Metadata file does not exist")
            
        return True
    except Exception as e:
        print(f"❌ Error checking FAISS data: {e}")
        return False

def main():
    """Run all diagnostics"""
    print("Starting comprehensive error diagnosis...")
    print("=" * 60)
    
    tests = [
        test_embeddings,
        test_faiss_index,
        check_faiss_data,
        test_qa_agent_directly,
        test_orchestrator,
        test_summary_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
        print("-" * 40)
    
    print(f"\nDiagnosis Results: {passed}/{total} tests passed")
    
    if passed < total:
        print("❌ Some components are failing. Check the errors above.")
    else:
        print("✅ All components are working. The error might be elsewhere.")

if __name__ == "__main__":
    main()
