#!/usr/bin/env python3
"""
Quick test to verify the FAISSAgent metadb fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_faiss_agent_metadb():
    """Test FAISSAgent metadb access"""
    print("=== Testing FAISSAgent metadb access ===")
    
    try:
        from backend.app.agents.agents import FAISSAgent
        
        # Create FAISSAgent
        faiss_agent = FAISSAgent(dim=384)
        
        # Test metadb access
        print(f"metadb length: {len(faiss_agent.idx.metadb)}")
        print(f"metadb type: {type(faiss_agent.idx.metadb)}")
        
        print("✅ FAISSAgent metadb access working")
        return True
        
    except Exception as e:
        print(f"❌ FAISSAgent metadb access failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_initialization():
    """Test orchestrator initialization"""
    print("\n=== Testing Orchestrator Initialization ===")
    
    try:
        from backend.app.crew_orchestrator import CrewOrchestrator
        
        # Initialize orchestrator
        orchestrator = CrewOrchestrator(use_crew_sdk=False)
        
        # Test metadb access
        print(f"FAISS metadb length: {len(orchestrator.faiss_agent.idx.metadb)}")
        
        print("✅ Orchestrator initialization working")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Testing FAISSAgent metadb fix...")
    print("=" * 50)
    
    tests = [
        test_faiss_agent_metadb,
        test_orchestrator_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed - fix is working!")
    else:
        print("❌ Some tests failed - check the errors above")

if __name__ == "__main__":
    main()
