#!/usr/bin/env python3
"""
Test the metadb fix directly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_metadb_access():
    """Test metadb access directly"""
    print("=== Testing metadb access ===")
    
    try:
        from backend.app.agents.agents import FAISSAgent
        
        # Create FAISSAgent
        faiss_agent = FAISSAgent(dim=384)
        
        # Test the correct way to access metadb
        print(f"metadb length: {len(faiss_agent.idx.metadb)}")
        print("✅ Correct metadb access working")
        
        # Test the wrong way (should fail)
        try:
            print(f"Wrong access: {len(faiss_agent.metadb)}")
            print("❌ This should have failed!")
        except AttributeError as e:
            print(f"✅ Correctly caught error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_metadb_access()
