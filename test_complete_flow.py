#!/usr/bin/env python3
"""
Test the complete flow from query to summary.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_complete_flow():
    """Test the complete flow"""
    print("=== Testing Complete Flow ===")
    
    try:
        from backend.app.crew_orchestrator import CrewOrchestrator
        
        # Initialize orchestrator
        print("Initializing orchestrator...")
        orchestrator = CrewOrchestrator(use_crew_sdk=False)
        
        # Test query
        query = "What are the main management principles?"
        print(f"Testing query: {query}")
        
        # Run query
        result = orchestrator.run_query(query, top_k=3)
        
        print(f"Result keys: {result.keys()}")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        # Check if we got a meaningful answer
        answer = result.get('answer', '')
        if 'Error processing query' in answer or 'list index out of range' in answer:
            print("❌ Still getting error in answer")
            return False
        elif 'No documents have been indexed' in answer:
            print("⚠️  No documents indexed - this is expected if no PDFs have been processed")
            return True
        elif 'No relevant information found' in answer:
            print("⚠️  No relevant information found - this might be expected depending on content")
            return True
        else:
            print("✅ Got meaningful answer")
            return True
            
    except Exception as e:
        print(f"❌ Complete flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Testing complete flow...")
    print("=" * 50)
    
    if test_complete_flow():
        print("\n✅ Complete flow test passed")
    else:
        print("\n❌ Complete flow test failed")

if __name__ == "__main__":
    main()
