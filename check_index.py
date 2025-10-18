#!/usr/bin/env python3
"""
Check if FAISS index exists and has data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def check_faiss_index():
    """Check FAISS index status"""
    print("=== Checking FAISS Index Status ===")
    
    try:
        from backend.app.agents.config import FAISS_INDEX_PATH, METADATA_DB
        from backend.app.agents.faiss_index import FaissIndex
        import json
        
        print(f"FAISS index path: {FAISS_INDEX_PATH}")
        print(f"Metadata path: {METADATA_DB}")
        
        # Check if files exist
        index_exists = FAISS_INDEX_PATH.exists()
        metadata_exists = METADATA_DB.exists()
        
        print(f"Index file exists: {index_exists}")
        print(f"Metadata file exists: {metadata_exists}")
        
        if not index_exists or not metadata_exists:
            print("❌ FAISS index or metadata file missing")
            print("Solution: Run the build_index script to create the index")
            return False
        
        # Check metadata content
        with open(METADATA_DB, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata entries: {len(metadata)}")
        
        if len(metadata) == 0:
            print("❌ FAISS index is empty")
            print("Solution: Run the build_index script to populate the index")
            return False
        
        # Try to load the index
        try:
            index = FaissIndex.load(dim=384)  # all-MiniLM-L6-v2 dimension
            print(f"✅ FAISS index loaded successfully with {len(index.metadb)} entries")
            
            # Show sample entries
            if len(index.metadb) > 0:
                print("Sample entries:")
                for i, entry in enumerate(index.metadb[:3]):
                    print(f"  {i+1}. {entry.get('text', 'No text')[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Checking FAISS index status...")
    print("=" * 50)
    
    if check_faiss_index():
        print("\n✅ FAISS index is ready")
    else:
        print("\n❌ FAISS index needs to be built")
        print("\nTo fix this, run:")
        print("cd backend")
        print("python scripts/build_index.py")

if __name__ == "__main__":
    main()
