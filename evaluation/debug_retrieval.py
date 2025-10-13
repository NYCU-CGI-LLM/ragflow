#!/usr/bin/env python3
"""
Debug script to test retrieval and see what document IDs are returned
"""

import sys
from pathlib import Path

# Add SDK path
sdk_path = Path(__file__).parent.parent / "sdk" / "python"
sys.path.insert(0, str(sdk_path))

# Patch importlib for local dev
import importlib.metadata
original_version = importlib.metadata.version

def patched_version(distribution_name):
    if distribution_name == "ragflow_sdk":
        return "0.1.0-dev"
    return original_version(distribution_name)

importlib.metadata.version = patched_version

from ragflow_sdk import RAGFlow
importlib.metadata.version = original_version

# Configuration
API_KEY = "YOUR_API_KEY"
DATASET_ID = "YOUR_DATASET_ID"
BASE_URL = "http://localhost:9380"
TEST_QUERY = "患者临床信息"

def main():
    print("="*80)
    print("RAGFlow Retrieval Debugger")
    print("="*80)
    
    # Initialize client
    rag = RAGFlow(api_key=API_KEY, base_url=BASE_URL)
    print(f"\n✓ Initialized RAGFlow client")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Dataset ID: {DATASET_ID}")
    
    # Test retrieval
    print(f"\n📝 Test Query: {TEST_QUERY[:100]}...")
    print(f"\nRetrieving documents...")
    
    try:
        chunks = rag.retrieve(
            question=TEST_QUERY,
            dataset_ids=[DATASET_ID],
            page_size=10,
            similarity_threshold=0.1,  # Lower threshold to get more results
            vector_similarity_weight=0.3
        )
        
        print(f"\n✓ Retrieved {len(chunks)} chunks\n")
        
        if not chunks:
            print("❌ No chunks retrieved!")
            print("\nPossible issues:")
            print("  1. Dataset ID is wrong")
            print("  2. No documents in the dataset")
            print("  3. Documents not indexed yet")
            print("  4. Query doesn't match any documents")
            return
        
        # Display chunk details
        print("-"*80)
        print("Chunk Details:")
        print("-"*80)
        
        for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
            print(f"\nChunk {i}:")
            print(f"  Document ID: {getattr(chunk, 'document_id', 'N/A')}")
            print(f"  Document Name: {getattr(chunk, 'document_name', 'N/A')}")
            print(f"  Chunk ID: {getattr(chunk, 'id', 'N/A')}")
            print(f"  Similarity: {getattr(chunk, 'similarity', 'N/A'):.4f}")
            print(f"  Content: {getattr(chunk, 'content', '')[:100]}...")
            
            # Show all attributes
            if hasattr(chunk, '__dict__'):
                print(f"  All fields: {list(chunk.__dict__.keys())}")
        
        # Extract document IDs
        doc_ids = []
        for chunk in chunks:
            if hasattr(chunk, 'document_id'):
                doc_ids.append(int(chunk.document_id))
        
        # Remove duplicates and show
        unique_doc_ids = list(set(doc_ids))
        print(f"\n" + "="*80)
        print(f"📊 Summary:")
        print("="*80)
        print(f"Total chunks: {len(chunks)}")
        print(f"Unique document IDs: {unique_doc_ids}")
        print(f"Count: {len(unique_doc_ids)}")
        
        print(f"\n💡 Use these document IDs in your ground truth dataset!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n⚠️  Edit this script first:")
    print("  1. Set API_KEY")
    print("  2. Set DATASET_ID")
    print("  3. Optionally change TEST_QUERY\n")
    
    if API_KEY == "YOUR_API_KEY" or DATASET_ID == "YOUR_DATASET_ID":
        print("❌ Please edit the script with your actual API_KEY and DATASET_ID\n")
        sys.exit(1)
    
    main()

