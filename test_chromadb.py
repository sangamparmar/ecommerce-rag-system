"""
Simple ChromaDB test to verify the database is working correctly.
"""

import chromadb
import os
import shutil

def test_chromadb():
    """Test basic ChromaDB functionality."""
    print("🔍 Testing ChromaDB...")
    
    # Clean up any existing test database
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=test_db_path)
        print("✅ ChromaDB client initialized")
        
        # Create collection
        collection = client.create_collection(
            name="test_collection",
            metadata={"description": "Test collection"}
        )
        print("✅ Collection created")
        
        # Add some test data
        collection.add(
            ids=["test1", "test2"],
            documents=["This is a test document", "Another test document"],
            metadatas=[{"category": "test"}, {"category": "test"}]
        )
        print("✅ Documents added")
        
        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        print(f"✅ Query successful: {results['ids']}")
        
        # Get collection stats
        count = collection.count()
        print(f"✅ Collection count: {count}")
        
        # Clean up
        client.delete_collection(name="test_collection")
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        print("✅ Test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        # Clean up on error
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        return False

if __name__ == "__main__":
    test_chromadb()
