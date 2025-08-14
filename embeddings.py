"""
Embedding generation and vector database management using ChromaDB and Sentence Transformers.
Handles creating embeddings for product data and storing them in a persistent ChromaDB collection.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# ChromaDB import with comprehensive error handling
CHROMADB_AVAILABLE = False
chromadb = None
Settings = None

def try_import_chromadb():
    """Attempt to import ChromaDB with error handling"""
    global CHROMADB_AVAILABLE, chromadb, Settings
    try:
        import chromadb as _chromadb
        from chromadb.config import Settings as _Settings
        chromadb = _chromadb
        Settings = _Settings
        CHROMADB_AVAILABLE = True
        return True
    except (ImportError, RuntimeError, Exception) as e:
        print(f"ChromaDB unavailable: {e}")
        CHROMADB_AVAILABLE = False
        return False

# Try to import ChromaDB, but don't fail if it doesn't work
try_import_chromadb()

# Safe imports
from sentence_transformers import SentenceTransformer
from data_loader import DataLoader


class EmbeddingManager:
    """Manages embeddings and ChromaDB operations for product data."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "./chroma_db"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model
            db_path: Path to store the ChromaDB database
        """
        self.model_name = model_name
        self.db_path = db_path
        self.model = None
        self.client = None
        self.collection = None
        self.collection_name = "product_embeddings"
        self.chromadb_available = CHROMADB_AVAILABLE
        
        # Initialize the embedding model
        self._load_model()
        
        # Initialize ChromaDB client with error handling
        if self.chromadb_available:
            self._init_chroma_client()
        else:
            print("⚠️ ChromaDB not available - using fallback storage")
            self._init_fallback_storage()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def _init_fallback_storage(self):
        """Initialize fallback storage when ChromaDB is not available."""
        self.fallback_storage = {
            'embeddings': {},
            'metadata': {},
            'products': {}
        }
        print("✅ Fallback storage initialized")
    
    def _init_chroma_client(self):
        """Initialize ChromaDB client and collection with SQLite compatibility check."""
        try:
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB not available")
            
            # Check SQLite version compatibility
            import sqlite3
            sqlite_version = sqlite3.sqlite_version_info
            required_version = (3, 35, 0)
            
            if sqlite_version < required_version:
                print(f"⚠️ SQLite version {sqlite3.sqlite_version} < 3.35.0 required by ChromaDB")
                print("🔄 Switching to fallback storage mode...")
                self.chromadb_available = False
                self._init_fallback_storage()
                return
                
            # Create the database directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"Loaded existing collection: {self.collection_name}")
            except Exception as e:
                # Collection doesn't exist, create it
                print(f"Collection not found ({e}), creating new collection...")
                try:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "E-commerce product embeddings"}
                    )
                    print(f"Created new collection: {self.collection_name}")
                except Exception as create_error:
                    print(f"Error creating collection: {create_error}")
                    # Try to get existing collections to debug
                    try:
                        existing_collections = self.client.list_collections()
                        print(f"Existing collections: {[c.name for c in existing_collections]}")
                        
                        # If our collection exists but get_collection failed, try to delete and recreate
                        for collection in existing_collections:
                            if collection.name == self.collection_name:
                                print(f"Found existing collection {self.collection_name}, deleting and recreating...")
                                self.client.delete_collection(name=self.collection_name)
                                break
                        
                        # Create fresh collection
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            metadata={"description": "E-commerce product embeddings"}
                        )
                        print(f"Successfully created fresh collection: {self.collection_name}")
                        
                    except Exception as final_error:
                        print(f"Final error in collection creation: {final_error}")
                        raise
                
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        if not self.model:
            self._load_model()
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def ensure_collection_exists(self):
        """Ensure the collection exists and is accessible."""
        if not self.collection:
            print("Collection not initialized, reinitializing...")
            self._init_chroma_client()
        
        # Test if collection is accessible
        try:
            if self.collection:
                self.collection.count()
        except Exception as e:
            print(f"Collection not accessible: {e}, reinitializing...")
            self._init_chroma_client()

    def add_products_to_collection(self, products: List[Dict[str, Any]], force_update: bool = False):
        """
        Add products to the ChromaDB collection with their embeddings.
        
        Args:
            products: List of product dictionaries
            force_update: Whether to update existing products
        """
        if not products:
            print("No products to add")
            return
        
        # Ensure collection exists
        self.ensure_collection_exists()
        
        # Check if collection already has data and skip if not forcing update
        if not force_update:
            try:
                existing_count = self.collection.count()
                if existing_count > 0:
                    print(f"Collection already contains {existing_count} products. Use force_update=True to refresh.")
                    return
            except Exception:
                pass
        
        print(f"Processing {len(products)} products for embedding...")
        
        # Prepare data for ChromaDB
        loader = DataLoader()
        ids = []
        documents = []
        metadatas = []
        
        for product in products:
            product_id = product.get('product_id', '')
            if not product_id:
                continue
                
            # Generate text for embedding
            document_text = loader.prepare_text_for_embedding(product)
            
            # Prepare metadata (ChromaDB doesn't support nested dicts, so flatten specifications)
            metadata = {
                'name': product.get('name', ''),
                'category': product.get('category', ''),
                'price': float(product.get('price', 0)),
                'description': product.get('description', '')[:500],  # Limit description length
            }
            
            # Flatten specifications into metadata
            specs = product.get('specifications', {})
            if isinstance(specs, dict):
                for key, value in specs.items():
                    # Clean key name for metadata
                    clean_key = f"spec_{key.replace(' ', '_').lower()}"
                    metadata[clean_key] = str(value)[:100]  # Limit length
            
            # Add review count
            reviews = product.get('reviews', [])
            metadata['review_count'] = len(reviews) if isinstance(reviews, list) else 0
            
            ids.append(product_id)
            documents.append(document_text)
            metadatas.append(metadata)
        
        try:
            # Clear collection if force update
            if force_update:
                # Delete all existing documents
                try:
                    existing_ids = self.collection.get()['ids']
                    if existing_ids:
                        self.collection.delete(ids=existing_ids)
                        print("Cleared existing collection data")
                except Exception as e:
                    print(f"Warning: Could not clear existing data: {e}")
            
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = self.generate_embeddings(documents)
            
            # Add to collection
            print("Adding to ChromaDB collection...")
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            
            print(f"Successfully added {len(products)} products to collection")
            
        except Exception as e:
            print(f"Error adding products to collection: {e}")
            raise
    
    def search_similar_products(self, query: str, n_results: int = 5, 
                              category_filter: str = None, 
                              price_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Search for similar products based on a query.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            category_filter: Optional category to filter by
            price_range: Optional price range (min, max) to filter by
            
        Returns:
            Dictionary with search results
        """
        # Check if ChromaDB is available
        if not self.chromadb_available:
            return self._fallback_search(query, n_results, category_filter, price_range)
        
        # Ensure collection exists
        self.ensure_collection_exists()
        
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0].tolist()
            
            # Prepare where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause['category'] = category_filter
            
            # Note: ChromaDB doesn't support range queries directly
            # We'll filter by price after retrieval if needed
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, 20),  # Get more results for filtering
                where=where_clause if where_clause else None
            )
            
            # Filter by price range if specified
            filtered_results = {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': []
            }
            
            for i, metadata in enumerate(results['metadatas'][0]):
                price = metadata.get('price', 0)
                
                # Apply price filter if specified
                if price_range and not (price_range[0] <= price <= price_range[1]):
                    continue
                
                # Add to filtered results
                filtered_results['ids'].append(results['ids'][0][i])
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['metadatas'].append(metadata)
                filtered_results['distances'].append(results['distances'][0][i])
                
                # Stop when we have enough results
                if len(filtered_results['ids']) >= n_results:
                    break
            
            return {
                'ids': filtered_results['ids'],
                'documents': filtered_results['documents'],
                'metadatas': filtered_results['metadatas'],
                'distances': filtered_results['distances'],
                'query': query,
                'total_found': len(filtered_results['ids'])
            }
            
        except Exception as e:
            print(f"Error searching products: {e}")
            return {
                'ids': [], 'documents': [], 'metadatas': [], 'distances': [],
                'query': query, 'total_found': 0
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {'error': 'Collection not initialized'}
        
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    'total_products': 0,
                    'categories': [],
                    'category_count': 0,
                    'price_range': (0, 0),
                    'model_name': self.model_name,
                    'collection_name': self.collection_name
                }
            
            # Get sample of metadata to understand categories and price ranges
            sample_data = self.collection.get(limit=min(count, 100))
            
            categories = set()
            prices = []
            
            for metadata in sample_data['metadatas']:
                if 'category' in metadata:
                    categories.add(metadata['category'])
                if 'price' in metadata:
                    try:
                        prices.append(float(metadata['price']))
                    except (ValueError, TypeError):
                        pass
            
            stats = {
                'total_products': count,
                'categories': list(categories),
                'category_count': len(categories),
                'price_range': (min(prices), max(prices)) if prices else (0, 0),
                'model_name': self.model_name,
                'collection_name': self.collection_name
            }
            
            return stats
            
        except Exception as e:
            return {'error': f'Error getting collection stats: {e}'}
    
    def delete_collection(self):
        """Delete the current collection."""
        try:
            if self.client and self.collection_name:
                self.client.delete_collection(name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
                self.collection = None
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def _fallback_search(self, query: str, n_results: int = 5, 
                        category_filter: str = None, 
                        price_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Fallback search method when ChromaDB is not available.
        Uses simple text matching and returns mock results.
        """
        try:
            # Load products for fallback search
            loader = DataLoader()
            products = loader.load_products()
            
            # Simple text matching
            query_lower = query.lower()
            matched_products = []
            
            for product in products:
                score = 0
                product_text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}".lower()
                
                # Simple scoring based on keyword matches
                query_words = query_lower.split()
                for word in query_words:
                    if word in product_text:
                        score += 1
                
                # Apply filters
                if category_filter and product.get('category') != category_filter:
                    continue
                    
                if price_range:
                    price = product.get('price', 0)
                    if not (price_range[0] <= price <= price_range[1]):
                        continue
                
                if score > 0:
                    matched_products.append({
                        'product': product,
                        'score': score,
                        'distance': 1.0 - (score / len(query_words))  # Convert to distance
                    })
            
            # Sort by score and limit results
            matched_products.sort(key=lambda x: x['score'], reverse=True)
            matched_products = matched_products[:n_results]
            
            # Format results to match ChromaDB output
            results = {
                'ids': [p['product']['product_id'] for p in matched_products],
                'distances': [p['distance'] for p in matched_products],
                'metadatas': [p['product'] for p in matched_products],
                'total_found': len(matched_products),
                'query': query,
                'fallback_mode': True
            }
            
            return results
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return {
                'ids': [],
                'distances': [],
                'metadatas': [],
                'total_found': 0,
                'query': query,
                'error': str(e),
                'fallback_mode': True
            }


def main():
    """Test the EmbeddingManager functionality."""
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Load products
    loader = DataLoader()
    products = loader.load_products()
    
    # Add products to collection
    print("Adding products to ChromaDB...")
    embedding_manager.add_products_to_collection(products, force_update=True)
    
    # Get collection stats
    stats = embedding_manager.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
    
    # Test search
    print("\nTesting search functionality...")
    test_queries = [
        "gaming laptop with good graphics",
        "budget smartphone under $500",
        "wireless headphones for music"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = embedding_manager.search_similar_products(query, n_results=3)
        
        print(f"Found {results['total_found']} results:")
        for i, (product_id, distance) in enumerate(zip(results['ids'], results['distances'])):
            metadata = results['metadatas'][i]
            print(f"  {i+1}. {metadata.get('name', 'Unknown')} - ${metadata.get('price', 0):.2f} (similarity: {1-distance:.3f})")


if __name__ == "__main__":
    main()
