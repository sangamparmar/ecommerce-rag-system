"""
Streamlit Cloud Compatible Embeddings Module
Fallback implementation without ChromaDB for SQLite compatibility
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from data_loader import DataLoader

class EmbeddingManager:
    """Lightweight embedding manager for Streamlit Cloud (no ChromaDB)."""
    
    def __init__(self, model_name: str = "simple", db_path: str = "./fallback_db"):
        """Initialize with fallback storage only."""
        self.model_name = model_name
        self.db_path = db_path
        self.chromadb_available = False
        
        # Initialize fallback storage
        self.fallback_storage = {
            'embeddings': {},
            'metadata': {},
            'products': {}
        }
        print("✅ Fallback storage initialized (ChromaDB disabled for Streamlit Cloud)")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple word-based embeddings as fallback."""
        embeddings = []
        for text in texts:
            # Simple bag-of-words style embedding
            words = text.lower().split()
            # Create a simple feature vector based on word counts
            embedding = np.zeros(100)  # 100-dimensional vector
            for i, word in enumerate(words[:100]):
                embedding[i % 100] += hash(word) % 1000 / 1000.0
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def add_products_to_collection(self, products: List[Dict], force_update: bool = False):
        """Add products to fallback storage."""
        try:
            print(f"Adding {len(products)} products to fallback storage...")
            
            for product in products:
                product_id = product['product_id']
                
                # Create searchable text
                searchable_text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
                
                # Store in fallback
                self.fallback_storage['products'][product_id] = product
                self.fallback_storage['metadata'][product_id] = {
                    'text': searchable_text,
                    'category': product.get('category', ''),
                    'price': product.get('price', 0)
                }
            
            print(f"✅ Added {len(products)} products to fallback storage")
            
        except Exception as e:
            print(f"Error adding products: {e}")
    
    def search_similar_products(self, query: str, n_results: int = 5, 
                              category_filter: str = None, 
                              price_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """Search using text matching fallback."""
        try:
            # Load products if not in storage
            if not self.fallback_storage['products']:
                loader = DataLoader()
                products = loader.load_products()
                self.add_products_to_collection(products)
            
            query_lower = query.lower()
            matched_products = []
            
            for product_id, metadata in self.fallback_storage['metadata'].items():
                product = self.fallback_storage['products'][product_id]
                
                # Calculate similarity score
                text = metadata['text'].lower()
                score = 0
                query_words = query_lower.split()
                
                for word in query_words:
                    if word in text:
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
                        'product_id': product_id,
                        'product': product,
                        'score': score,
                        'distance': 1.0 - (score / len(query_words))
                    })
            
            # Sort by score and limit results
            matched_products.sort(key=lambda x: x['score'], reverse=True)
            matched_products = matched_products[:n_results]
            
            # Format results
            results = {
                'ids': [p['product_id'] for p in matched_products],
                'distances': [p['distance'] for p in matched_products],
                'metadatas': [p['product'] for p in matched_products],
                'total_found': len(matched_products),
                'query': query,
                'fallback_mode': True
            }
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return {
                'ids': [],
                'distances': [],
                'metadatas': [],
                'total_found': 0,
                'query': query,
                'error': str(e),
                'fallback_mode': True
            }
    
    def ensure_collection_exists(self):
        """Ensure storage is ready (fallback mode)."""
        if not self.fallback_storage['products']:
            loader = DataLoader()
            products = loader.load_products()
            self.add_products_to_collection(products)
    
    def get_collection_stats(self):
        """Get storage statistics."""
        return {
            'total_products': len(self.fallback_storage['products']),
            'storage_type': 'fallback_text_search',
            'chromadb_available': False,
            'note': 'Using text-based search due to Streamlit Cloud SQLite limitations'
        }
    
    def delete_collection(self):
        """Clear storage."""
        self.fallback_storage = {'embeddings': {}, 'metadata': {}, 'products': {}}
