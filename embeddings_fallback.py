"""
Fallback embedding manager when ChromaDB is not available
"""

import json
from typing import List, Dict, Any, Tuple
from data_loader import DataLoader


class FallbackEmbeddingManager:
    """Fallback embedding manager using simple text matching when ChromaDB unavailable."""
    
    def __init__(self, model_name: str = "text-matching", db_path: str = None):
        """Initialize fallback embedding manager."""
        self.model_name = model_name
        self.chromadb_available = False
        self.fallback_mode = True
        
        # Load products for text search
        try:
            loader = DataLoader()
            self.products = loader.load_products()
        except Exception as e:
            print(f"Warning: Could not load products: {e}")
            self.products = []
        
        print("✅ Fallback embedding manager initialized (text-based search)")
    
    def search_similar_products(self, query: str, n_results: int = 5, 
                              category_filter: str = None, 
                              price_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """Search products using text matching."""
        query_lower = query.lower()
        matches = []
        
        print(f"🔍 Fallback search for: '{query}' in {len(self.products)} products")
        
        for product in self.products:
            # Create searchable text
            searchable_text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}".lower()
            
            # Calculate simple match score
            score = 0
            query_words = query_lower.split()
            
            # Check for exact phrase match (higher score)
            if query_lower in searchable_text:
                score += 5
            
            # Check for individual word matches
            for word in query_words:
                if word in searchable_text:
                    score += 1
            
            # Apply filters
            if category_filter and product.get('category') != category_filter:
                continue
            
            if price_range:
                price = product.get('price', 0)
                if not (price_range[0] <= price <= price_range[1]):
                    continue
            
            # Include products with any score > 0, or if no word matches, include random products
            if score > 0:
                matches.append({
                    'product': product,
                    'score': score,
                    'distance': max(0, 1.0 - (score / (len(query_words) + 5)))
                })
        
        # If no matches found, return some random products (to ensure we always have results)
        if not matches and not category_filter:
            print("⚠️ No matches found, returning random products")
            for product in self.products[:n_results]:
                matches.append({
                    'product': product,
                    'score': 0.1,
                    'distance': 0.9
                })
        
        # Sort by score and limit results
        matches.sort(key=lambda x: x['score'], reverse=True)
        matches = matches[:n_results]
        
        print(f"✅ Found {len(matches)} matches")
        
        # Format results
        return {
            'ids': [m['product']['product_id'] for m in matches],
            'metadatas': [m['product'] for m in matches],
            'distances': [m['distance'] for m in matches],
            'total_found': len(matches),
            'query': query,
            'fallback_mode': True
        }
    
    def add_products_to_collection(self, products: List[Dict], force_update: bool = False):
        """Add products to fallback storage."""
        self.products = products
        print(f"✅ Added {len(products)} products to fallback storage")
    
    def get_collection_stats(self):
        """Get collection statistics."""
        return {
            'total_products': len(self.products),
            'mode': 'fallback_text_search',
            'status': 'operational'
        }
    
    def ensure_collection_exists(self):
        """Ensure collection exists (no-op for fallback)."""
        pass
