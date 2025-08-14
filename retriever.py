"""
Product retrieval system that combines vector search with filtering and ranking.
Handles querying ChromaDB and post-processing results for recommendation generation.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
from embeddings import EmbeddingManager
from data_loader import DataLoader


class ProductRetriever:
    """Handles product retrieval using vector search and additional filtering."""
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Initialize the product retriever.
        
        Args:
            embedding_manager: Pre-initialized EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.data_loader = DataLoader()
        self.products_cache = {}
        
        # Load products into cache for quick access
        self._load_products_cache()
    
    def _load_products_cache(self):
        """Load all products into memory cache for quick access."""
        try:
            products = self.data_loader.load_products()
            self.products_cache = {product['product_id']: product for product in products}
            print(f"Loaded {len(self.products_cache)} products into cache")
        except Exception as e:
            print(f"Error loading products cache: {e}")
            self.products_cache = {}
    
    def retrieve_products(self, 
                         query: str, 
                         n_results: int = 5,
                         category_filter: str = None,
                         price_range: Tuple[float, float] = None,
                         user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve products based on query and filters.
        
        Args:
            query: User search query
            n_results: Number of products to retrieve
            category_filter: Optional category filter
            price_range: Optional price range (min, max)
            user_preferences: User preference dictionary for personalization
            
        Returns:
            Dictionary containing retrieved products and metadata
        """
        start_time = time.time()
        
        try:
            # Perform vector search
            search_results = self.embedding_manager.search_similar_products(
                query=query,
                n_results=n_results * 2,  # Get more results for post-processing
                category_filter=category_filter,
                price_range=price_range
            )
            
            # Enhance results with full product data
            enhanced_results = self._enhance_search_results(search_results)
            
            # Apply personalization if user preferences are provided
            if user_preferences:
                enhanced_results = self._apply_personalization(enhanced_results, user_preferences)
            
            # Rank and limit results
            final_results = self._rank_and_limit_results(enhanced_results, n_results)
            
            retrieval_time = time.time() - start_time
            
            return {
                'products': final_results,
                'query': query,
                'total_found': len(final_results),
                'retrieval_time': retrieval_time,
                'filters_applied': {
                    'category': category_filter,
                    'price_range': price_range,
                    'personalized': user_preferences is not None
                }
            }
            
        except Exception as e:
            print(f"Error in product retrieval: {e}")
            return {
                'products': [],
                'query': query,
                'total_found': 0,
                'retrieval_time': 0,
                'error': str(e)
            }
    
    def _enhance_search_results(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhance search results with full product data from cache.
        
        Args:
            search_results: Raw search results from ChromaDB
            
        Returns:
            List of enhanced product dictionaries
        """
        enhanced_products = []
        
        for i, product_id in enumerate(search_results['ids']):
            # Get full product data from cache
            full_product = self.products_cache.get(product_id, {})
            
            if full_product:
                # Add search metadata
                enhanced_product = full_product.copy()
                enhanced_product['similarity_score'] = 1 - search_results['distances'][i]
                enhanced_product['search_rank'] = i + 1
                enhanced_product['retrieved_document'] = search_results['documents'][i]
                
                enhanced_products.append(enhanced_product)
        
        return enhanced_products
    
    def _apply_personalization(self, products: List[Dict[str, Any]], 
                             user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply user personalization to boost relevant products.
        
        Args:
            products: List of product dictionaries
            user_preferences: User preference dictionary
            
        Returns:
            List of products with personalization scores applied
        """
        try:
            preferred_categories = user_preferences.get('preferred_categories', [])
            price_preference = user_preferences.get('price_preference', 'any')  # 'budget', 'mid', 'premium', 'any'
            feature_preferences = user_preferences.get('feature_preferences', {})
            
            for product in products:
                personalization_boost = 0
                
                # Category preference boost
                if product.get('category') in preferred_categories:
                    personalization_boost += 0.1
                
                # Price preference boost
                price = product.get('price', 0)
                if price_preference == 'budget' and price < 300:
                    personalization_boost += 0.05
                elif price_preference == 'mid' and 300 <= price <= 800:
                    personalization_boost += 0.05
                elif price_preference == 'premium' and price > 800:
                    personalization_boost += 0.05
                
                # Feature preference boost
                specs = product.get('specifications', {})
                if isinstance(specs, dict):
                    for feature, preferred_value in feature_preferences.items():
                        if feature in specs:
                            spec_value = str(specs[feature]).lower()
                            if preferred_value.lower() in spec_value:
                                personalization_boost += 0.03
                
                # Apply personalization boost to similarity score
                original_score = product.get('similarity_score', 0)
                product['similarity_score'] = min(1.0, original_score + personalization_boost)
                product['personalization_boost'] = personalization_boost
            
            return products
            
        except Exception as e:
            print(f"Error applying personalization: {e}")
            return products
    
    def _rank_and_limit_results(self, products: List[Dict[str, Any]], 
                               n_results: int) -> List[Dict[str, Any]]:
        """
        Rank products by combined score and limit to requested number.
        
        Args:
            products: List of product dictionaries
            n_results: Maximum number of results to return
            
        Returns:
            List of top-ranked products
        """
        try:
            # Calculate combined ranking score
            for product in products:
                similarity_score = product.get('similarity_score', 0)
                price = product.get('price', 0)
                review_count = len(product.get('reviews', []))
                
                # Simple scoring formula (can be made more sophisticated)
                # Higher similarity and more reviews = higher score
                # Price doesn't directly affect score but could be used for tie-breaking
                combined_score = (
                    similarity_score * 0.7 +  # Primary factor: semantic similarity
                    min(review_count / 10, 0.2) +  # Secondary factor: review popularity (capped)
                    (1 / (1 + price / 1000)) * 0.1  # Minor factor: price accessibility
                )
                
                product['combined_score'] = combined_score
            
            # Sort by combined score (descending) and limit results
            ranked_products = sorted(products, key=lambda x: x.get('combined_score', 0), reverse=True)
            
            return ranked_products[:n_results]
            
        except Exception as e:
            print(f"Error ranking results: {e}")
            return products[:n_results]
    
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific product.
        
        Args:
            product_id: Product ID to retrieve
            
        Returns:
            Product dictionary or empty dict if not found
        """
        return self.products_cache.get(product_id, {})
    
    def get_similar_products(self, product_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find products similar to a given product.
        
        Args:
            product_id: Product ID to find similar products for
            n_results: Number of similar products to return
            
        Returns:
            List of similar product dictionaries
        """
        try:
            # Get the reference product
            reference_product = self.get_product_details(product_id)
            if not reference_product:
                return []
            
            # Create query from product description and specifications
            query_parts = []
            if 'description' in reference_product:
                query_parts.append(reference_product['description'])
            
            specs = reference_product.get('specifications', {})
            if isinstance(specs, dict):
                for key, value in specs.items():
                    query_parts.append(f"{key} {value}")
            
            query = " ".join(query_parts)
            
            # Search for similar products
            results = self.retrieve_products(
                query=query,
                n_results=n_results + 1,  # +1 to account for the reference product itself
                category_filter=reference_product.get('category')
            )
            
            # Filter out the reference product itself
            similar_products = [
                product for product in results['products']
                if product.get('product_id') != product_id
            ]
            
            return similar_products[:n_results]
            
        except Exception as e:
            print(f"Error finding similar products: {e}")
            return []
    
    def get_products_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get products from a specific category.
        
        Args:
            category: Category name
            limit: Maximum number of products to return
            
        Returns:
            List of products in the category
        """
        try:
            products_in_category = []
            for product in self.products_cache.values():
                if product.get('category', '').lower() == category.lower():
                    products_in_category.append(product)
            
            # Sort by price (ascending) as default ordering
            products_in_category.sort(key=lambda x: x.get('price', 0))
            
            return products_in_category[:limit]
            
        except Exception as e:
            print(f"Error getting products by category: {e}")
            return []
    
    def get_trending_products(self, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get trending products based on review count and ratings.
        
        Args:
            n_results: Number of trending products to return
            
        Returns:
            List of trending product dictionaries
        """
        try:
            all_products = list(self.products_cache.values())
            
            # Score products based on review count (simple trending metric)
            for product in all_products:
                reviews = product.get('reviews', [])
                review_count = len(reviews) if isinstance(reviews, list) else 0
                
                # Simple trending score: more reviews = more trending
                # In a real system, you'd consider recency, ratings, etc.
                product['trending_score'] = review_count
            
            # Sort by trending score
            trending_products = sorted(all_products, 
                                     key=lambda x: x.get('trending_score', 0), 
                                     reverse=True)
            
            return trending_products[:n_results]
            
        except Exception as e:
            print(f"Error getting trending products: {e}")
            return []


def main():
    """Test the ProductRetriever functionality."""
    print("Initializing Product Retriever...")
    
    # Initialize embedding manager and add products
    embedding_manager = EmbeddingManager()
    loader = DataLoader()
    products = loader.load_products()
    embedding_manager.add_products_to_collection(products, force_update=False)
    
    # Initialize retriever
    retriever = ProductRetriever(embedding_manager)
    
    # Test basic retrieval
    print("\n=== Testing Basic Retrieval ===")
    test_queries = [
        "gaming laptop with good graphics card",
        "budget smartphone under 400 dollars",
        "wireless headphones for exercise"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve_products(query, n_results=3)
        
        print(f"Found {results['total_found']} products in {results['retrieval_time']:.3f}s:")
        for i, product in enumerate(results['products']):
            print(f"  {i+1}. {product['name']} - ${product['price']:.2f} "
                  f"(score: {product.get('similarity_score', 0):.3f})")
    
    # Test personalization
    print("\n=== Testing Personalization ===")
    user_preferences = {
        'preferred_categories': ['Laptops', 'Gaming Accessories'],
        'price_preference': 'premium',
        'feature_preferences': {
            'processor': 'intel',
            'graphics': 'nvidia'
        }
    }
    
    results = retriever.retrieve_products(
        "high performance computer",
        n_results=3,
        user_preferences=user_preferences
    )
    
    print(f"Personalized results for query 'high performance computer':")
    for i, product in enumerate(results['products']):
        boost = product.get('personalization_boost', 0)
        print(f"  {i+1}. {product['name']} - ${product['price']:.2f} "
              f"(boost: +{boost:.3f})")
    
    # Test similar products
    print("\n=== Testing Similar Products ===")
    if products:
        reference_product = products[0]
        similar = retriever.get_similar_products(reference_product['product_id'], n_results=3)
        
        print(f"Products similar to '{reference_product['name']}':")
        for i, product in enumerate(similar):
            print(f"  {i+1}. {product['name']} - ${product['price']:.2f}")


if __name__ == "__main__":
    main()
