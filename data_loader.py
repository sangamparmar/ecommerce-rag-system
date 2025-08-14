"""
Data loading and preprocessing utilities for the e-commerce recommendation system.
Handles loading product data from JSON/CSV files and preparing it for embedding generation.
"""

import json
import pandas as pd
from typing import List, Dict, Any
import os


class DataLoader:
    """Handles loading and preprocessing of product data."""
    
    def __init__(self, data_path: str = "products.json"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the product data file (JSON or CSV)
        """
        self.data_path = data_path
        self.products = []
        
    def load_products(self) -> List[Dict[str, Any]]:
        """
        Load products from the data file.
        
        Returns:
            List of product dictionaries
        """
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found. Creating sample data...")
            return self._create_sample_data()
            
        try:
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.products = json.load(f)
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                self.products = df.to_dict('records')
                # Convert specifications from string to dict if needed
                for product in self.products:
                    if isinstance(product.get('specifications'), str):
                        try:
                            product['specifications'] = json.loads(product['specifications'])
                        except json.JSONDecodeError:
                            product['specifications'] = {}
                    if isinstance(product.get('reviews'), str):
                        try:
                            product['reviews'] = json.loads(product['reviews'])
                        except json.JSONDecodeError:
                            product['reviews'] = []
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
                
            print(f"Loaded {len(self.products)} products from {self.data_path}")
            return self.products
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """
        Create sample product data if no data file is found.
        
        Returns:
            List of sample product dictionaries
        """
        sample_products = [
            {
                "product_id": "sample_001",
                "name": "Sample Laptop",
                "category": "Laptops",
                "description": "A sample laptop for demonstration purposes.",
                "specifications": {"processor": "Intel i5", "ram": "8GB", "storage": "256GB SSD"},
                "price": 699.99,
                "image_url": "https://example.com/sample.jpg",
                "reviews": ["Good laptop for the price.", "Works well for basic tasks."]
            }
        ]
        
        # Save sample data for future use
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(sample_products, f, indent=2)
            print(f"Created sample data file: {self.data_path}")
        except Exception as e:
            print(f"Could not save sample data: {e}")
            
        return sample_products
    
    def get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        """
        Get a specific product by its ID.
        
        Args:
            product_id: The product ID to search for
            
        Returns:
            Product dictionary or empty dict if not found
        """
        if not self.products:
            self.load_products()
            
        for product in self.products:
            if product.get('product_id') == product_id:
                return product
        return {}
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all products in a specific category.
        
        Args:
            category: Category name to filter by
            
        Returns:
            List of products in the specified category
        """
        if not self.products:
            self.load_products()
            
        return [product for product in self.products 
                if product.get('category', '').lower() == category.lower()]
    
    def get_all_categories(self) -> List[str]:
        """
        Get all unique product categories.
        
        Returns:
            List of unique category names
        """
        if not self.products:
            self.load_products()
            
        categories = set()
        for product in self.products:
            if 'category' in product:
                categories.add(product['category'])
        return list(categories)
    
    def get_price_range(self) -> tuple:
        """
        Get the price range of all products.
        
        Returns:
            Tuple of (min_price, max_price)
        """
        if not self.products:
            self.load_products()
            
        prices = [product.get('price', 0) for product in self.products if 'price' in product]
        if prices:
            return min(prices), max(prices)
        return 0, 0
    
    def prepare_text_for_embedding(self, product: Dict[str, Any]) -> str:
        """
        Prepare product text for embedding generation by combining all relevant fields.
        
        Args:
            product: Product dictionary
            
        Returns:
            Combined text string for embedding
        """
        text_parts = []
        
        # Add basic product info
        if 'name' in product:
            text_parts.append(f"Product: {product['name']}")
        
        if 'category' in product:
            text_parts.append(f"Category: {product['category']}")
            
        if 'description' in product:
            text_parts.append(f"Description: {product['description']}")
        
        # Add specifications
        if 'specifications' in product and isinstance(product['specifications'], dict):
            specs_text = []
            for key, value in product['specifications'].items():
                specs_text.append(f"{key}: {value}")
            if specs_text:
                text_parts.append(f"Specifications: {', '.join(specs_text)}")
        
        # Add review summaries (first few words of each review)
        if 'reviews' in product and isinstance(product['reviews'], list):
            review_snippets = []
            for review in product['reviews'][:3]:  # Limit to first 3 reviews
                if isinstance(review, str) and len(review.strip()) > 0:
                    # Take first 10 words of each review
                    words = review.strip().split()[:10]
                    review_snippets.append(' '.join(words))
            if review_snippets:
                text_parts.append(f"Reviews: {' | '.join(review_snippets)}")
        
        # Add price information
        if 'price' in product:
            text_parts.append(f"Price: ${product['price']}")
        
        return " | ".join(text_parts)
    
    def filter_products_by_price(self, min_price: float = 0, max_price: float = float('inf')) -> List[Dict[str, Any]]:
        """
        Filter products by price range.
        
        Args:
            min_price: Minimum price (inclusive)
            max_price: Maximum price (inclusive)
            
        Returns:
            List of products within the price range
        """
        if not self.products:
            self.load_products()
            
        filtered_products = []
        for product in self.products:
            price = product.get('price', 0)
            if min_price <= price <= max_price:
                filtered_products.append(product)
        
        return filtered_products
    
    def search_products_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search products by keyword in name, description, or specifications.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of products matching the keyword
        """
        if not self.products:
            self.load_products()
            
        keyword_lower = keyword.lower()
        matching_products = []
        
        for product in self.products:
            # Search in name
            if keyword_lower in product.get('name', '').lower():
                matching_products.append(product)
                continue
                
            # Search in description
            if keyword_lower in product.get('description', '').lower():
                matching_products.append(product)
                continue
                
            # Search in specifications
            specs = product.get('specifications', {})
            if isinstance(specs, dict):
                for key, value in specs.items():
                    if keyword_lower in str(key).lower() or keyword_lower in str(value).lower():
                        matching_products.append(product)
                        break
        
        return matching_products


def main():
    """Test the DataLoader functionality."""
    loader = DataLoader()
    products = loader.load_products()
    
    print(f"Loaded {len(products)} products")
    print(f"Categories: {loader.get_all_categories()}")
    print(f"Price range: ${loader.get_price_range()[0]:.2f} - ${loader.get_price_range()[1]:.2f}")
    
    # Test embedding text preparation
    if products:
        sample_product = products[0]
        embedding_text = loader.prepare_text_for_embedding(sample_product)
        print(f"\nSample embedding text:\n{embedding_text}")


if __name__ == "__main__":
    main()
