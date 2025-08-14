"""
Google Gemini API client for generating recommendations and product comparisons.
Handles API calls to Gemini 1.5 Flash for natural language generation tasks.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
import time
import json
from dotenv import load_dotenv


class GeminiClient:
    """Handles interactions with Google Gemini API for recommendation generation."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google Gemini API key (if None, loads from environment)
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env file")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model_name = 'gemini-1.5-flash'
        self.model = genai.GenerativeModel(self.model_name)
        
        # Generation config for consistent responses
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        print(f"Gemini client initialized with model: {self.model_name}")
    
    def generate_recommendations(self, 
                               retrieved_products: List[Dict[str, Any]], 
                               user_query: str,
                               user_preferences: Dict[str, Any] = None,
                               max_recommendations: int = 5) -> str:
        """
        Generate product recommendations based on retrieved products and user query.
        
        Args:
            retrieved_products: List of products from vector search
            user_query: Original user query
            user_preferences: User preference data for personalization
            max_recommendations: Maximum number of recommendations to generate
            
        Returns:
            Generated recommendation text
        """
        try:
            # Prepare product data for the prompt
            products_context = self._format_products_for_prompt(retrieved_products[:max_recommendations])
            
            # Prepare user context
            user_context = ""
            if user_preferences:
                user_context = self._format_user_preferences(user_preferences)
            
            # Create prompt
            prompt = f"""
You are an expert e-commerce product recommendation assistant. Based on the user's query and the retrieved product information, provide personalized product recommendations.

USER QUERY: "{user_query}"

{user_context}

RETRIEVED PRODUCTS:
{products_context}

INSTRUCTIONS:
1. Analyze the user's query to understand their needs and preferences
2. Recommend the most suitable products from the retrieved list
3. For each recommendation, explain why it matches the user's needs
4. Consider price, features, reviews, and user preferences
5. Provide recommendations in order of relevance
6. Keep each recommendation concise but informative
7. If applicable, mention any trade-offs or considerations

FORMAT YOUR RESPONSE AS:
## 🎯 Recommendations for "{user_query}"

### 1. [Product Name] - $[Price]
**Why it's perfect for you:** [Explanation]
**Key features:** [Bullet points]
**Customer sentiment:** [Brief review summary]

[Continue for other products...]

## 💡 Additional Tips
[Any helpful advice for the user's search]
"""

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text if response.text else "Sorry, I couldn't generate recommendations at this time."
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"
    
    def compare_products(self, products: List[Dict[str, Any]]) -> str:
        """
        Generate a detailed comparison of selected products.
        
        Args:
            products: List of products to compare
            
        Returns:
            Generated comparison text
        """
        try:
            if len(products) < 2:
                return "Please select at least 2 products to compare."
            
            if len(products) > 4:
                products = products[:4]  # Limit to 4 products for readability
            
            # Format products for comparison
            products_data = self._format_products_for_comparison(products)
            
            prompt = f"""
You are an expert product comparison analyst. Compare the following products in detail to help users make an informed decision.

PRODUCTS TO COMPARE:
{products_data}

INSTRUCTIONS:
1. Create a comprehensive side-by-side comparison
2. Highlight key differences and similarities
3. Identify the best use cases for each product
4. Consider value for money
5. Provide a clear recommendation based on different user types
6. Use tables and structured formatting for clarity

FORMAT YOUR RESPONSE AS:

## 📊 Product Comparison

### Quick Overview
[Brief summary table with key specs]

### Detailed Analysis

#### 💰 Price & Value
[Compare pricing and value proposition]

#### 🔧 Features & Specifications
[Detailed feature comparison]

#### ⭐ Customer Reviews & Ratings
[Compare review sentiment and satisfaction]

#### 🎯 Best For
- **[Product 1]**: [Ideal user type and use case]
- **[Product 2]**: [Ideal user type and use case]
[Continue for other products...]

#### ✅ Final Verdict
[Overall recommendation with reasoning]
"""

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text if response.text else "Sorry, I couldn't generate the comparison at this time."
            
        except Exception as e:
            print(f"Error generating comparison: {e}")
            return f"Error generating comparison: {str(e)}"
    
    def summarize_reviews(self, 
                         product_name: str, 
                         sentiment_summary: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of review sentiment.
        
        Args:
            product_name: Name of the product
            sentiment_summary: Sentiment analysis results
            
        Returns:
            Generated review summary
        """
        try:
            # Extract key information
            total_reviews = sentiment_summary.get('total_reviews', 0)
            valid_reviews = sentiment_summary.get('valid_reviews', 0)
            distribution = sentiment_summary.get('sentiment_distribution', {})
            overall_sentiment = sentiment_summary.get('overall_sentiment', 'neutral')
            
            # Get sample reviews if available
            detailed_results = sentiment_summary.get('detailed_results', [])
            sample_positive = []
            sample_negative = []
            
            for result in detailed_results[:10]:  # Limit to first 10 for performance
                if result['sentiment'] == 'positive' and len(sample_positive) < 3:
                    sample_positive.append(result.get('cleaned_text', ''))
                elif result['sentiment'] == 'negative' and len(sample_negative) < 3:
                    sample_negative.append(result.get('cleaned_text', ''))
            
            prompt = f"""
You are a product review analyst. Provide a natural language summary of customer sentiment for this product.

PRODUCT: {product_name}

REVIEW STATISTICS:
- Total reviews analyzed: {total_reviews}
- Valid reviews: {valid_reviews}
- Overall sentiment: {overall_sentiment}
- Positive: {distribution.get('positive', 0):.1f}%
- Neutral: {distribution.get('neutral', 0):.1f}%
- Negative: {distribution.get('negative', 0):.1f}%

SAMPLE POSITIVE REVIEWS:
{chr(10).join([f"- {review}" for review in sample_positive])}

SAMPLE NEGATIVE REVIEWS:
{chr(10).join([f"- {review}" for review in sample_negative])}

INSTRUCTIONS:
1. Provide a concise, informative summary of customer sentiment
2. Highlight what customers love about the product
3. Mention common concerns or complaints
4. Give an overall assessment of customer satisfaction
5. Keep it balanced and objective
6. Use a friendly, conversational tone

FORMAT AS:
## 📝 Customer Review Summary for {product_name}

[Your analysis here]
"""

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text if response.text else "Sorry, I couldn't generate the review summary at this time."
            
        except Exception as e:
            print(f"Error generating review summary: {e}")
            return f"Error generating review summary: {str(e)}"
    
    def generate_search_suggestions(self, partial_query: str, categories: List[str]) -> List[str]:
        """
        Generate search suggestions based on partial query and available categories.
        
        Args:
            partial_query: Partial user input
            categories: Available product categories
            
        Returns:
            List of suggested search queries
        """
        try:
            categories_text = ", ".join(categories)
            
            prompt = f"""
Generate 5 helpful search suggestions for an e-commerce product search based on the partial query.

PARTIAL QUERY: "{partial_query}"

AVAILABLE CATEGORIES: {categories_text}

INSTRUCTIONS:
1. Complete the partial query with relevant product searches
2. Make suggestions specific and actionable
3. Consider different price ranges and use cases
4. Include category-specific terms where relevant
5. Keep suggestions concise (5-8 words each)

RETURN ONLY A JSON LIST OF STRINGS:
["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4", "suggestion 5"]
"""

            response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.8, 'max_output_tokens': 200}
            )
            
            if response.text:
                try:
                    # Try to parse as JSON
                    suggestions = json.loads(response.text.strip())
                    if isinstance(suggestions, list):
                        return suggestions[:5]
                except json.JSONDecodeError:
                    # Fallback: extract suggestions from text
                    lines = response.text.strip().split('\n')
                    suggestions = []
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            clean_line = line.strip().strip('"-•').strip()
                            if clean_line:
                                suggestions.append(clean_line)
                    return suggestions[:5]
            
            return []
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []
    
    def _format_products_for_prompt(self, products: List[Dict[str, Any]]) -> str:
        """Format products for inclusion in prompts."""
        formatted_products = []
        
        for i, product in enumerate(products, 1):
            # Basic info
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            category = product.get('category', 'Unknown')
            description = product.get('description', 'No description available')
            
            # Specifications
            specs = product.get('specifications', {})
            spec_text = ""
            if isinstance(specs, dict):
                spec_items = [f"{k}: {v}" for k, v in specs.items()]
                spec_text = ", ".join(spec_items)
            
            # Reviews summary
            reviews = product.get('reviews', [])
            review_count = len(reviews) if isinstance(reviews, list) else 0
            
            # Similarity score if available
            similarity = product.get('similarity_score', 0)
            
            formatted_product = f"""
{i}. **{name}** (${price:.2f}) - {category}
   Description: {description}
   Specifications: {spec_text}
   Reviews: {review_count} customer reviews
   Relevance Score: {similarity:.3f}
"""
            formatted_products.append(formatted_product.strip())
        
        return "\n\n".join(formatted_products)
    
    def _format_products_for_comparison(self, products: List[Dict[str, Any]]) -> str:
        """Format products specifically for comparison prompts."""
        formatted_products = []
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown Product')
            price = product.get('price', 0)
            category = product.get('category', 'Unknown')
            description = product.get('description', 'No description available')
            
            # Format specifications as key-value pairs
            specs = product.get('specifications', {})
            spec_lines = []
            if isinstance(specs, dict):
                for key, value in specs.items():
                    spec_lines.append(f"   - {key}: {value}")
            
            spec_text = "\n".join(spec_lines) if spec_lines else "   - No specifications available"
            
            # Review summary
            reviews = product.get('reviews', [])
            review_count = len(reviews) if isinstance(reviews, list) else 0
            
            formatted_product = f"""
PRODUCT {i}: {name}
   Price: ${price:.2f}
   Category: {category}
   Description: {description}
   Specifications:
{spec_text}
   Customer Reviews: {review_count} reviews
"""
            formatted_products.append(formatted_product.strip())
        
        return "\n\n".join(formatted_products)
    
    def _format_user_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format user preferences for inclusion in prompts."""
        pref_parts = []
        
        if 'preferred_categories' in preferences:
            categories = preferences['preferred_categories']
            if categories:
                pref_parts.append(f"Preferred categories: {', '.join(categories)}")
        
        if 'price_preference' in preferences:
            price_pref = preferences['price_preference']
            pref_parts.append(f"Price preference: {price_pref}")
        
        if 'feature_preferences' in preferences:
            features = preferences['feature_preferences']
            if isinstance(features, dict) and features:
                feature_list = [f"{k}: {v}" for k, v in features.items()]
                pref_parts.append(f"Feature preferences: {', '.join(feature_list)}")
        
        if pref_parts:
            return f"USER PREFERENCES:\n{chr(10).join([f'- {pref}' for pref in pref_parts])}\n"
        
        return ""


def main():
    """Test the GeminiClient functionality."""
    try:
        print("Initializing Gemini client...")
        client = GeminiClient()
        
        # Test sample products
        sample_products = [
            {
                'product_id': 'test_1',
                'name': 'Gaming Laptop Pro',
                'price': 1299.99,
                'category': 'Laptops',
                'description': 'High-performance gaming laptop with RTX graphics',
                'specifications': {'processor': 'Intel i7', 'ram': '16GB', 'storage': '1TB SSD'},
                'reviews': ['Great laptop!', 'Fast and reliable', 'Perfect for gaming'],
                'similarity_score': 0.95
            },
            {
                'product_id': 'test_2', 
                'name': 'Budget Office Laptop',
                'price': 599.99,
                'category': 'Laptops',
                'description': 'Affordable laptop for office work and productivity',
                'specifications': {'processor': 'Intel i5', 'ram': '8GB', 'storage': '256GB SSD'},
                'reviews': ['Good value', 'Perfect for work', 'Reliable'],
                'similarity_score': 0.87
            }
        ]
        
        print("\n=== Testing Recommendation Generation ===")
        recommendations = client.generate_recommendations(
            retrieved_products=sample_products,
            user_query="laptop for gaming under $1500",
            user_preferences={'preferred_categories': ['Laptops'], 'price_preference': 'mid'}
        )
        print(recommendations)
        
        print("\n=== Testing Product Comparison ===")
        comparison = client.compare_products(sample_products)
        print(comparison)
        
        print("\n=== Testing Search Suggestions ===")
        suggestions = client.generate_search_suggestions("gaming", ["Laptops", "Gaming Accessories"])
        print(f"Suggestions: {suggestions}")
        
    except Exception as e:
        print(f"Error testing Gemini client: {e}")
        print("Make sure you have set GEMINI_API_KEY in your .env file")


if __name__ == "__main__":
    main()
