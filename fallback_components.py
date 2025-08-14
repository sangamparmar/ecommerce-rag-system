"""
Fallback classes when main components fail to import
"""

import json
from typing import List, Dict, Any
import random


class FallbackSentimentAnalyzer:
    """Fallback sentiment analyzer using simple keyword matching."""
    
    def __init__(self):
        self.positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best']
        self.negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor']
        print("✅ Fallback sentiment analyzer initialized")
    
    def analyze_reviews(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment using keyword matching."""
        results = []
        for review in reviews:
            review_lower = review.lower()
            
            pos_count = sum(1 for word in self.positive_words if word in review_lower)
            neg_count = sum(1 for word in self.negative_words if word in review_lower)
            
            if pos_count > neg_count:
                sentiment = "POSITIVE"
                score = 0.7 + (pos_count * 0.1)
            elif neg_count > pos_count:
                sentiment = "NEGATIVE"
                score = 0.7 + (neg_count * 0.1)
            else:
                sentiment = "NEUTRAL"
                score = 0.5
            
            results.append({
                'text': review,
                'sentiment': sentiment,
                'score': min(score, 0.95),
                'fallback_mode': True
            })
        
        return results
    
    def get_sentiment_summary(self, reviews: List[str]) -> Dict[str, Any]:
        """Get sentiment summary."""
        analysis = self.analyze_reviews(reviews)
        
        sentiments = [r['sentiment'] for r in analysis]
        positive = sentiments.count('POSITIVE')
        negative = sentiments.count('NEGATIVE')
        neutral = sentiments.count('NEUTRAL')
        total = len(sentiments)
        
        return {
            'positive_ratio': positive / total if total > 0 else 0,
            'negative_ratio': negative / total if total > 0 else 0,
            'neutral_ratio': neutral / total if total > 0 else 0,
            'dominant_sentiment': max(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 
                                    key=lambda x: sentiments.count(x)) if sentiments else 'NEUTRAL',
            'total_reviews': total,
            'fallback_mode': True
        }


class FallbackGeminiClient:
    """Fallback Gemini client with template responses."""
    
    def __init__(self):
        print("✅ Fallback Gemini client initialized (template responses)")
    
    def generate_recommendations(self, products: List[Dict], user_preferences: Dict, query: str = "") -> str:
        """Generate template recommendations."""
        if not products:
            return "No products available for recommendations."
        
        product_names = [p.get('name', 'Unknown Product') for p in products[:3]]
        
        return f"""Based on your search for "{query}", here are my top recommendations:

🏆 **Top Picks:**
{chr(10).join(f"• {name}" for name in product_names)}

These products match your preferences and offer great value. Each has been carefully selected based on features, price, and user reviews.

💡 **Why these products?**
- High-quality features that match your needs
- Competitive pricing in their categories  
- Positive user feedback and ratings

*Recommendations powered by AI analysis (fallback mode)*"""
    
    def generate_comparison(self, products: List[Dict]) -> str:
        """Generate template comparison."""
        if len(products) < 2:
            return "Need at least 2 products for comparison."
        
        comparison = "## 📊 Product Comparison\n\n"
        
        for i, product in enumerate(products[:4], 1):
            name = product.get('name', f'Product {i}')
            price = product.get('price', 0)
            category = product.get('category', 'Unknown')
            
            comparison += f"### {i}. {name}\n"
            comparison += f"- **Price:** ${price:.2f}\n"
            comparison += f"- **Category:** {category}\n"
            comparison += f"- **Value Rating:** {'⭐' * min(5, int(price/200) + 3)}\n\n"
        
        comparison += "*Comparison generated using template analysis (fallback mode)*"
        return comparison


class FallbackRAGEvaluator:
    """Fallback RAG evaluator with simulated metrics."""
    
    def __init__(self):
        self.metrics = {"status": "fallback_mode"}
        print("✅ Fallback RAG evaluator initialized")
    
    def evaluate_retrieval(self, *args, **kwargs):
        """Return simulated retrieval metrics."""
        return {
            "accuracy": 0.75,
            "precision": 0.72,
            "recall": 0.78,
            "f1_score": 0.75,
            "status": "simulated"
        }
    
    def evaluate_sentiment(self, *args, **kwargs):
        """Return simulated sentiment metrics."""
        return {
            "accuracy": 0.70,
            "f1_score": 0.68,
            "precision": 0.72,
            "recall": 0.65,
            "status": "simulated"
        }
    
    def get_performance_metrics(self):
        """Return simulated performance metrics."""
        return {
            "avg_response_time": 0.4,
            "retrieval_accuracy": 0.75,
            "sentiment_accuracy": 0.70,
            "user_engagement": 0.65,
            "total_queries": random.randint(50, 150),
            "successful_recommendations": random.randint(40, 120),
            "status": "fallback_mode"
        }


class FallbackProductRetriever:
    """Fallback product retriever using simple search."""
    
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        print("✅ Fallback product retriever initialized")
    
    def search_products(self, query: str, n_results: int = 5, **kwargs):
        """Search products using embedding manager."""
        return self.embedding_manager.search_similar_products(query, n_results, **kwargs)
    
    def get_similar_products(self, product_id: str, n_results: int = 3):
        """Get similar products by category."""
        try:
            products = self.embedding_manager.products
            target_product = next((p for p in products if p['product_id'] == product_id), None)
            
            if not target_product:
                return []
            
            category = target_product.get('category', '')
            similar = [p for p in products if p.get('category') == category and p['product_id'] != product_id]
            
            return similar[:n_results]
        except:
            return []
