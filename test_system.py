"""
Test script to verify the E-commerce RAG system components are working correctly.
Run this script to test individual components before running the main Streamlit app.
"""

import os
import sys
import time
from dotenv import load_dotenv

def test_data_loader():
    """Test the data loader functionality."""
    print("🔍 Testing Data Loader...")
    try:
        from data_loader import DataLoader
        loader = DataLoader()
        products = loader.load_products()
        print(f"✅ Loaded {len(products)} products")
        print(f"✅ Categories: {loader.get_all_categories()}")
        print(f"✅ Price range: ${loader.get_price_range()[0]:.2f} - ${loader.get_price_range()[1]:.2f}")
        return True
    except Exception as e:
        print(f"❌ Data Loader test failed: {e}")
        return False

def test_embeddings():
    """Test the embedding and ChromaDB functionality."""
    print("\n🔍 Testing Embeddings & ChromaDB...")
    try:
        from embeddings import EmbeddingManager
        from data_loader import DataLoader
        
        # Load data
        loader = DataLoader()
        products = loader.load_products()
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        print("✅ Embedding model loaded")
        
        # Add products to collection
        embedding_manager.add_products_to_collection(products[:3], force_update=True)  # Test with 3 products
        print("✅ Products added to ChromaDB")
        
        # Test search
        results = embedding_manager.search_similar_products("laptop", n_results=2)
        print(f"✅ Search test: Found {results['total_found']} results")
        
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test the sentiment analysis functionality."""
    print("\n🔍 Testing Sentiment Analysis...")
    try:
        from sentiment import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        print("✅ Sentiment model loaded")
        
        # Test with sample reviews
        test_reviews = [
            "This product is amazing! Great quality.",
            "Terrible quality, broke immediately.",
            "It's okay, nothing special."
        ]
        
        results = analyzer.analyze_reviews_batch(test_reviews)
        print(f"✅ Analyzed {len(results)} reviews")
        
        summary = analyzer.get_sentiment_summary(test_reviews)
        print(f"✅ Sentiment summary: {summary['overall_sentiment']}")
        
        return True
    except Exception as e:
        print(f"❌ Sentiment Analysis test failed: {e}")
        return False

def test_gemini_client():
    """Test the Gemini API client."""
    print("\n🔍 Testing Gemini API Client...")
    try:
        # Check if API key is available
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            print("⚠️  Gemini API key not configured. Please set GEMINI_API_KEY in .env file")
            return False
        
        from gemini_client import GeminiClient
        
        client = GeminiClient()
        print("✅ Gemini client initialized")
        
        # Test with simple recommendation
        sample_products = [{
            'product_id': 'test_1',
            'name': 'Test Laptop',
            'price': 999.99,
            'category': 'Laptops',
            'description': 'A test laptop for demonstration',
            'specifications': {'processor': 'Intel i5', 'ram': '8GB'},
            'reviews': ['Good laptop', 'Works well'],
            'similarity_score': 0.9
        }]
        
        recommendations = client.generate_recommendations(
            retrieved_products=sample_products,
            user_query="affordable laptop",
            max_recommendations=1
        )
        
        if recommendations and len(recommendations) > 50:  # Basic check for reasonable response
            print("✅ Gemini recommendations generated successfully")
            return True
        else:
            print("⚠️  Gemini response seems too short or empty")
            return False
            
    except Exception as e:
        print(f"❌ Gemini Client test failed: {e}")
        return False

def test_retriever():
    """Test the product retriever functionality."""
    print("\n🔍 Testing Product Retriever...")
    try:
        from retriever import ProductRetriever
        from embeddings import EmbeddingManager
        from data_loader import DataLoader
        
        # Initialize components
        loader = DataLoader()
        products = loader.load_products()
        
        embedding_manager = EmbeddingManager()
        embedding_manager.add_products_to_collection(products[:5], force_update=True)  # Test with 5 products
        
        retriever = ProductRetriever(embedding_manager)
        print("✅ Retriever initialized")
        
        # Test retrieval
        results = retriever.retrieve_products("gaming laptop", n_results=3)
        print(f"✅ Retrieved {results['total_found']} products in {results['retrieval_time']:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting E-commerce RAG System Tests")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Data Loader", test_data_loader),
        ("Embeddings & ChromaDB", test_embeddings),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Product Retriever", test_retriever),
        ("Gemini API Client", test_gemini_client),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
        if passed < 3:
            print("\n💡 Quick fixes:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Set your Gemini API key in .env file")
            print("3. Make sure you have internet connection for downloading models")

if __name__ == "__main__":
    main()
