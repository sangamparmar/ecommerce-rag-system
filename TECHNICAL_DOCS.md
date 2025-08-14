# 🛍️ E-commerce Product Recommendation RAG System - Technical Documentation

## Overview

This is a complete Python-based Retrieval-Augmented Generation (RAG) system for e-commerce product recommendations. The system combines vector search with AI-powered natural language generation to provide intelligent, personalized product recommendations.

## 🏗️ Architecture

### Core Components

1. **Data Layer** (`data_loader.py`)
   - Loads product data from JSON/CSV files
   - Handles data preprocessing and validation
   - Provides search and filtering utilities

2. **Vector Database** (`embeddings.py`)
   - Uses ChromaDB for persistent vector storage
   - Sentence Transformers for embedding generation
   - Handles similarity search and retrieval

3. **Retrieval Engine** (`retriever.py`)
   - Combines vector search with business logic
   - Implements personalization and ranking
   - Handles query processing and result filtering

4. **Sentiment Analysis** (`sentiment.py`)
   - Analyzes product review sentiment
   - Filters spam and low-quality reviews
   - Provides sentiment summaries

5. **AI Generation** (`gemini_client.py`)
   - Integrates with Google Gemini 1.5 Flash
   - Generates personalized recommendations
   - Creates product comparisons and summaries

6. **Web Interface** (`app.py`)
   - Streamlit-based user interface
   - Three main pages: Recommendations, Comparison, Review Explorer
   - Session-based personalization

## 🔄 RAG Pipeline Flow

```
User Query → Embedding Generation → Vector Search → Product Retrieval → 
Context Assembly → Gemini Generation → Response Display
```

### Detailed Flow

1. **Query Processing**
   - User enters natural language query
   - Query is converted to embedding using Sentence Transformers
   - Additional filters (category, price) are applied

2. **Vector Retrieval**
   - ChromaDB performs similarity search
   - Top-K most relevant products are retrieved
   - Results include similarity scores and metadata

3. **Context Enhancement**
   - Retrieved products are enriched with full data
   - User preferences are applied for personalization
   - Results are ranked using combined scoring

4. **AI Generation**
   - Product data is formatted for Gemini prompt
   - Context-aware prompt is constructed
   - Gemini generates personalized recommendations

5. **Response Delivery**
   - AI-generated text is displayed
   - Product cards show detailed information
   - User preferences are updated based on interaction

## 📊 Data Schema

### Product Data Structure

```json
{
  "product_id": "unique_identifier",
  "name": "Product Name",
  "category": "Product Category",
  "description": "Detailed description",
  "specifications": {
    "key1": "value1",
    "key2": "value2"
  },
  "price": 999.99,
  "image_url": "https://example.com/image.jpg",
  "reviews": [
    "Review text 1",
    "Review text 2"
  ]
}
```

### Vector Database Schema

- **Document**: Combined text from description, specs, and reviews
- **Metadata**: Flattened product information for filtering
- **Embeddings**: 384-dimensional vectors from all-MiniLM-L6-v2

## 🤖 AI Integration

### Gemini Prompts

#### Recommendation Generation
- Retrieves top products based on query
- Considers user preferences and search history
- Generates structured recommendations with explanations

#### Product Comparison
- Takes 2-4 products for side-by-side analysis
- Highlights key differences and similarities
- Provides purchase recommendations for different user types

#### Review Summarization
- Analyzes sentiment distribution
- Extracts key themes from reviews
- Generates natural language summaries

## 🎯 Personalization Features

### User Preference Tracking
- **Categories**: Tracks frequently searched categories
- **Price Range**: Learns budget preferences
- **Features**: Remembers important specifications

### Adaptive Recommendations
- Search history influences future recommendations
- Category preferences boost relevant products
- Price behavior affects value scoring

## 📈 Evaluation Metrics

### Retrieval Quality
- **Similarity Scores**: Cosine similarity from vector search
- **Relevance Assessment**: Manual evaluation of top results
- **Category Accuracy**: Percentage of results in expected category

### Performance Metrics
- **Latency**: End-to-end response time
- **Throughput**: Queries handled per second
- **Cache Hit Rate**: Embedding cache effectiveness

### User Experience
- **Session Length**: Time spent in application
- **Interaction Rate**: Clicks on recommended products
- **Comparison Usage**: Products added to comparison

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_api_key_here
```

### Model Configuration
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Sentiment Model**: cardiffnlp/twitter-roberta-base-sentiment-latest
- **LLM**: Gemini 1.5 Flash (free tier)

### Database Settings
- **ChromaDB**: Persistent storage in ./chroma_db
- **Collection**: Single collection with product embeddings
- **Batch Size**: 32 for embedding generation

## 🚀 Performance Optimization

### Caching Strategies
- **Product Cache**: In-memory storage of full product data
- **Embedding Cache**: ChromaDB persistent storage
- **Session Cache**: Streamlit session state for user data

### Batch Processing
- **Embedding Generation**: Batch processing for multiple products
- **Sentiment Analysis**: Parallel processing of reviews
- **Vector Search**: Optimized similarity search with filters

### Resource Management
- **Model Loading**: Lazy loading of heavy models
- **Memory Usage**: Efficient data structures and cleanup
- **API Limits**: Rate limiting for Gemini API calls

## 🛡️ Quality Assurance

### Data Validation
- **Review Filtering**: Removes spam and low-quality reviews
- **Input Sanitization**: Cleans and validates user inputs
- **Error Handling**: Graceful degradation on failures

### Sentiment Analysis Quality
- **Confidence Thresholds**: Filters low-confidence predictions
- **Spam Detection**: Multiple heuristics for fake reviews
- **Validation**: Cross-validation with manual labels

### AI Safety
- **Prompt Engineering**: Structured prompts prevent hallucination
- **Response Validation**: Basic checks on generated content
- **Fallback Responses**: Default responses when AI fails

## 🔄 Extension Points

### Adding New Data Sources
1. Extend `DataLoader` class with new format support
2. Update schema validation in `prepare_text_for_embedding`
3. Add new metadata fields to ChromaDB schema

### Custom Embedding Models
1. Modify `EmbeddingManager` to use different models
2. Update vector dimensions in ChromaDB configuration
3. Retrain/rebuild vector database with new embeddings

### Additional AI Providers
1. Create new client class following `GeminiClient` interface
2. Implement provider-specific prompt formatting
3. Add provider selection in configuration

### New Recommendation Types
1. Add new prompt templates in `GeminiClient`
2. Create corresponding UI components in `app.py`
3. Update retrieval logic for specific use cases

## 📚 Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Embedding generation
- **Transformers**: Sentiment analysis models
- **Google Generative AI**: Gemini API client

### Supporting Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Python-dotenv**: Environment variable management
- **Pillow**: Image processing for UI

## 🐛 Troubleshooting

### Common Issues

#### ChromaDB Connection Errors
- Check file permissions on ./chroma_db directory
- Ensure sufficient disk space for embeddings
- Verify ChromaDB version compatibility

#### Gemini API Errors
- Validate API key in .env file
- Check API quota and rate limits
- Ensure stable internet connection

#### Model Loading Failures
- Check available disk space for model downloads
- Verify internet connection for initial download
- Consider using local model cache

#### Performance Issues
- Monitor memory usage with large datasets
- Consider batch size reduction for embedding generation
- Optimize ChromaDB query parameters

### Debug Mode
Enable debugging by setting environment variables:
```env
STREAMLIT_DEBUG=true
CHROMADB_DEBUG=true
```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters
- Document all public methods and classes

### Testing
- Unit tests for individual components
- Integration tests for RAG pipeline
- Performance benchmarks for optimization

### Documentation
- Inline comments for complex logic
- Docstrings for all public functions
- README updates for new features

This system provides a solid foundation for building sophisticated e-commerce recommendation systems using modern RAG techniques. The modular architecture makes it easy to extend and customize for specific use cases.
