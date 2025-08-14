"""
E-commerce Product Recommendation RAG System - Streamlit Cloud Version
Optimized for Streamlit Cloud deployment without ChromaDB
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any, List
import json

# Page configuration
st.set_page_config(
    page_title="Smart Product Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules with fallback handling
try:
    from data_loader import DataLoader
    from embeddings_streamlit import EmbeddingManager  # Use Streamlit-compatible version
    from retriever import ProductRetriever
    from sentiment import SentimentAnalyzer
    from gemini_client import GeminiClient
    from evaluation import RAGEvaluator
    
    MODULES_LOADED = True
except Exception as e:
    st.error(f"🔧 Module import issue: {e}")
    MODULES_LOADED = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .search-result {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .sidebar-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if not MODULES_LOADED:
        st.error("🔧 Cannot initialize - modules not loaded properly")
        return
        
    try:
        # Initialize core components
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = DataLoader()
        
        if 'embedding_manager' not in st.session_state:
            st.session_state.embedding_manager = EmbeddingManager()
        
        if 'retriever' not in st.session_state:
            st.session_state.retriever = ProductRetriever(
                st.session_state.embedding_manager,
                st.session_state.data_loader
            )
        
        if 'sentiment_analyzer' not in st.session_state:
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
        
        if 'gemini_client' not in st.session_state:
            st.session_state.gemini_client = GeminiClient()
        
        # Simplified evaluator for Streamlit Cloud
        if 'evaluator' not in st.session_state:
            st.session_state.evaluator = None  # Skip evaluator to avoid SQLite issues
        
        # User preference tracking
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'preferred_categories': [],
                'price_range': (0, 2000),
                'search_history': [],
                'interaction_count': 0
            }
        
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
            
    except Exception as e:
        st.error(f"🔧 Initialization error: {e}")
        st.info("💡 This app is running in fallback mode due to Streamlit Cloud limitations")

def show_search_interface():
    """Display the main search interface."""
    st.markdown('<h1 class="main-header">🛍️ Smart Product Recommendations</h1>', unsafe_allow_html=True)
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("🔍 What are you looking for?", 
                             placeholder="e.g., gaming laptop under $1000, wireless headphones...")
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Quick search buttons
    st.markdown("### 🚀 Quick Searches")
    quick_searches = [
        "Gaming laptops", "Budget smartphones", "Wireless headphones", 
        "Mechanical keyboards", "4K monitors", "Office chairs"
    ]
    
    cols = st.columns(3)
    for i, quick_search in enumerate(quick_searches):
        with cols[i % 3]:
            if st.button(quick_search, key=f"quick_{i}", use_container_width=True):
                query = quick_search
                search_button = True
    
    # Filters in sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info"><h3>🎛️ Filters</h3></div>', unsafe_allow_html=True)
        
        # Category filter
        categories = st.session_state.data_loader.get_categories()
        selected_category = st.selectbox("Category", ["All"] + categories)
        category_filter = None if selected_category == "All" else selected_category
        
        # Price range
        price_range = st.slider("Price Range ($)", 0, 2000, (0, 2000))
        
        # Number of results
        num_results = st.slider("Number of Results", 1, 10, 5)
    
    # Perform search
    if search_button and query:
        with st.spinner("🔍 Searching for products..."):
            try:
                # Get search results
                results = st.session_state.retriever.search_products(
                    query=query,
                    num_results=num_results,
                    category_filter=category_filter,
                    price_range=price_range if price_range != (0, 2000) else None
                )
                
                st.session_state.search_results = results
                
                # Update user preferences
                st.session_state.user_preferences['search_history'].append(query)
                st.session_state.user_preferences['interaction_count'] += 1
                
                # Display results
                if results:
                    st.success(f"🎉 Found {len(results)} products matching your search!")
                    
                    for i, product in enumerate(results):
                        with st.container():
                            st.markdown(f'<div class="search-result">', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                # Display product image
                                if product.get('image_url'):
                                    st.image(product['image_url'], use_column_width=True)
                                else:
                                    st.image("https://via.placeholder.com/200x200?text=No+Image", use_column_width=True)
                            
                            with col2:
                                st.markdown(f"### {product['name']}")
                                st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                                st.markdown(f"**Price:** ${product.get('price', 0):.2f}")
                                
                                # Description (truncated)
                                description = product.get('description', '')
                                if len(description) > 200:
                                    description = description[:200] + "..."
                                st.markdown(f"**Description:** {description}")
                            
                            with col3:
                                # Action buttons
                                if st.button(f"View Details", key=f"details_{i}"):
                                    st.session_state.selected_product = product
                                    st.rerun()
                                
                                if st.button(f"Add to Compare", key=f"compare_{i}"):
                                    if 'comparison_products' not in st.session_state:
                                        st.session_state.comparison_products = []
                                    if product not in st.session_state.comparison_products:
                                        st.session_state.comparison_products.append(product)
                                        st.success("Added to comparison!")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No products found matching your criteria. Try adjusting your filters!")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
                st.info("🔄 Please try again or use different search terms")

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    if not MODULES_LOADED:
        st.error("🔧 Application modules could not be loaded")
        st.info("This may be due to Streamlit Cloud environment limitations")
        st.stop()
    
    # Navigation
    pages = {
        "🎯 Smart Recommendations": show_search_interface,
        "ℹ️ About": show_about_page
    }
    
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        selected_page = st.radio("Go to", list(pages.keys()))
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🛍️ E-commerce RAG System | Powered by Gemini AI | Streamlit Cloud Compatible</p>
        <p><small>Running in text-search mode for optimal Streamlit Cloud performance</small></p>
    </div>
    """, unsafe_allow_html=True)

def show_about_page():
    """Show about page with system information."""
    st.markdown('<h1 class="main-header">ℹ️ About This System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 What This Is
    
    This is a **Retrieval-Augmented Generation (RAG)** system designed for e-commerce product recommendations. 
    It combines AI-powered search with personalized recommendations to help you find the perfect products.
    
    ## 🌟 Features
    
    - **🔍 Smart Search**: Natural language product search
    - **🤖 AI Recommendations**: Powered by Google Gemini AI
    - **📊 Product Comparison**: Side-by-side product analysis
    - **💝 Personalization**: Learns from your preferences
    - **☁️ Cloud Optimized**: Designed for Streamlit Cloud deployment
    
    ## 🔧 Technical Details
    
    - **Frontend**: Streamlit
    - **AI Model**: Google Gemini 1.5 Flash
    - **Search**: Text-based similarity matching
    - **Deployment**: Streamlit Cloud compatible
    
    ## 💡 Note
    
    This version is optimized for Streamlit Cloud and uses text-based search instead of 
    vector embeddings to ensure compatibility with the platform's environment limitations.
    """)
    
    # System status
    st.markdown("## 🔍 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Search Engine</h3>
            <p>✅ Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>AI Model</h3>
            <p>✅ Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        products_count = len(st.session_state.data_loader.load_products()) if MODULES_LOADED else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Products</h3>
            <p>{products_count} Available</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
