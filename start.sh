#!/bin/bash
# Render startup script

echo "🚀 Starting E-commerce RAG System..."
echo "📊 Using Streamlit on port $PORT"

# Run the Streamlit app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
