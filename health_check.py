#!/usr/bin/env python3
"""
Health check script for Render deployment
"""

import sys
import time
import requests
import os

def health_check():
    """Simple health check for the application"""
    try:
        # Check if environment variables are set
        if not os.getenv('GEMINI_API_KEY'):
            print("❌ GEMINI_API_KEY not found")
            return False
            
        # Basic imports test
        import streamlit
        import pandas
        import numpy
        print("✅ Core packages imported successfully")
        
        # Test custom modules
        from data_loader import DataLoader
        from embeddings import EmbeddingManager
        print("✅ Custom modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        print("🎉 Application health check passed!")
        sys.exit(0)
    else:
        print("💥 Application health check failed!")
        sys.exit(1)
