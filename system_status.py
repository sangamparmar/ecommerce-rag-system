"""
System status checker for Streamlit deployment
"""

import sqlite3
import sys
import streamlit as st

def check_system_status():
    """Check and display system status"""
    status = {}
    
    # Check Python version
    status['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Check SQLite version
    try:
        sqlite_version = sqlite3.sqlite_version_info
        status['sqlite_version'] = f"{sqlite_version[0]}.{sqlite_version[1]}.{sqlite_version[2]}"
        status['sqlite_compatible'] = sqlite_version >= (3, 35, 0)
    except Exception as e:
        status['sqlite_version'] = f"Error: {e}"
        status['sqlite_compatible'] = False
    
    # Check ChromaDB
    try:
        import chromadb
        status['chromadb_available'] = True
        status['chromadb_version'] = chromadb.__version__
    except ImportError as e:
        status['chromadb_available'] = False
        status['chromadb_error'] = str(e)
    
    # Check other packages
    packages = ['sentence_transformers', 'google.generativeai', 'transformers']
    for package in packages:
        try:
            __import__(package)
            status[f'{package}_available'] = True
        except ImportError:
            status[f'{package}_available'] = False
    
    return status

def display_status_info():
    """Display system status in Streamlit"""
    status = check_system_status()
    
    with st.expander("🔧 System Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core System:**")
            st.write(f"Python: {status['python_version']}")
            st.write(f"SQLite: {status['sqlite_version']}")
            
            if status['sqlite_compatible']:
                st.success("✅ SQLite compatible with ChromaDB")
            else:
                st.warning("⚠️ SQLite version may cause ChromaDB issues")
        
        with col2:
            st.write("**AI Components:**")
            
            if status['chromadb_available']:
                st.success(f"✅ ChromaDB v{status.get('chromadb_version', 'unknown')}")
            else:
                st.error("❌ ChromaDB unavailable - using fallback")
            
            if status['sentence_transformers_available']:
                st.success("✅ Sentence Transformers")
            else:
                st.error("❌ Sentence Transformers")
            
            if status['google.generativeai_available']:
                st.success("✅ Google Gemini API")
            else:
                st.error("❌ Google Gemini API")
    
    return status
