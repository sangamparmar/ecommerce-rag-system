"""
Setup script for the E-commerce Product Recommendation RAG System.
This script helps set up the environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Please use Python 3.8+")
        return False

def install_dependencies():
    """Install required Python packages."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def setup_env_file():
    """Set up the .env file with API key."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔧 Setting up .env file...")
    
    # Get API key from user
    print("\n" + "="*60)
    print("🔑 GEMINI API KEY SETUP")
    print("="*60)
    print("To use this system, you need a Google Gemini API key.")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy the API key")
    print("="*60)
    
    api_key = input("\nPaste your Gemini API key here (or press Enter to skip): ").strip()
    
    if api_key and api_key != "your_gemini_api_key_here":
        env_content = f"GEMINI_API_KEY={api_key}\n"
        try:
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ .env file created with your API key")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        # Create placeholder .env file
        env_content = "GEMINI_API_KEY=your_gemini_api_key_here\n"
        try:
            with open(".env", "w") as f:
                f.write(env_content)
            print("⚠️  .env file created with placeholder. You'll need to add your API key later.")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

def test_installation():
    """Test if the installation was successful."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("  Testing imports...")
        import streamlit
        import pandas
        import sentence_transformers
        import chromadb
        import google.generativeai
        from dotenv import load_dotenv
        import transformers
        print("  ✅ All imports successful")
        
        # Test if models can be loaded
        print("  Testing model loading...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✅ Sentence transformer model loaded")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error during testing: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 E-commerce RAG System Setup")
    print("=" * 50)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Check Python version
    if check_python_version():
        success_count += 1
    
    # Step 2: Install dependencies
    if install_dependencies():
        success_count += 1
    
    # Step 3: Setup environment file
    if setup_env_file():
        success_count += 1
    
    # Step 4: Test installation
    if test_installation():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Setup Summary: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 Setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Make sure your Gemini API key is set in .env file")
        print("2. Run the test script: python test_system.py")
        print("3. Start the application: streamlit run app.py")
        
    elif success_count >= 2:
        print("⚠️  Setup mostly successful with some issues.")
        print("\n💡 To complete setup:")
        if success_count < 4:
            print("- Check error messages above and fix any issues")
        print("- Ensure your Gemini API key is correctly set in .env file")
        print("- Run: python test_system.py")
        
    else:
        print("❌ Setup failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("- Make sure you have Python 3.8+ installed")
        print("- Check your internet connection")
        print("- Try running: pip install --upgrade pip")
        print("- Install dependencies manually: pip install streamlit pandas sentence-transformers chromadb google-generativeai python-dotenv transformers pillow")

if __name__ == "__main__":
    main()
