# Render Configuration for E-commerce RAG System

## Service Settings
- **Service Type**: Web Service
- **Environment**: Python 3.11
- **Region**: Choose closest to your users
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

## Environment Variables
```
GEMINI_API_KEY=your_gemini_api_key_here
PYTHONUNBUFFERED=1
```

## Instance Recommendations
- **Free Tier**: May work but slower builds
- **Starter ($7/month)**: Recommended for stable performance
- **Standard ($25/month)**: Best for production use

## Build Settings
- **Auto-Deploy**: Enabled
- **Health Check Path**: `/`
- **Deploy Timeout**: 20 minutes (for ML packages)

## Expected Build Time
- **First Deploy**: 10-15 minutes (downloading/compiling packages)
- **Subsequent Deploys**: 2-5 minutes (using cache)

## Troubleshooting
If build fails:
1. Check logs for specific package errors
2. Increase instance size if memory issues
3. Use fallback mode if ChromaDB compilation fails
