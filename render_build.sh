# Render Deployment Script
#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y build-essential cmake pkg-config

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Deployment setup complete!"
