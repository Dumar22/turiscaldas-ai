#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Building Alana Legal Sense for production..."

# Upgrade pip first
pip install --upgrade pip

# Try different requirements files - Render is using requirements.minimal.txt
if [ -f requirements.minimal.txt ]; then
    echo "📦 Installing minimal dependencies (Render default)..."
    pip install -r requirements.minimal.txt
elif [ -f requirements.production.txt ]; then
    echo "📦 Installing production dependencies..."
    pip install -r requirements.production.txt  
elif [ -f requirements.ultra-minimal.txt ]; then
    echo "📦 Installing ultra-minimal dependencies..."
    pip install -r requirements.ultra-minimal.txt
else
    echo "📦 Installing full dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p uploads
mkdir -p vector_db
mkdir -p chatbot/__pycache__

# Set proper permissions
chmod -R 755 uploads
chmod -R 755 vector_db

echo "✅ Build completed successfully!"