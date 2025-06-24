#!/bin/bash

# Railway Deployment Script for FastAPI Backend
# This script helps deploy the backend from a monorepo structure

echo "🚀 Deploying FastAPI Backend to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    curl -fsSL https://railway.app/install.sh | sh
    export PATH=$PATH:~/.railway/bin
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
railway whoami || railway login

# Link to project (you'll need to run this once)
echo "🔗 Linking to Railway project..."
railway link

# Set environment variables (example - replace with your actual values)
echo "⚙️  Setting environment variables..."
# Uncomment and modify these as needed:
# railway variables set SUPABASE_URL="your-supabase-url"
# railway variables set SUPABASE_KEY="your-supabase-key" 
# railway variables set OPENAI_API_KEY="your-openai-key"
# railway variables set DATABASE_URL="your-database-url"

# Deploy the application
echo "🚀 Deploying to Railway..."
railway deploy

echo "✅ Deployment complete!"
echo "📱 Your API should be available at your Railway domain"
echo "🔍 Check deployment status: railway status"
echo "📋 View logs: railway logs" 