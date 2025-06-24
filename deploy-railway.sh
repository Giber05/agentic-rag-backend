#!/bin/bash

# 🚀 Railway Deployment Script for Agentic RAG AI Agent Backend
# Production-ready deployment with best practices

set -e  # Exit on any error

echo "🚀 Starting Railway deployment for Agentic RAG AI Agent Backend..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Railway CLI is installed
print_status "Checking Railway CLI installation..."
if ! command -v railway &> /dev/null; then
    print_warning "Railway CLI not found. Installing..."
    curl -fsSL https://railway.app/install.sh | sh
    
    # Add to PATH for current session
    export PATH=$PATH:~/.railway/bin
    
    # Verify installation
    if ! command -v railway &> /dev/null; then
        print_error "Failed to install Railway CLI. Please install manually: https://docs.railway.app/develop/cli"
        exit 1
    fi
    print_success "Railway CLI installed successfully"
else
    print_success "Railway CLI found"
fi

# Check Railway authentication
print_status "Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    print_warning "Not logged in to Railway. Please authenticate..."
    railway login
    
    # Verify authentication
    if ! railway whoami &> /dev/null; then
        print_error "Railway authentication failed"
        exit 1
    fi
fi
print_success "Railway authentication verified"

# Link to Railway project
print_status "Linking to Railway project..."
if ! railway status &> /dev/null; then
    print_warning "Project not linked. Please link to your Railway project..."
    railway link
    
    # Verify link
    if ! railway status &> /dev/null; then
        print_error "Failed to link Railway project"
        exit 1
    fi
fi
print_success "Railway project linked"

# Check required environment variables
print_status "Checking required environment variables..."

required_vars=("SUPABASE_URL" "SUPABASE_KEY" "OPENAI_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if ! railway variables get "$var" &> /dev/null; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_error "Missing required environment variables: ${missing_vars[*]}"
    echo ""
    echo "Please set these variables using:"
    for var in "${missing_vars[@]}"; do
        echo "  railway variables set $var=\"your-value\""
    done
    echo ""
    print_warning "You can also set them through the Railway dashboard"
    echo ""
    read -p "Continue with deployment anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 1
    fi
else
    print_success "All required environment variables are set"
fi

# Set production environment variable
print_status "Setting production environment variables..."
railway variables set ENVIRONMENT="production"
railway variables set LOG_LEVEL="INFO"

# Optional: Set additional production variables
print_status "Setting optional production variables..."
if ! railway variables get SECRET_KEY &> /dev/null; then
    # Generate a random secret key
    SECRET_KEY=$(openssl rand -hex 32)
    railway variables set SECRET_KEY="$SECRET_KEY"
    print_success "Generated and set SECRET_KEY"
fi

# Build verification
print_status "Verifying project build requirements..."

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found in current directory"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory"
    exit 1
fi

# Check if main app exists
if [ ! -f "app/main.py" ]; then
    print_error "app/main.py not found"
    exit 1
fi

print_success "All required files found"

# Deploy the application
print_status "Starting deployment to Railway..."
echo "This may take several minutes..."

if railway deploy --detach; then
    print_success "Deployment initiated successfully!"
    
    # Wait a moment for deployment to start
    sleep 5
    
    # Show deployment status
    print_status "Checking deployment status..."
    railway status
    
    echo ""
    print_success "🎉 Deployment completed!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Check deployment status: railway status"
    echo "  2. View logs: railway logs"
    echo "  3. Open your app: railway open"
    echo "  4. Monitor health: Check /health endpoint"
    echo ""
    echo "🔗 Your API endpoints:"
    railway_url=$(railway status --json | grep -o '"url":"[^"]*' | cut -d'"' -f4 2>/dev/null || echo "Check 'railway open' for URL")
    if [ "$railway_url" != "Check 'railway open' for URL" ]; then
        echo "  • Health check: $railway_url/health"
        echo "  • API docs: $railway_url/api/v1/docs"
        echo "  • OpenAPI spec: $railway_url/api/v1/openapi.json"
    else
        echo "  • Run 'railway open' to get your app URL"
    fi
    
else
    print_error "Deployment failed!"
    echo ""
    echo "🔍 Troubleshooting steps:"
    echo "  1. Check logs: railway logs"
    echo "  2. Verify environment variables: railway variables"
    echo "  3. Check Railway service status: railway status"
    echo "  4. Review Dockerfile and requirements.txt"
    exit 1
fi 