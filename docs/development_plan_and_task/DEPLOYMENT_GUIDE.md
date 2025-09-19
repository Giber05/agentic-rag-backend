# Vercel Deployment Guide

## Prerequisites

1. Create a Vercel account at [vercel.com](https://vercel.com)
2. Connect your GitHub account to Vercel
3. Install Vercel CLI: `npm install -g vercel`

## Deployment Steps

### 1. Login to Vercel

```bash
vercel login
```

### 2. Deploy the Backend

```bash
vercel --prod
```

### 3. Set Environment Variables

After deployment, you need to set these environment variables in your Vercel dashboard:

**Go to your project dashboard → Settings → Environment Variables**

#### Required Environment Variables:

```
# Database Configuration (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key_here
SUPABASE_SERVICE_KEY=your_supabase_key_here
DATABASE_URL=postgresql://postgres:your_password@your_project.supabase.co:5432/postgres

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=Agentic RAG AI Agent
VERSION=1.0.0
DEBUG=false

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production-make-it-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Agent Configuration
QUERY_REWRITER_ENABLED=true
CONTEXT_DECISION_ENABLED=true
SOURCE_RETRIEVAL_ENABLED=true
ANSWER_GENERATION_ENABLED=true
VALIDATION_REFINEMENT_ENABLED=true

# Performance Configuration
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
VECTOR_SEARCH_TIMEOUT=5
EMBEDDING_BATCH_SIZE=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Models
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.7
```

### 4. Update Flutter App

Once deployed, update your Flutter app's API client to use the Vercel URL:

```dart
// In mobile_app/lib/core/network/api_client.dart
static ApiClient get instance {
  _instance ??= ApiClient._internal(
    baseUrl: 'https://your-vercel-app-name.vercel.app',
    logger: Logger()
  );
  return _instance!;
}
```

## Alternative: Quick Deploy with GitHub

1. Push your code to GitHub
2. Go to Vercel dashboard
3. Click "New Project"
4. Import your GitHub repository
5. Vercel will automatically detect it's a Python project
6. Set the environment variables in the dashboard
7. Deploy!

## Testing the Deployment

Once deployed, test these endpoints:

1. Health check: `https://your-app.vercel.app/health`
2. API docs: `https://your-app.vercel.app/api/v1/docs`
3. RAG pipeline: `https://your-app.vercel.app/api/v1/rag/pipeline/status`

## Troubleshooting

### Common Issues:

1. **Build fails**: Check that all dependencies are in requirements.txt
2. **Environment variables not working**: Make sure they're set in Vercel dashboard
3. **Database connection fails**: Verify Supabase credentials
4. **OpenAI API fails**: Check API key is valid

### Logs:

View deployment logs in Vercel dashboard → Functions → View Function Logs
