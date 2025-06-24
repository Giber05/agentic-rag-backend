# Agentic RAG AI Agent - Backend API

Advanced Agentic Retrieval-Augmented Generation (RAG) AI agent backend featuring 5 specialized agents working in coordination.

## 🏗️ Architecture

The backend implements a FastAPI-based microservice architecture with:

- **Query Rewriting Agent** - Optimizes user queries for better retrieval
- **Context Decision Agent** - Determines if additional context is needed
- **Source Retrieval Agent** - Retrieves relevant information from vector database
- **Answer Generation Agent** - Generates responses using LLM with retrieved context
- **Validation & Refinement Agent** - Validates and iteratively improves responses

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or poetry
- Supabase account (for vector database)
- OpenAI API key

### Installation

1. **Clone and navigate to backend directory:**

   ```bash
   cd backend
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   ```bash
   cp env.example .env
   # Edit .env with your actual configuration values
   ```

5. **Run the development server:**

   ```bash
   python scripts/run_dev.py
   ```

6. **Test OpenAI Integration** (Optional):
   ```bash
   python test_openai.py
   ```

The API will be available at:

- **Server:** http://localhost:8000
- **API Docs:** http://localhost:8000/api/v1/docs
- **Health Check:** http://localhost:8000/health
- **OpenAI Usage Stats:** http://localhost:8000/api/v1/openai/usage/stats

## 📁 Project Structure

```
backend/
├── app/
│   ├── api/           # API route handlers
│   ├── core/          # Core configuration and utilities
│   ├── models/        # Pydantic data models
│   ├── services/      # Business logic services
│   └── utils/         # Utility functions
├── tests/             # Test files
├── scripts/           # Development scripts
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

#### Required Settings

```bash
# OpenAI API
OPENAI_API_KEY=your-openai-api-key

# Supabase Database
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-role-key
```

#### Optional Settings

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production

# Agent Configuration
QUERY_REWRITER_ENABLED=True
CONTEXT_DECISION_ENABLED=True
SOURCE_RETRIEVAL_ENABLED=True
ANSWER_GENERATION_ENABLED=True
VALIDATION_REFINEMENT_ENABLED=True

# Redis Configuration (Optional - will use in-memory cache if not available)
REDIS_URL=redis://localhost:6379

# OpenAI Service Settings (Optional - defaults are configured)
# OPENAI_MAX_RETRIES=3
# OPENAI_TIMEOUT=60
# OPENAI_DEFAULT_MODEL=gpt-4-turbo
# OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

## 📚 API Endpoints

### Health & Status

- `GET /health` - Health check
- `GET /api/v1/status` - Detailed API status

### OpenAI Integration (✅ Task 1.4 Complete)

- `POST /api/v1/openai/chat/completions` - Chat completions with GPT models
- `POST /api/v1/openai/embeddings` - Text embeddings generation
- `GET /api/v1/openai/usage/stats` - Usage statistics and metrics
- `GET /api/v1/openai/health` - OpenAI service health check
- `GET /api/v1/openai/models` - Available OpenAI models

### Coming Soon (Phase 2)

- `POST /api/v1/documents/upload` - Upload documents
- `POST /api/v1/search/semantic` - Semantic search
- `POST /api/v1/rag/process` - RAG pipeline processing
- `WS /api/v1/rag/stream` - Real-time streaming

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=app --cov-report=html
```

## 🔍 Development

### Code Quality

Format code with black:

```bash
black app/ tests/
```

Sort imports with isort:

```bash
isort app/ tests/
```

Lint with flake8:

```bash
flake8 app/ tests/
```

### Logging

The application uses structured logging with configurable output:

- JSON format for production
- Console format for development
- Configurable log levels

### Error Handling

- Structured error responses with request IDs
- Comprehensive exception handling
- Request/response logging for debugging

## 🚀 Deployment

### Docker (Coming Soon)

```bash
docker build -t agentic-rag-backend .
docker run -p 8000:8000 agentic-rag-backend
```

### Production Considerations

- Set `DEBUG=False`
- Use strong `SECRET_KEY`
- Configure proper CORS origins
- Set up Redis for caching
- Configure database connection pooling

## 📖 API Documentation

Once the server is running, visit:

- **Swagger UI:** http://localhost:8000/api/v1/docs
- **ReDoc:** http://localhost:8000/api/v1/redoc

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Follow Python PEP 8 style guidelines

## 📄 License

This project is part of the Agentic RAG AI Agent system.
# agentic-rag-ai-agent
