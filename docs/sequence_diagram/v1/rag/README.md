# RAG System Documentation Index

## 📚 Complete Documentation Suite

Welcome to the comprehensive documentation for the Retrieval-Augmented Generation (RAG) system. This documentation suite provides everything developers need to understand, work with, and extend the RAG pipeline.

## 🗂️ Documentation Structure

### 1. [RAG Process Overview](./rag_process_overview.md)

**High-level system architecture and component overview**

- System architecture and core components
- Pipeline variants (Optimized vs Full)
- Key features and capabilities
- Cost optimization strategies
- Performance characteristics

### 2. [API Request Flow](./api_request_flow.md)

**Detailed API endpoint analysis and request lifecycle**

- FastAPI endpoint structure
- Request/response models
- Dependency injection flow
- Error handling patterns
- Pipeline selection logic

### 3. [Sequence Diagrams](./sequence_diagrams.md)

**Visual representation of system interactions**

- Complete RAG process flow
- Dependency injection patterns
- Vector search detailed flow
- Pipeline comparison diagrams
- Streaming response flow
- Error handling sequences
- Performance monitoring flow

### 4. [Services and Models](./services_and_models.md)

**Comprehensive service and data model documentation**

- Core services architecture
- Service interfaces and methods
- Data models and schemas
- Service interaction patterns
- Configuration management
- Performance metrics

### 5. [Technical Implementation Guide](./technical_implementation_guide.md)

**Developer-focused implementation details**

- Quick start and setup
- Development patterns
- Performance optimization
- Testing strategies
- Deployment considerations
- Best practices

### 6. [Query Rewriter Documentation](./query_rewriter.md)

**Specialized documentation for query rewriting component**

- Query rewriting agent details
- Implementation specifics
- Usage patterns

## 🎯 Quick Navigation

### For New Developers

1. Start with [RAG Process Overview](./rag_process_overview.md) to understand the system
2. Review [API Request Flow](./api_request_flow.md) to understand the endpoints
3. Study [Sequence Diagrams](./sequence_diagrams.md) for visual understanding
4. Follow [Technical Implementation Guide](./technical_implementation_guide.md) for setup

### For System Architects

1. [RAG Process Overview](./rag_process_overview.md) - Architecture decisions
2. [Sequence Diagrams](./sequence_diagrams.md) - System interactions
3. [Services and Models](./services_and_models.md) - Component details
4. [Technical Implementation Guide](./technical_implementation_guide.md) - Deployment patterns

### For API Consumers

1. [API Request Flow](./api_request_flow.md) - Endpoint documentation
2. [Services and Models](./services_and_models.md) - Request/response schemas
3. [Sequence Diagrams](./sequence_diagrams.md) - Expected flow patterns

### For Maintainers

1. [Technical Implementation Guide](./technical_implementation_guide.md) - Development patterns
2. [Services and Models](./services_and_models.md) - Service interfaces
3. [Sequence Diagrams](./sequence_diagrams.md) - Error handling flows

## 🔍 Key System Concepts

### Pipeline Variants

- **Optimized Pipeline**: Cost-efficient with 60-70% cost reduction
- **Full Pipeline**: Maximum accuracy with comprehensive processing

### Core Services

- **OpenAI Service**: AI model interactions with caching
- **Vector Search Service**: Semantic search against Supabase
- **Cache Service**: Multi-level caching strategy
- **Token Tracker**: Cost monitoring and optimization
- **Document Service**: Document processing and management

### Optimization Features

- **Aggressive Caching**: 24-hour response cache, 7-day embedding cache
- **Pattern Matching**: Immediate responses for simple queries
- **Smart Model Selection**: Dynamic model choice based on complexity
- **Context Decision**: AI-powered retrieval necessity determination

## 📊 System Performance

### Cost Optimization

- **Cache Hits**: Save ~$0.08 per cached response
- **Pattern Responses**: Save ~$0.06 for simple queries
- **Model Selection**: 60-70% cost reduction with optimized pipeline
- **Embedding Cache**: Save ~$0.0004 per cached embedding

### Response Times

- **Cached Responses**: ~0.01s
- **Pattern Responses**: ~0.05s
- **Full RAG Process**: ~1.5-3s
- **Vector Search**: ~0.3s

### Accuracy Metrics

- **Optimized Pipeline**: 85-90% accuracy
- **Full Pipeline**: 95-98% accuracy
- **Cache Hit Rate**: 70-85%
- **Context Decision Accuracy**: 92%

## 🛠️ Development Workflow

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone <repository>
cd backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run development server
uvicorn app.main:app --reload
```

### Testing the System

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Test specific component
pytest tests/unit/test_rag_pipeline.py
```

### Making API Requests

```bash
# Test optimized pipeline
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "conversation_history": []
  }'

# Test full pipeline
curl -X POST "http://localhost:8000/api/v1/rag/process?use_full_pipeline=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain deep learning architectures",
    "conversation_history": []
  }'
```

## 🔧 Configuration Reference

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional
REDIS_URL=redis://localhost:6379
OPENAI_MODEL_PRIMARY=gpt-3.5-turbo
OPENAI_MODEL_FALLBACK=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Pipeline Configuration

```python
# Optimized pipeline settings
CACHE_TTL_RESPONSES = 86400  # 24 hours
CACHE_TTL_EMBEDDINGS = 604800  # 7 days
SIMILARITY_THRESHOLD = 0.7
MAX_SEARCH_RESULTS = 10
RECENCY_BOOST_FACTOR = 0.1

# Performance settings
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3
```

## 🚨 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors

- **Rate Limits**: Check API quota and implement backoff
- **Invalid API Key**: Verify environment variable
- **Model Access**: Ensure model availability

#### 2. Supabase Connection Issues

- **Network Connectivity**: Check URL and firewall
- **Authentication**: Verify API key permissions
- **Vector Extension**: Ensure pgvector is enabled

#### 3. Cache Performance

- **Redis Connection**: Verify Redis server status
- **Memory Usage**: Monitor cache size and eviction
- **TTL Configuration**: Adjust cache expiration times

#### 4. Performance Issues

- **Slow Responses**: Check database indexes and query optimization
- **High Costs**: Review caching strategy and model selection
- **Memory Leaks**: Monitor service resource usage

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
uvicorn app.main:app --reload --log-level debug
```

## 📈 Monitoring and Analytics

### Key Metrics to Monitor

- **Request Volume**: Requests per minute/hour
- **Response Times**: Average and 95th percentile
- **Cost Tracking**: Token usage and API costs
- **Cache Performance**: Hit rates and eviction rates
- **Error Rates**: Failed requests and error types

### Health Check Endpoint

```bash
# Check system health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "checks": {
    "openai": "healthy",
    "supabase": "healthy",
    "cache": "healthy"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔄 System Updates and Maintenance

### Regular Maintenance Tasks

1. **Cache Cleanup**: Monitor and clean expired cache entries
2. **Database Optimization**: Update indexes and analyze query performance
3. **Cost Analysis**: Review token usage and optimize model selection
4. **Security Updates**: Keep dependencies and API keys current

### Deployment Updates

1. **Staging Testing**: Test changes in staging environment
2. **Gradual Rollout**: Deploy with feature flags
3. **Monitoring**: Watch metrics during deployment
4. **Rollback Plan**: Prepare rollback procedures

## 📞 Support and Contributing

### Getting Help

- Review this documentation suite
- Check existing issues and solutions
- Contact the development team
- Submit bug reports with detailed context

### Contributing

- Follow the development patterns in the technical guide
- Write tests for new features
- Update documentation for changes
- Follow code review processes

---

This documentation suite provides comprehensive coverage of the RAG system, enabling developers to effectively understand, use, and extend the platform. Each document builds upon the others to create a complete picture of the system architecture and implementation.
