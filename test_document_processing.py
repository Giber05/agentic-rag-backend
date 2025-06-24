#!/usr/bin/env python3
"""
Test script for document processing functionality.
Tests document upload, processing, chunking, and storage.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

# Test document content
TEST_DOCUMENTS = {
    "sample.txt": """
# Introduction to Artificial Intelligence

Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. Some of the activities computers with artificial intelligence are designed for include:

- Speech recognition
- Learning
- Planning
- Problem solving

## History of AI

The field of AI research was born at a Dartmouth conference in 1956. AI research has gone through several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding.

## Types of AI

### Narrow AI
Narrow AI, also known as weak AI, is an AI system that is designed and trained for a particular task. Virtual personal assistants, such as Apple's Siri, are a form of narrow AI.

### General AI
General AI, also known as strong AI or artificial general intelligence (AGI), refers to a machine that has the ability to apply intelligence to any problem, rather than just one specific problem.

## Applications

AI is used in various fields including:
- Healthcare: Diagnosis and treatment recommendations
- Finance: Fraud detection and algorithmic trading
- Transportation: Autonomous vehicles
- Entertainment: Recommendation systems
- Education: Personalized learning platforms

## Conclusion

As AI continues to evolve, it promises to transform many aspects of our daily lives and work. However, it also raises important questions about ethics, employment, and the future of human-machine interaction.
""",
    
    "technical_doc.txt": """
# RAG System Architecture

## Overview
Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. This approach allows AI systems to access and utilize information beyond their training data.

## Components

### 1. Document Ingestion
- File upload and processing
- Text extraction from various formats (PDF, DOCX, TXT)
- Content cleaning and preprocessing

### 2. Text Chunking
- Semantic chunking strategies
- Overlap management
- Token count optimization

### 3. Embedding Generation
- Vector representation of text chunks
- Similarity search capabilities
- Efficient storage and retrieval

### 4. Query Processing
- Query understanding and rewriting
- Context determination
- Source retrieval optimization

### 5. Answer Generation
- Context-aware response generation
- Source attribution
- Quality validation

## Implementation Details

### Vector Database
Using Supabase with pgvector extension for:
- Storing document embeddings
- Performing similarity searches
- Managing metadata and relationships

### Chunking Strategy
- Maximum chunk size: 1000 tokens
- Overlap: 200 tokens
- Sentence boundary preservation
- Metadata preservation

### Search Algorithm
- Cosine similarity for vector matching
- Hybrid search combining semantic and keyword matching
- Result ranking and filtering

## Performance Considerations

### Scalability
- Horizontal scaling of vector operations
- Efficient indexing strategies
- Caching mechanisms

### Accuracy
- Embedding model selection
- Chunk size optimization
- Context window management

### Latency
- Parallel processing
- Result caching
- Connection pooling

## Security

### Data Protection
- Encryption at rest and in transit
- Access control and authentication
- Data retention policies

### Privacy
- User data isolation
- Anonymization techniques
- Compliance with regulations
"""
}

async def create_test_files() -> Dict[str, str]:
    """Create temporary test files and return their paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = {}
    
    for filename, content in TEST_DOCUMENTS.items():
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        file_paths[filename] = file_path
    
    print(f"📁 Created test files in: {temp_dir}")
    return file_paths

async def test_document_processor():
    """Test the document processor functionality."""
    print("🧪 Testing Document Processor...")
    
    try:
        from app.services.document_processor import DocumentProcessor
        from app.services.openai_service import OpenAIService
        from app.services.cache_service import CacheService
        from app.services.rate_limiter import RateLimiterService
        
        # Initialize services (without actual OpenAI calls)
        openai_service = OpenAIService()
        processor = DocumentProcessor(openai_service)
        
        # Create test files
        file_paths = await create_test_files()
        
        # Test each file
        for filename, file_path in file_paths.items():
            print(f"\n📄 Processing: {filename}")
            
            # Test file reading
            with open(file_path, 'rb') as f:
                file_content = f.read()
            content = await processor._extract_text_plain(file_content)
            print(f"   ✅ Content extracted: {len(content)} characters")
            
            # Test chunking (without embeddings)
            chunks = await processor._create_chunks(content, 1000, 200, filename)
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Display chunk info
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"   📝 Chunk {i+1}: {chunk.token_count} tokens, {len(chunk.text)} chars")
                if len(chunk.text) > 100:
                    print(f"      Preview: {chunk.text[:100]}...")
                else:
                    print(f"      Content: {chunk.text}")
        
        print("\n✅ Document processor tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test the API endpoints (without actual server)."""
    print("\n🌐 Testing API Endpoint Structure...")
    
    try:
        from app.api.v1.documents import router
        from app.models.document_models import DocumentResponse, SearchRequest
        
        # Test model validation
        search_request = SearchRequest(
            query="What is artificial intelligence?",
            limit=5,
            threshold=0.7
        )
        print(f"   ✅ SearchRequest model: {search_request.query}")
        
        # Check router endpoints
        routes = [route.path for route in router.routes]
        print(f"   ✅ Available endpoints: {routes}")
        
        print("✅ API endpoint structure tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_models():
    """Test the database models."""
    print("\n🗄️ Testing Database Models...")
    
    try:
        from app.models.document_models import DocumentChunk, ProcessingResult, DocumentResponse
        
        # Test DocumentChunk
        chunk = DocumentChunk(
            text="This is a test chunk",
            chunk_index=0,
            token_count=5,
            metadata={"source": "test"}
        )
        print(f"   ✅ DocumentChunk: {chunk.id[:8]}... ({chunk.token_count} tokens)")
        
        # Test ProcessingResult
        result = ProcessingResult(
            filename="test.txt",
            content_type="text/plain",
            text_content="This is test content",
            chunks=[chunk],
            metadata={"test": True}
        )
        print(f"   ✅ ProcessingResult: {result.filename} ({len(result.chunks)} chunks)")
        
        print("✅ Database model tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dependencies():
    """Test dependency injection."""
    print("\n🔧 Testing Dependencies...")
    
    try:
        from app.core.dependencies import get_cache_service, get_rate_limiter
        
        # Test cache service
        cache_service = get_cache_service()
        print(f"   ✅ Cache service: {type(cache_service).__name__}")
        
        # Test rate limiter
        rate_limiter = get_rate_limiter()
        print(f"   ✅ Rate limiter: {type(rate_limiter).__name__}")
        
        print("✅ Dependency injection tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Document Processing System Tests\n")
    
    tests = [
        ("Database Models", test_database_models),
        ("Dependencies", test_dependencies),
        ("Document Processor", test_document_processor),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        result = await test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Document processing system is ready.")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 