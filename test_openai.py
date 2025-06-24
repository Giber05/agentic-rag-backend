# !/usr/bin/env python3
"""
Test script for OpenAI integration.
Run this after setting up your OpenAI API key to verify the integration works.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.openai_service import get_openai_service
from app.core.config import settings


async def test_openai_integration():
    """Test OpenAI service integration."""
    print("🧪 Testing OpenAI Integration...")
    print(f"OpenAI API Key configured: {'✅' if settings.OPENAI_API_KEY else '❌'}")
    
    if not settings.OPENAI_API_KEY:
        print("❌ OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        return False
    
    try:
        # Test health check
        print("\n1. Testing health check...")
        health = await get_openai_service().health_check()
        print(f"Health status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
            return False
        
        # Test chat completion
        print("\n2. Testing chat completion...")
        messages = [
            {"role": "user", "content": "Say 'Hello from OpenAI integration test!'"}
        ]
        
        response = await get_openai_service().create_chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"Chat response: {response.choices[0].message.content}")
        
        # Test embedding
        print("\n3. Testing embedding generation...")
        embedding = await get_openai_service().create_embedding(
            text="This is a test embedding",
            model="text-embedding-ada-002"
        )
        
        print(f"Embedding generated: {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test usage stats
        print("\n4. Testing usage statistics...")
        stats = get_openai_service().get_usage_stats()
        print(f"Chat requests: {stats['chat_requests']}")
        print(f"Embedding requests: {stats['embedding_requests']}")
        print(f"Total tokens: {stats['total_tokens']}")
        
        print("\n✅ All OpenAI integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n🚦 Testing rate limiting...")
    
    try:
        from app.services.rate_limiter import rate_limiter
        
        # Get rate limiter stats
        chat_stats = rate_limiter.get_limiter_stats("chat")
        embedding_stats = rate_limiter.get_limiter_stats("embedding")
        
        print(f"Chat rate limiter: {chat_stats['available_slots']}/{chat_stats['max_requests']} slots available")
        print(f"Embedding rate limiter: {embedding_stats['available_slots']}/{embedding_stats['max_requests']} slots available")
        
        print("✅ Rate limiting system operational")
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False


async def test_caching():
    """Test caching functionality."""
    print("\n💾 Testing caching system...")
    
    try:
        from app.services.cache_service import cache_service
        
        # Test cache operations
        test_key = "test_key"
        test_value = {"test": "data", "number": 42}
        
        # Set value
        await cache_service.set(test_key, test_value, ttl=60)
        print("✅ Cache set operation successful")
        
        # Get value
        cached_value = await cache_service.get(test_key)
        if cached_value == test_value:
            print("✅ Cache get operation successful")
        else:
            print(f"❌ Cache get failed. Expected: {test_value}, Got: {cached_value}")
            return False
        
        # Delete value
        await cache_service.delete(test_key)
        deleted_value = await cache_service.get(test_key)
        if deleted_value is None:
            print("✅ Cache delete operation successful")
        else:
            print(f"❌ Cache delete failed. Value still exists: {deleted_value}")
            return False
        
        print("✅ Caching system operational")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting OpenAI Integration Tests\n")
    
    tests = [
        ("Rate Limiting", test_rate_limiting),
        ("Caching", test_caching),
        ("OpenAI Integration", test_openai_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        result = await test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! OpenAI integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 