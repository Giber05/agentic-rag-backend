#!/usr/bin/env python3
"""
Basic test script for vector search functionality (non-OpenAI endpoints).
"""

import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_health_endpoint():
    """Test the search service health endpoint."""
    print("🔍 Testing search service health...")
    
    try:
        response = requests.get(f"{BASE_URL}/search/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Database: {data.get('database_connection', 'unknown')}")
            print(f"   OpenAI: {data.get('openai_connection', 'unknown')}")
            print(f"   Service: {data.get('service', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_analytics_endpoint():
    """Test the search analytics endpoint."""
    print("\n📊 Testing search analytics...")
    
    try:
        response = requests.get(f"{BASE_URL}/search/analytics")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics retrieved")
            print(f"   Total searches: {data.get('total_searches', 0)}")
            print(f"   Avg query time: {data.get('avg_query_time', 0.0):.3f}s")
            print(f"   Popular queries: {len(data.get('popular_queries', []))}")
            return True
        else:
            print(f"❌ Analytics failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return False

def test_suggestions_endpoint():
    """Test the search suggestions endpoint."""
    print("\n💡 Testing search suggestions...")
    
    try:
        response = requests.get(f"{BASE_URL}/search/suggestions?query=test&limit=5")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            suggestions = response.json()
            print(f"✅ Suggestions retrieved: {len(suggestions)} suggestions")
            if suggestions:
                print(f"   Suggestions: {suggestions}")
            else:
                print("   No suggestions (expected for new system)")
            return True
        else:
            print(f"❌ Suggestions failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Suggestions error: {e}")
        return False

def test_keyword_search_basic():
    """Test keyword search endpoint with basic functionality."""
    print("\n🔍 Testing keyword search (basic)...")
    
    search_request = {
        "query": "test query",
        "max_results": 5,
        "boost_recent": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/search/keyword",
            json=search_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        
        # We expect this to work now (even if no results)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Keyword search completed")
            print(f"   Query: {data.get('query', 'unknown')}")
            print(f"   Search type: {data.get('search_type', 'unknown')}")
            print(f"   Results: {len(data.get('results', []))}")
            print(f"   Query time: {data.get('query_time', 0.0):.3f}s")
            return True
        else:
            print(f"❌ Keyword search failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Keyword search error: {e}")
        return False

def main():
    """Run basic vector search tests."""
    print("🚀 Starting Basic Vector Search API Tests")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_analytics_endpoint,
        test_suggestions_endpoint,
        test_keyword_search_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(0.5)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Vector search API basic functionality is working.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 