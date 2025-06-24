#!/usr/bin/env python3
"""
Test Supabase connection with correct credentials
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_supabase_connection():
    """Test Supabase connection"""
    print("🔍 Testing Supabase Connection...")
    print("=" * 50)
    
    # Get credentials
    url = "https://zpwgvyfxvmhfayylwwmz.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpwd2d2eWZ4dm1oZmF5eWx3d216Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMzU0OTEsImV4cCI6MjA2MzkxMTQ5MX0.LyAdNg7hdye7uneh0mau972WpzzjRYh4b_5uXime16U"
    
    print(f"URL: {url}")
    print(f"Key: {key[:50]}...")
    
    try:
        # Create Supabase client with correct parameters for v2.8.1
        supabase: Client = create_client(url, key)
        print("✅ Supabase client created successfully")
        
        # Test basic connection by querying embeddings table
        response = supabase.table('embeddings').select('id').limit(1).execute()
        print(f"✅ Database connection successful")
        print(f"✅ Found {len(response.data)} embeddings in database")
        
        # Test the search_embeddings function
        print("\n🔍 Testing search_embeddings function...")
        
        # Get a sample embedding to test with
        embedding_response = supabase.table('embeddings').select('embedding').limit(1).execute()
        if embedding_response.data:
            sample_embedding = embedding_response.data[0]['embedding']
            print(f"✅ Got sample embedding with {len(sample_embedding)} dimensions")
            
            # Test the search function
            search_response = supabase.rpc(
                'search_embeddings',
                {
                    'query_embedding': sample_embedding,
                    'match_threshold': 0.5,
                    'match_count': 3,
                    'document_ids': None,
                    'filter_user_id': None  # Use filter_user_id to avoid ambiguity
                }
            ).execute()
            
            print(f"✅ Search function works! Found {len(search_response.data)} results")
            
            if search_response.data:
                for i, result in enumerate(search_response.data):
                    similarity = result.get('similarity', 0)
                    chunk_text = result.get('chunk_text', '')[:100] + "..."
                    print(f"  Result {i+1} (similarity: {similarity:.3f}): {chunk_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_env_credentials():
    """Test credentials from .env file"""
    print("\n🔍 Testing .env file credentials...")
    print("=" * 50)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    print(f"URL from .env: {url}")
    print(f"Key from .env: {key[:50] if key else 'None'}...")
    
    if not url or not key:
        print("❌ Missing credentials in .env file")
        return False
    
    try:
        supabase: Client = create_client(url, key)
        response = supabase.table('embeddings').select('id').limit(1).execute()
        print("✅ .env credentials work!")
        return True
    except Exception as e:
        print(f"❌ .env credentials failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Supabase Connection Test")
    
    # Test with correct credentials
    direct_success = test_supabase_connection()
    
    # Test with .env credentials
    env_success = test_env_credentials()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Direct credentials: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f".env credentials: {'✅ PASS' if env_success else '❌ FAIL'}")
    
    if direct_success and not env_success:
        print("\n💡 Recommendation: Update your .env file with the correct credentials")
    elif direct_success and env_success:
        print("\n🎉 All connections working! Your backend should work now.")
    else:
        print("\n🔧 Need to troubleshoot connection issues") 