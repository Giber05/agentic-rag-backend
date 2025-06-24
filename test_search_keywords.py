#!/usr/bin/env python3
"""
Test search_keywords function signature
"""

from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

def test_search_keywords():
    supabase = create_client(
        'https://zpwgvyfxvmhfayylwwmz.supabase.co',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpwd2d2eWZ4dm1oZmF5eWx3d216Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMzU0OTEsImV4cCI6MjA2MzkxMTQ5MX0.LyAdNg7hdye7uneh0mau972WpzzjRYh4b_5uXime16U'
    )
    
    print("Testing search_keywords function...")
    
    # Test with the expected signature from migration
    try:
        result = supabase.rpc('search_keywords', {
            'search_query': 'test',
            'match_count': 1,
            'document_ids': None,
            'user_id': None
        }).execute()
        print('✅ Function call with migration signature successful')
        print(f'   Results: {len(result.data)} items')
    except Exception as e:
        print(f'❌ Function call with migration signature failed: {e}')
    
    # Test with the signature suggested by the error
    try:
        result = supabase.rpc('search_keywords', {
            'document_ids': None,
            'filter_user_id': None,
            'match_count': 1,
            'search_query': 'test'
        }).execute()
        print('✅ Function call with error suggestion signature successful')
        print(f'   Results: {len(result.data)} items')
    except Exception as e:
        print(f'❌ Function call with error suggestion signature failed: {e}')

if __name__ == "__main__":
    test_search_keywords() 