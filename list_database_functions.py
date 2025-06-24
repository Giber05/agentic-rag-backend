#!/usr/bin/env python3
"""
List all functions in the Supabase database
"""

from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

def list_database_functions():
    supabase = create_client(
        'https://zpwgvyfxvmhfayylwwmz.supabase.co',
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpwd2d2eWZ4dm1oZmF5eWx3d216Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMzU0OTEsImV4cCI6MjA2MzkxMTQ5MX0.LyAdNg7hdye7uneh0mau972WpzzjRYh4b_5uXime16U'
    )
    
    print("Listing all functions in the database...")
    
    # Query to list all functions
    try:
        # Query the information_schema to get function information
        result = supabase.rpc('pg_get_functiondef', {'function_oid': 'search_keywords'}).execute()
        print('Function definition:', result.data)
    except Exception as e:
        print(f'Error getting function definition: {e}')
        
    # Try to find search-related functions
    functions_to_check = [
        'search_embeddings',
        'search_keywords', 
        'search_hybrid',
        'match_documents',
        'vector_search'
    ]
    
    for func_name in functions_to_check:
        print(f"\nChecking function: {func_name}")
        try:
            # Try calling with minimal parameters to see what signature works
            result = supabase.rpc(func_name, {}).execute()
            print(f"✅ {func_name} exists and callable")
        except Exception as e:
            error_msg = str(e)
            if 'Could not find the function' in error_msg:
                print(f"❌ {func_name} does not exist")
            elif 'parameters' in error_msg or 'required' in error_msg:
                print(f"⚠️  {func_name} exists but needs parameters")
                print(f"   Error: {error_msg}")
            else:
                print(f"⚠️  {func_name} exists but has other issues")
                print(f"   Error: {error_msg}")

if __name__ == "__main__":
    list_database_functions() 