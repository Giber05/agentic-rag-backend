#!/usr/bin/env python3
"""
Direct RAG test script that bypasses Supabase client issues
"""

import asyncio
import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_query_embedding(query: str):
    """Generate embedding for the query"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def search_embeddings_direct(query_embedding, max_results=5):
    """Search embeddings using direct Supabase REST API"""
    try:
        # Use Supabase REST API to call the stored procedure
        url = f"{SUPABASE_URL}/rest/v1/rpc/search_embeddings"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use the correct parameter names that match the database function
        data = {
            "query_embedding": query_embedding,
            "match_threshold": 0.3,  # Changed from similarity_threshold
            "match_count": max_results,  # Changed from max_results
            "document_ids": None,
            "user_id": "test-user"  # Use string instead of null to resolve overloading
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error searching embeddings: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Error in direct search: {e}")
        return []

def generate_answer(query: str, sources: list):
    """Generate answer using OpenAI with retrieved sources"""
    try:
        # Prepare context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"Source {i}: {source.get('chunk_text', '')}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the user's question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def test_rag_pipeline(query: str):
    """Test the complete RAG pipeline"""
    print(f"\n🔍 Testing RAG Pipeline with query: '{query}'")
    print("=" * 60)
    
    # Step 1: Generate query embedding
    print("1. Generating query embedding...")
    query_embedding = get_query_embedding(query)
    if not query_embedding:
        print("❌ Failed to generate embedding")
        return
    print(f"✅ Generated embedding with {len(query_embedding)} dimensions")
    
    # Step 2: Search for relevant sources
    print("\n2. Searching for relevant sources...")
    sources = search_embeddings_direct(query_embedding)
    print(f"✅ Found {len(sources)} relevant sources")
    
    if sources:
        print("\nRetrieved sources:")
        for i, source in enumerate(sources, 1):
            similarity = source.get('similarity', 0)
            chunk_text = source.get('chunk_text', '')[:100] + "..."
            print(f"  Source {i} (similarity: {similarity:.3f}): {chunk_text}")
    
    # Step 3: Generate answer
    print("\n3. Generating answer...")
    answer = generate_answer(query, sources)
    print(f"✅ Generated answer")
    
    print(f"\n📝 Final Answer:")
    print("-" * 40)
    print(answer)
    print("-" * 40)
    
    return {
        "query": query,
        "sources_found": len(sources),
        "sources": sources,
        "answer": answer
    }

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is the commission system for Smarco and Sprout?",
        "Tell me about the GoJek delivery integration",
        "What are the main features of the e-commerce platform?",
        "What is the project timeline and budget?"
    ]
    
    print("🚀 Starting Direct RAG Pipeline Test")
    print("This bypasses the Supabase client and uses direct REST API calls")
    
    for query in test_queries:
        result = test_rag_pipeline(query)
        print("\n" + "="*80 + "\n")
    
    print("✅ RAG Pipeline test completed!") 