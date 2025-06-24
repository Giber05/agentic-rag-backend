import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def insert_embedding_via_api(document_id, chunk_index, chunk_text, embedding, metadata):
    """Insert embedding using Supabase REST API."""
    url = f"{SUPABASE_URL}/rest/v1/embeddings"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    data = {
        'document_id': document_id,
        'chunk_index': chunk_index,
        'chunk_text': chunk_text,
        'embedding': embedding,
        'chunk_metadata': metadata
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response

def main():
    print("🚀 Inserting embeddings into Supabase...")
    
    # Load embeddings data
    with open('embeddings_data.json', 'r') as f:
        embeddings_data = json.load(f)
    
    document_id = '31c9d102-a111-40ce-a179-55af9819f62c'  # The document we uploaded
    
    for i, embedding_data in enumerate(embeddings_data):
        print(f"📝 Inserting embedding {i+1}/{len(embeddings_data)}...")
        
        metadata = {
            'chunk_size': 1000,
            'overlap': 200,
            'model': 'text-embedding-ada-002',
            'source': 'business_requirements'
        }
        
        response = insert_embedding_via_api(
            document_id=document_id,
            chunk_index=embedding_data['chunk_index'],
            chunk_text=embedding_data['content'],
            embedding=embedding_data['embedding'],
            metadata=metadata
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully inserted embedding {i+1}")
        else:
            print(f"❌ Failed to insert embedding {i+1}: {response.status_code}")
            print(f"   Error: {response.text}")
            break
    
    print("\n🎉 Embedding insertion complete!")
    print("💡 Now test the RAG system with queries about Smarco & Sprout")

if __name__ == "__main__":
    main() 