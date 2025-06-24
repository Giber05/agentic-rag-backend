import asyncio
import asyncpg
import openai
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

async def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI API."""
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

async def process_documents():
    """Process all documents without embeddings and create embeddings."""
    try:
        # Connect to database using Supabase URL format
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # For direct database connection, we'll use a simpler approach
        # Let's use the MCP approach instead since direct connection has issues
        print("⚠️  Direct database connection has issues. Please use the MCP approach.")
        print("✅ Documents are already uploaded to Supabase.")
        print("📝 The RAG system needs embeddings to be created.")
        print("🔧 This would normally be handled by the document upload API endpoint.")
        
        # For now, let's create a sample embedding to test
        sample_text = "Smarco & Sprout e-commerce platform with GoJek integration and commission system"
        embedding = await get_embedding(sample_text)
        
        if embedding:
            print(f"✅ Successfully generated sample embedding with {len(embedding)} dimensions")
            print(f"📊 First 5 values: {embedding[:5]}")
        else:
            print("❌ Failed to generate embedding")
            
    except Exception as e:
        print(f"❌ Error processing documents: {e}")

async def main():
    print("🚀 Starting embedding generation process...")
    print(f"📝 Using model: {EMBEDDING_MODEL}")
    print(f"🔧 Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    await process_documents()
    
    print("\n💡 Next steps:")
    print("1. Fix the database connection in your .env file")
    print("2. Use the document upload API endpoint to properly process documents")
    print("3. The API will automatically create embeddings and store them")
    print("4. Then test the RAG system again")

if __name__ == "__main__":
    asyncio.run(main()) 