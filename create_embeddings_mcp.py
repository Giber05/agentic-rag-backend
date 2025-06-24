import openai
import os
import json
from dotenv import load_dotenv
from typing import List
from app.core.openai_config import OpenAIModels

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = OpenAIModels.TEXT_EMBEDDING_3_SMALL

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

def get_embedding(text: str) -> List[float]:
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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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

# Sample business requirements content
business_requirements = """Smarco & Sprout E-commerce Platform - Comprehensive Business Requirements

PROJECT OVERVIEW:
- E-commerce mobile application for Smarco & Sprout grocery/retail business
- Integration with existing Afari POS system (developed by Budi)
- Contract value: 1.2 million IDR
- 3-year maintenance contract: 3 million IDR/month starting January 14, 2020

KEY FEATURES:
1. PRODUCT MANAGEMENT SYSTEM:
   - Bulk product upload via Excel template with photo management
   - SKU-based photo naming convention (max 200KB per image)
   - Category mapping between Afari system and customer-facing categories
   - Product priority classification (Pareto vs non-Pareto items)

2. DELIVERY & LOGISTICS:
   - GoJek integration for 1-hour express delivery
   - Real-time order tracking for customers
   - Service charge configuration (percentage or fixed amount per item/brand)

3. COMMISSION & POINT SYSTEM:
   - Commission based on profit margin: 5% of gross profit
   - Point system for customer rewards (1 point = 1 IDR)
   - Special handling for loss-making products: 1% of loss amount

4. TECHNICAL INTEGRATION:
   - API integration with Budi's Afari POS system
   - VAT handling (PKP vs NPKP products)
   - In-store payment via app with barcode scanning"""

def main():
    print("🚀 Generating embeddings for Smarco & Sprout business requirements...")
    
    # Chunk the text
    chunks = chunk_text(business_requirements)
    print(f"📝 Created {len(chunks)} chunks from the business requirements")
    
    # Generate embeddings for each chunk
    embeddings_data = []
    
    for i, chunk in enumerate(chunks):
        print(f"🔄 Processing chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        
        if embedding:
            embeddings_data.append({
                'chunk_index': i,
                'content': chunk,
                'embedding': embedding,
                'embedding_dimensions': len(embedding)
            })
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
        else:
            print(f"❌ Failed to generate embedding for chunk {i+1}")
    
    print(f"\n🎉 Successfully generated {len(embeddings_data)} embeddings!")
    
    # Save embeddings to a JSON file for manual insertion
    with open('embeddings_data.json', 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print("💾 Embeddings saved to 'embeddings_data.json'")
    print("\n📋 Sample embedding data:")
    if embeddings_data:
        sample = embeddings_data[0]
        print(f"   Content preview: {sample['content'][:100]}...")
        print(f"   Embedding dimensions: {sample['embedding_dimensions']}")
        print(f"   First 5 embedding values: {sample['embedding'][:5]}")
    
    print("\n💡 Next steps:")
    print("1. Use the MCP tools to insert these embeddings into Supabase")
    print("2. Test the RAG system with queries about Smarco & Sprout")
    print("3. The system should now be able to retrieve relevant information")

if __name__ == "__main__":
    main() 