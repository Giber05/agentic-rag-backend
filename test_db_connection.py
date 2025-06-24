import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv('DATABASE_URL')

# Alternative direct connection
direct_url = "postgresql://postgres.zpwgvyfxvmhfayylwwmz:Kalijati123@db.zpwgvyfxvmhfayylwwmz.supabase.co:5432/postgres"

async def test_connection(url, name):
    try:
        print(f"Testing {name} connection to: {url}")
        conn = await asyncpg.connect(url)
        print(f'✅ {name} connection successful!')
        
        # Test pgvector extension
        result = await conn.fetchval("SELECT 1")
        print(f'✅ Query test successful: {result}')
        
        await conn.close()
        return True
    except Exception as e:
        print(f'❌ {name} connection failed: {e}')
        print(f'Error type: {type(e).__name__}')
        return False

async def main():
    print("Testing database connections...")
    
    # Test pooler connection
    success1 = await test_connection(database_url, "Pooler")
    
    print("\n" + "="*50 + "\n")
    
    # Test direct connection
    success2 = await test_connection(direct_url, "Direct")
    
    if not success1 and not success2:
        print("\n❌ Both connection methods failed. Please check credentials.")
    elif success2:
        print(f"\n✅ Use this URL in your .env file:")
        print(f"DATABASE_URL={direct_url}")

if __name__ == "__main__":
    asyncio.run(main()) 