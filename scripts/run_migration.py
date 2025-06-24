#!/usr/bin/env python3
"""
Database migration script for the Agentic RAG AI Agent backend.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import execute_migration
from app.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)


async def run_initial_migration():
    """Run the initial database migration."""
    try:
        # Read the migration file
        migration_file = Path(__file__).parent.parent / "app" / "database" / "migrations" / "001_initial_schema.sql"
        
        if not migration_file.exists():
            logger.error("Migration file not found", file=str(migration_file))
            return False
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        logger.info("Running initial database migration...")
        
        # Execute the migration
        success = await execute_migration(migration_sql)
        
        if success:
            logger.info("✅ Initial migration completed successfully!")
            print("✅ Database schema created successfully!")
            print("📊 Tables created:")
            print("   - documents (for storing uploaded documents)")
            print("   - embeddings (for vector search with pgvector)")
            print("   - conversations (for chat history)")
            print("   - messages (for conversation messages)")
            print("   - agent_logs (for agent performance tracking)")
            print()
            print("🔧 Extensions enabled:")
            print("   - vector (pgvector for semantic search)")
            print("   - uuid-ossp (for UUID generation)")
            print()
            print("🚀 Database is ready for the Agentic RAG AI Agent!")
        else:
            logger.error("❌ Migration failed!")
            print("❌ Migration failed! Check the logs for details.")
            return False
        
        return True
        
    except Exception as e:
        logger.error("Migration script failed", error=str(e))
        print(f"❌ Migration script failed: {str(e)}")
        return False


async def main():
    """Main function."""
    print("🔄 Starting database migration for Agentic RAG AI Agent...")
    print()
    
    success = await run_initial_migration()
    
    if success:
        print()
        print("🎉 Migration completed! You can now:")
        print("   1. Test database connectivity: curl http://localhost:8000/api/v1/database/status")
        print("   2. Create test documents via the API")
        print("   3. Start implementing the RAG pipeline")
        sys.exit(0)
    else:
        print()
        print("💡 Make sure you have:")
        print("   1. Supabase project created")
        print("   2. pgvector extension enabled in Supabase")
        print("   3. Correct DATABASE_URL in your .env file")
        print("   4. Network connectivity to your Supabase instance")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 