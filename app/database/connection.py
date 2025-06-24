"""
Database connection and client management for Supabase.
"""

import asyncio
from typing import Optional
import asyncpg
from supabase import create_client, Client
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Global Supabase client instance
_supabase_client: Optional[Client] = None
_asyncpg_pool: Optional[asyncpg.Pool] = None


def get_supabase_client() -> Client:
    """
    Get or create Supabase client instance.
    
    Returns:
        Client: Configured Supabase client
        
    Raises:
        ValueError: If Supabase configuration is missing
    """
    global _supabase_client
    
    if _supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError(
                "Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_KEY"
            )
        
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
        
        logger.info("Supabase client initialized", url=settings.SUPABASE_URL)
    
    return _supabase_client


async def get_asyncpg_pool() -> asyncpg.Pool:
    """
    Get or create AsyncPG connection pool for direct PostgreSQL operations.
    
    Returns:
        asyncpg.Pool: Database connection pool
        
    Raises:
        ValueError: If database configuration is missing
    """
    global _asyncpg_pool
    
    if _asyncpg_pool is None:
        if not settings.DATABASE_URL:
            raise ValueError(
                "Database configuration missing. Please set DATABASE_URL"
            )
        
        try:
            _asyncpg_pool = await asyncpg.create_pool(
                settings.DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            logger.info("AsyncPG connection pool created")
            
        except Exception as e:
            logger.error("Failed to create database connection pool", error=str(e))
            raise
    
    return _asyncpg_pool


async def close_database_connections():
    """Close all database connections."""
    global _asyncpg_pool
    
    if _asyncpg_pool:
        await _asyncpg_pool.close()
        _asyncpg_pool = None
        logger.info("Database connections closed")


async def execute_migration(migration_sql: str) -> bool:
    """
    Execute a database migration.
    
    Args:
        migration_sql: SQL migration script
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        pool = await get_asyncpg_pool()
        async with pool.acquire() as connection:
            await connection.execute(migration_sql)
            
        logger.info("Migration executed successfully")
        return True
        
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        return False


async def test_database_connection() -> bool:
    """
    Test database connectivity.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Test Supabase client
        supabase = get_supabase_client()
        
        # Test AsyncPG pool
        pool = await get_asyncpg_pool()
        async with pool.acquire() as connection:
            result = await connection.fetchval("SELECT 1")
            
        logger.info("Database connection test successful")
        return True
        
    except Exception as e:
        logger.error("Database connection test failed", error=str(e))
        return False


async def check_vector_extension() -> bool:
    """
    Check if pgvector extension is available.
    
    Returns:
        bool: True if pgvector is available, False otherwise
    """
    try:
        pool = await get_asyncpg_pool()
        async with pool.acquire() as connection:
            result = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            
        if result:
            logger.info("pgvector extension is available")
        else:
            logger.warning("pgvector extension is not installed")
            
        return result
        
    except Exception as e:
        logger.error("Failed to check pgvector extension", error=str(e))
        return False 