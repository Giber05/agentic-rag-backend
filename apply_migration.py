#!/usr/bin/env python3
"""
Script to apply the enhanced vector search migration to Supabase.
"""

import os
import sys
from app.core.config import settings
from supabase import create_client

def apply_migration():
    """Apply the enhanced vector search migration"""
    try:
        # Read migration file
        migration_path = 'migrations/003_enhanced_vector_search.sql'
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        print(f"Applying migration: {migration_path}")
        
        # Connect to Supabase
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Split migration into individual statements
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        print(f"Found {len(statements)} SQL statements to execute")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            try:
                print(f"Executing statement {i}/{len(statements)}...")
                
                # Use direct SQL execution for PostgreSQL functions
                result = supabase.postgrest.rpc('exec_sql', {'sql': statement + ';'}).execute()
                
                print(f"✅ Statement {i} executed successfully")
                
            except Exception as e:
                print(f"❌ Statement {i} failed: {e}")
                print(f"Statement was: {statement[:100]}...")
                
                # Continue with other statements
                continue
        
        print("🎉 Migration application completed!")
        
    except FileNotFoundError:
        print(f"❌ Migration file not found: {migration_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    apply_migration() 