-- Migration to add user_id column to documents table
-- This enables user-specific document ownership and filtering

-- Add user_id column to documents table
ALTER TABLE documents 
ADD COLUMN user_id UUID;

-- Create index for user_id for better query performance
CREATE INDEX idx_documents_user_id ON documents(user_id);

-- Update the RLS policy to include user_id filtering when authentication is implemented
-- For now, keep the permissive policy but prepare for future user-based restrictions
DROP POLICY IF EXISTS "Allow all operations on documents" ON documents;
CREATE POLICY "Allow all operations on documents" ON documents FOR ALL USING (true);

-- Add comment for future reference
COMMENT ON COLUMN documents.user_id IS 'User ID for document ownership - will be used when authentication is implemented'; 