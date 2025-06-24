# Supabase Setup & Vector Database Configuration

## ✅ Task 1.2 - COMPLETED

This document outlines the complete Supabase setup for the Agentic RAG AI Agent project.

## 🎯 Project Details

- **Project Name:** Agentic RAG AI Agent
- **Project ID:** `zpwgvyfxvmhfayylwwmz`
- **Region:** ap-southeast-1 (Singapore)
- **Status:** ACTIVE_HEALTHY
- **Database Host:** `db.zpwgvyfxvmhfayylwwmz.supabase.co`
- **API URL:** `https://zpwgvyfxvmhfayylwwmz.supabase.co`

## 🔧 Extensions Enabled

### ✅ pgvector (v0.8.0)

- **Purpose:** Vector similarity search and embeddings storage
- **Schema:** public
- **Features:** HNSW and IVFFlat indexes, cosine similarity operations

### ✅ uuid-ossp (v1.1)

- **Purpose:** UUID generation for primary keys
- **Schema:** extensions
- **Features:** `uuid_generate_v4()` function for automatic ID generation

## 📊 Database Schema

### Tables Created

#### 1. `documents`

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    file_type VARCHAR(50),
    file_size INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 2. `embeddings`

```sql
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI ada-002 embedding dimension
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 3. `conversations`

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID, -- Will be used when authentication is implemented
    title TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 4. `messages`

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    agent_data JSONB DEFAULT '{}', -- Store agent processing information
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### 5. `agent_logs`

```sql
CREATE TABLE agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    agent_type VARCHAR(50) NOT NULL,
    agent_input JSONB,
    agent_output JSONB,
    processing_time_ms INTEGER,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🚀 Performance Optimizations

### Indexes Created

#### Vector Search Index (HNSW)

```sql
CREATE INDEX idx_embeddings_vector ON embeddings
USING hnsw (embedding vector_cosine_ops);
```

#### Standard Indexes

```sql
-- Documents
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_file_type ON documents(file_type);

-- Embeddings
CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX idx_embeddings_chunk_index ON embeddings(chunk_index);

-- Conversations
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- Messages
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Agent Logs
CREATE INDEX idx_agent_logs_conversation_id ON agent_logs(conversation_id);
CREATE INDEX idx_agent_logs_agent_type ON agent_logs(agent_type);
CREATE INDEX idx_agent_logs_status ON agent_logs(status);
CREATE INDEX idx_agent_logs_created_at ON agent_logs(created_at);
```

## 🔒 Security Configuration

### Row Level Security (RLS)

- **Status:** Enabled on all tables
- **Current Policies:** Temporary "allow all" policies for development
- **Future:** Will be restricted when authentication is implemented

### Triggers

- **Updated At Trigger:** Automatically updates `updated_at` timestamp on record modifications

## 🔑 API Keys & Configuration

### Environment Variables

```bash
# Supabase Configuration
SUPABASE_URL=https://zpwgvyfxvmhfayylwwmz.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpwd2d2eWZ4dm1oZmF5eWx3d216Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzMzU0OTEsImV4cCI6MjA2MzkxMTQ5MX0.LyAdNg7hdye7uneh0mau972WpzzjRYh4b_5uXime16U
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.zpwgvyfxvmhfayylwwmz.supabase.co:5432/postgres
```

## ✅ Testing & Validation

### Tests Performed

1. **Extension Installation:** ✅ pgvector and uuid-ossp successfully installed
2. **Table Creation:** ✅ All 5 tables created with proper relationships
3. **Index Creation:** ✅ Vector and standard indexes operational
4. **Data Insertion:** ✅ Test records successfully inserted
5. **Vector Functionality:** ✅ 1536-dimensional vector support confirmed
6. **Foreign Key Constraints:** ✅ Relationships working correctly

### Sample Test Data

```sql
-- Test document
INSERT INTO documents (title, content, file_type)
VALUES ('Test Document', 'This is a test document for the Agentic RAG AI Agent system.', 'txt');

-- Test conversation
INSERT INTO conversations (title)
VALUES ('Test Conversation');

-- Test message
INSERT INTO messages (conversation_id, role, content)
VALUES ('4b58f4b2-915e-41a2-8734-67b5a9fc9410', 'user', 'Hello, this is a test message');

-- Test agent log
INSERT INTO agent_logs (conversation_id, message_id, agent_type, agent_input, agent_output, processing_time_ms, status)
VALUES ('4b58f4b2-915e-41a2-8734-67b5a9fc9410', '16777ab8-ad00-47aa-a3f4-19712deb6381', 'query_rewriter', '{"query": "Hello"}', '{"rewritten_query": "Hello"}', 150, 'success');
```

## 📝 TypeScript Types

Generated TypeScript types are available in `app/types/supabase.ts` for type-safe database operations.

## 🎯 Next Steps

1. **Task 1.3:** Flutter Project Setup & Architecture Foundation
2. **Task 1.4:** OpenAI API Integration (Backend)
3. **Task 2.1:** Document Ingestion API (Backend)

## 📚 Resources

- **Supabase Dashboard:** [https://supabase.com/dashboard/project/zpwgvyfxvmhfayylwwmz](https://supabase.com/dashboard/project/zpwgvyfxvmhfayylwwmz)
- **pgvector Documentation:** [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **Supabase Python Client:** [https://github.com/supabase/supabase-py](https://github.com/supabase/supabase-py)

---

**Task 1.2 Status:** ✅ **COMPLETED**
**Date:** 2025-05-27
**Duration:** ~30 minutes
**Next Task:** 1.3 - Flutter Project Setup & Architecture Foundation
