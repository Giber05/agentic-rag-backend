# Postman Collections for Agentic RAG Backend

This directory contains Postman collections for testing the Agentic RAG AI Agent backend with the new Intelligent MCP Chaining capabilities.

## 📁 Available Collections

### 1. 🧠 `intelligent_mcp_quick_test.json` - **RECOMMENDED FOR QUICK TESTING**

A focused collection designed for rapid testing of the new intelligent MCP features.

**Perfect for:**

- Quick feature validation
- Development testing
- Demonstrating intelligent chaining capabilities
- Performance comparison tests

**Contains:**

- ✅ Health checks for intelligent MCP components
- 🧠 Core intelligent operation tests (issue analysis, project overview, documentation search, troubleshooting)
- 👁️ Operation preview and planning endpoints
- 📊 Performance monitoring and statistics
- ⚙️ Configuration management
- 🔧 Direct MCP testing
- 🚀 Comparison tests (intelligent vs standard processing)

### 2. 📚 `agentic_rag_intelligent_mcp_collection.json` - **COMPLETE COLLECTION**

A comprehensive collection with ALL endpoints in the system.

**Perfect for:**

- Complete system testing
- Integration testing
- Full API documentation
- Production testing

**Contains:**

- All intelligent MCP endpoints
- Complete RAG pipeline endpoints
- Individual agent testing
- Document management
- Authentication
- Analytics and monitoring
- Legacy endpoints

### 3. 📋 Legacy Collections

- `postman_collection.json` - Original complete collection
- `postman_collection_v1.1.json` - Previous version

## 🚀 Quick Start Guide

### Step 1: Import Collections

1. Open Postman
2. Click "Import"
3. Select `intelligent_mcp_quick_test.json` for quick testing
4. Or select `agentic_rag_intelligent_mcp_collection.json` for complete testing

### Step 2: Configure Environment

1. Set the `baseUrl` variable to your server URL (default: `http://localhost:8000`)
2. Optionally set `authToken` if authentication is required

### Step 3: Start Testing

#### For Quick Feature Testing:

1. **Health Checks**: Run the "🔥 Quick Health Checks" folder
2. **Core Features**: Test "🧠 Intelligent MCP Core Features"
3. **Operation Preview**: Use "👁️ Operation Preview & Planning" to see what operations would be executed

#### For Development Testing:

1. Start with health checks
2. Run individual intelligent queries
3. Compare with standard processing using the "🚀 Comparison Tests"

## 🧠 Intelligent MCP Features to Test

### 1. Intent Detection & Operation Planning

The system detects query intent and plans appropriate operations:

```json
{
  "query": "Show me details for issue PROJ-123 and related documentation",
  "source": "intelligent"
}
```

**Expected:** Jira issue retrieval → Confluence documentation search

### 2. Project Overview Queries

```json
{
  "query": "What's the current status of project DEMO?",
  "source": "intelligent"
}
```

**Expected:** Combined Jira + Confluence search with project context

### 3. Documentation-Focused Search

```json
{
  "query": "Find documentation about API integration guidelines",
  "source": "intelligent"
}
```

**Expected:** Confluence search → Related Jira issues (if any)

### 4. Troubleshooting Workflows

```json
{
  "query": "How to fix authentication errors in the API?",
  "source": "intelligent"
}
```

**Expected:** Confluence troubleshooting docs → Related Jira issues

## 📊 Key Metrics to Monitor

When testing, watch for these metrics in responses:

- **`total_operations`**: Number of operations executed
- **`successful_operations`**: Operations that completed successfully
- **`chain_duration`**: Total time for operation chain
- **`intent`**: Detected query intent (issue_analysis, documentation_search, etc.)
- **`confidence`**: Confidence score for the operation plan

## 🔧 Advanced Testing

### Configuration Management

Test configuration updates:

```json
{
  "updates": {
    "performance": {
      "max_operations": 5,
      "timeout": 30.0,
      "confidence_threshold": 0.3
    }
  }
}
```

### Custom Operation Templates

Add custom operation workflows:

```json
{
  "name": "custom_workflow",
  "intent_types": ["troubleshooting"],
  "operations": [
    {
      "type": "CONFLUENCE_SEARCH",
      "params": { "query": "troubleshooting {error_type}" }
    },
    { "type": "JIRA_SEARCH", "params": { "jql": "text ~ '{error_type}'" } }
  ]
}
```

## 🎯 Testing Scenarios

### Scenario 1: Issue Deep Dive

```
Query: "Show me details for issue DEMO-456 and find all related documentation"
Expected: Jira issue details → Confluence search for related docs
```

### Scenario 2: Project Health Check

```
Query: "What's the current status of our development project?"
Expected: Jira project search → Confluence project documentation
```

### Scenario 3: Knowledge Discovery

```
Query: "Find all documentation about database migration procedures"
Expected: Confluence search → Related Jira migration issues
```

### Scenario 4: Bug Investigation

```
Query: "Are there any known issues with the payment gateway?"
Expected: Jira bug search → Confluence troubleshooting guides
```

## 🆚 Comparison Testing

Use the "🚀 Comparison Tests" folder to compare:

1. **Standard Multi-Source** (`source: "all"`)

   - Executes all sources in parallel
   - Fixed operation pattern
   - No intelligent planning

2. **Intelligent Chaining** (`source: "intelligent"`)
   - Analyzes query intent
   - Plans optimal operation sequence
   - Adaptive based on context and results

## ⚡ Performance Tips

- Use **Preview** endpoints to understand operation plans without execution
- Monitor **Statistics** to track performance patterns
- Adjust **Configuration** based on your use case
- Use **Health Checks** to verify system readiness

## 🔒 Authentication

If authentication is enabled:

1. Use the authentication endpoints to get a token
2. Set the `authToken` variable in your environment
3. The collections are configured to use Bearer token authentication

## 🐛 Troubleshooting

- **500 Errors**: Check MCP service health and configuration
- **Timeout Issues**: Increase timeout in configuration or reduce max_operations
- **Low Confidence**: Adjust confidence_threshold or improve query specificity
- **No Operations**: Verify query intent detection in preview responses

## 📈 Success Metrics

A successful intelligent MCP test should show:

- ✅ `healthy: true` in health checks
- ✅ `total_operations > 0` in responses
- ✅ `successful_operations = total_operations`
- ✅ Reasonable `chain_duration` (< 30 seconds)
- ✅ High `confidence` scores (> 0.5)
- ✅ Appropriate `intent` detection

---

🎉 **Ready to test the future of intelligent document processing!** 🚀
