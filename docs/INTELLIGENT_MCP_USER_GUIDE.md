# Intelligent MCP Chaining - User Guide

## Overview

The Intelligent MCP Chaining system provides AI-powered orchestration of operations across Jira and Confluence, enabling complex multi-step workflows through natural language queries. This system acts like an intelligent assistant that can analyze your questions and automatically execute the most appropriate sequence of operations.

## Key Features

### 🧠 **Intelligent Query Analysis**

- **7 Intent Types**: Automatically detects what you're trying to accomplish
- **Entity Extraction**: Identifies issue keys, project names, usernames, and statuses
- **Confidence Scoring**: Provides confidence levels for operation plans

### 🔗 **Smart Operation Chaining**

- **Dynamic Planning**: Creates operation sequences based on context
- **Dependency Resolution**: Handles operation dependencies automatically
- **Result Synthesis**: Combines results from multiple operations intelligently

### ⚡ **High Performance**

- **Caching**: 5-minute TTL caching for repeated queries
- **Concurrent Execution**: Parallel operations where possible
- **Timeout Handling**: Graceful degradation when operations take too long

### 🔧 **Flexible Configuration**

- **Runtime Configuration**: Update settings without restarting
- **Operation Templates**: Customize operation sequences
- **Performance Tuning**: Adjust timeouts, limits, and thresholds

## Quick Start

### 1. Basic Usage

The simplest way to use intelligent MCP chaining is through the RAG pipeline with `source: "intelligent"`:

```python
# Using the RAG API
response = await rag_pipeline.process({
    "query": "Show me details for issue PROJ-123 and related documentation",
    "source": "intelligent"
})
```

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me details for issue PROJ-123 and related documentation",
    "source": "intelligent"
  }'
```

### 2. Operation Preview

Before executing operations, you can preview what the system plans to do:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/intelligent/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all high priority bugs in project DEMO"
  }'
```

**Response:**

```json
{
  "plan": {
    "intent": "project_overview",
    "confidence": 0.85,
    "estimated_cost": 0.3,
    "estimated_duration": 5.2,
    "operations": [
      {
        "type": "JIRA_SEARCH",
        "description": "Search for high priority bugs in DEMO project",
        "estimated_duration": 3.0
      },
      {
        "type": "CONFLUENCE_SEARCH",
        "description": "Search for related documentation",
        "estimated_duration": 2.2
      }
    ]
  }
}
```

## Query Types and Examples

### 1. Issue Analysis

**Intent**: Deep dive into specific issues with related context

**Examples:**

- `"Show me details for issue PROJ-123"`
- `"Analyze bug DEMO-456 and find related issues"`
- `"Get comprehensive information about PROJ-789"`

**What it does:**

1. Fetches detailed issue information from Jira
2. Searches for related documentation in Confluence
3. Combines results for comprehensive analysis

### 2. Documentation Search

**Intent**: Find documentation and knowledge base articles

**Examples:**

- `"How to deploy the application"`
- `"Find setup instructions for new developers"`
- `"Search for API documentation"`

**What it does:**

1. Searches Confluence for relevant documentation
2. Looks for related issues in Jira for practical examples
3. Provides comprehensive documentation resources

### 3. Project Overview

**Intent**: Get status and overview of projects

**Examples:**

- `"What's the status of project DEMO?"`
- `"Show me all issues in project PROJ"`
- `"Project overview for Q4 release"`

**What it does:**

1. Searches Jira for project-related issues
2. Finds project documentation in Confluence
3. Provides status summary and key information

### 4. Troubleshooting

**Intent**: Resolve problems and find solutions

**Examples:**

- `"How to fix login errors"`
- `"Troubleshoot database connection issues"`
- `"Find solutions for performance problems"`

**What it does:**

1. Searches Confluence for troubleshooting guides
2. Looks for related issues and their resolutions in Jira
3. Provides step-by-step solutions

### 5. Status Check

**Intent**: Check current status of systems or processes

**Examples:**

- `"What issues are currently in progress?"`
- `"Show me today's deployments"`
- `"Current sprint status"`

**What it does:**

1. Searches Jira for status-related information
2. Gets relevant process documentation from Confluence
3. Provides current state overview

### 6. Relationship Mapping

**Intent**: Understand connections between issues, projects, or components

**Examples:**

- `"What issues are related to PROJ-123?"`
- `"Find dependencies for feature XYZ"`
- `"Show me linked issues"`

**What it does:**

1. Fetches issue details and relationships from Jira
2. Searches for related documentation and processes
3. Maps connections and dependencies

### 7. General Search

**Intent**: Broad search across both platforms

**Examples:**

- `"Find anything related to authentication"`
- `"Search for user management"`
- `"Look up API changes"`

**What it does:**

1. Parallel search across Confluence and Jira
2. Combines and ranks results by relevance
3. Provides comprehensive cross-platform results

## Configuration Management

### Viewing Current Configuration

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/config"
```

### Updating Configuration

```bash
curl -X PUT "http://localhost:8000/api/v1/rag/intelligent/config" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "performance": {
        "max_operations": 7,
        "timeout": 45.0
      }
    }
  }'
```

### Performance Settings

| Setting                | Default | Description                      |
| ---------------------- | ------- | -------------------------------- |
| `max_operations`       | 5       | Maximum operations per chain     |
| `timeout`              | 30.0    | Total chain timeout (seconds)    |
| `confidence_threshold` | 0.3     | Minimum confidence for execution |
| `cost_limit`           | 1.0     | Maximum cost per operation chain |
| `enable_caching`       | true    | Enable result caching            |
| `cache_ttl`            | 300     | Cache time-to-live (seconds)     |

### Operation Templates

View available templates:

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/config/templates"
```

Add custom template:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/intelligent/config/templates" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_workflow",
    "intent_types": ["custom"],
    "operations": [
      {"type": "JIRA_SEARCH", "params": {"jql": "project = {project}"}},
      {"type": "CONFLUENCE_SEARCH", "params": {"query": "{project} documentation"}}
    ],
    "description": "Custom project workflow"
  }'
```

## Performance Monitoring

### Health Check

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/health"
```

**Response:**

```json
{
  "healthy": true,
  "status": "IDLE",
  "agent_id": "intelligent_mcp",
  "orchestrator": {
    "healthy": true,
    "mcp_service_healthy": true
  },
  "statistics": {
    "total_queries": 145,
    "successful_chains": 138,
    "avg_chain_duration": 4.2
  }
}
```

### Performance Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/statistics"
```

**Response includes:**

- Operation success rates
- Average response times
- Intent distribution
- Cache hit rates
- Optimization recommendations

### Optimization Recommendations

The system automatically analyzes performance and provides recommendations:

- **Performance Issues**: Slow operations identified with suggestions
- **Reliability Issues**: High failure rates with remediation steps
- **Resource Issues**: Memory or CPU warnings with optimization tips
- **Concurrency Issues**: Too many parallel operations detected

## Best Practices

### 1. Query Formulation

**✅ Good Queries:**

- Specific and clear: `"Show me details for issue PROJ-123"`
- Include context: `"Find deployment documentation for project DEMO"`
- Use natural language: `"What bugs are currently blocking the release?"`

**❌ Avoid:**

- Too vague: `"Find stuff"`
- Too complex: `"Show me all issues created last week by John that are high priority and related to the API but not including documentation issues"`
- Missing context: `"Fix this"`

### 2. Performance Optimization

**For Fast Responses:**

- Use specific issue keys when possible
- Cache frequently accessed results
- Limit operation chains to essential steps

**For Comprehensive Results:**

- Allow higher operation limits
- Use broader search terms
- Enable all operation types

### 3. Configuration Tuning

**For High-Volume Usage:**

```json
{
  "performance": {
    "max_operations": 3,
    "timeout": 15.0,
    "enable_caching": true,
    "cache_ttl": 600
  }
}
```

**For Comprehensive Analysis:**

```json
{
  "performance": {
    "max_operations": 8,
    "timeout": 60.0,
    "confidence_threshold": 0.2
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Slow Response Times

**Symptoms:** Operations taking > 30 seconds
**Solutions:**

- Check MCP service health
- Reduce `max_operations` limit
- Increase `confidence_threshold`
- Enable caching

#### 2. Low Confidence Scores

**Symptoms:** Plans not executing (confidence < threshold)
**Solutions:**

- Lower `confidence_threshold`
- Use more specific queries
- Add relevant operation templates

#### 3. High Failure Rates

**Symptoms:** Operations failing frequently
**Solutions:**

- Check Jira/Confluence connectivity
- Verify MCP configuration
- Review operation templates
- Check system resources

### Debug Mode

Enable debug logging for detailed operation traces:

```bash
# Set environment variable
export INTELLIGENT_MCP_DEBUG=true

# Or update configuration
curl -X PUT "http://localhost:8000/api/v1/rag/intelligent/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"debug_mode": true}}'
```

### Health Monitoring

Regular health checks help identify issues early:

```bash
# Automated health monitoring
while true; do
  curl -s "http://localhost:8000/api/v1/rag/intelligent/health" | jq '.healthy'
  sleep 60
done
```

## API Reference

### Endpoints

| Endpoint                                     | Method          | Description                                 |
| -------------------------------------------- | --------------- | ------------------------------------------- |
| `/api/v1/rag/process`                        | POST            | Main RAG processing with intelligent source |
| `/api/v1/rag/intelligent/preview`            | POST            | Preview operation plans                     |
| `/api/v1/rag/intelligent/health`             | GET             | System health check                         |
| `/api/v1/rag/intelligent/statistics`         | GET             | Performance statistics                      |
| `/api/v1/rag/intelligent/config`             | GET/PUT         | Configuration management                    |
| `/api/v1/rag/intelligent/config/templates`   | GET/POST/DELETE | Template management                         |
| `/api/v1/rag/intelligent/config/performance` | GET/PUT         | Performance settings                        |
| `/api/v1/rag/intelligent/config/reset`       | POST            | Reset to defaults                           |

### Request/Response Models

#### RAG Processing Request

```json
{
  "query": "string",
  "source": "intelligent",
  "context": {},
  "config": {
    "max_operations": 5,
    "timeout": 30.0
  }
}
```

#### Operation Preview Response

```json
{
  "plan": {
    "intent": "issue_analysis",
    "confidence": 0.85,
    "estimated_cost": 0.4,
    "estimated_duration": 6.0,
    "operations": [...]
  },
  "preview_only": true
}
```

## Advanced Usage

### Custom Operation Templates

Create specialized workflows for your organization:

```python
# Custom template for release management
release_template = {
    "name": "release_management",
    "intent_types": ["release", "deployment"],
    "operations": [
        {
            "type": "JIRA_SEARCH",
            "params": {"jql": "fixVersion = '{version}' AND status != Done"}
        },
        {
            "type": "CONFLUENCE_SEARCH",
            "params": {"query": "release notes {version}"}
        },
        {
            "type": "JIRA_SEARCH",
            "params": {"jql": "project = '{project}' AND priority = High"}
        }
    ],
    "conditions": {
        "min_entities": ["version"],
        "max_operations": 3
    }
}
```

### Integration with External Systems

The intelligent MCP system can be integrated with:

- **CI/CD Pipelines**: Trigger intelligent analysis from builds
- **Monitoring Systems**: Use for incident response and investigation
- **Chatbots**: Provide natural language interface to Jira/Confluence
- **Dashboards**: Real-time operation status and metrics

### Scaling Considerations

For production deployments:

1. **Resource Requirements:**

   - CPU: 2+ cores for concurrent operations
   - Memory: 4GB+ for caching and operation processing
   - Network: Stable connection to Jira/Confluence

2. **Performance Tuning:**

   - Adjust `max_concurrent_chains` based on load
   - Monitor cache hit rates and adjust TTL
   - Use connection pooling for MCP services

3. **Monitoring:**
   - Set up alerts for health check failures
   - Monitor operation success rates
   - Track response time trends

## Support and Feedback

### Getting Help

1. **Check Health Status**: Always start with health checks
2. **Review Statistics**: Look for patterns in failures
3. **Enable Debug Mode**: Get detailed operation traces
4. **Check Configuration**: Verify settings match your needs

### Performance Issues

If experiencing performance problems:

1. Run optimization recommendations endpoint
2. Adjust configuration based on suggestions
3. Monitor system resources during operations
4. Consider reducing operation complexity

### Feature Requests

The system is designed to be extensible. Consider:

- Custom operation templates for your workflows
- Additional intent types for your use cases
- Performance tuning for your environment
- Integration with your existing tools

---

_This intelligent MCP chaining system provides powerful automation for Jira and Confluence operations. With proper configuration and usage, it can significantly improve productivity and provide insights across your Atlassian ecosystem._
