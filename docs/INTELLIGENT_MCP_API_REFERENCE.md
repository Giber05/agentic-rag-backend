# Intelligent MCP API Reference

## Overview

This document provides comprehensive API reference for the Intelligent MCP Chaining system. All endpoints use REST conventions with JSON request/response bodies.

## Base URL

```
http://localhost:8000/api/v1/rag
```

## Authentication

Currently, the API uses basic authentication or API key authentication depending on your configuration. Include authentication headers as required by your setup.

## Content Types

All requests and responses use `Content-Type: application/json` unless otherwise specified.

## Error Handling

All endpoints return structured error responses with HTTP status codes:

```json
{
  "error": "Error description",
  "details": "Additional error context",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (authentication required)
- `422`: Validation Error (invalid request body)
- `500`: Internal Server Error
- `503`: Service Unavailable (MCP services down)

---

## Core Endpoints

### 1. Process Query with Intelligent MCP

Process a query using intelligent operation chaining.

**Endpoint:** `POST /process`

**Request Body:**

```json
{
  "query": "string",
  "source": "intelligent",
  "context": {},
  "config": {
    "max_operations": 5,
    "timeout": 30.0,
    "confidence_threshold": 0.3
  }
}
```

**Parameters:**

- `query` (required): Natural language query string
- `source` (required): Must be `"intelligent"` for MCP chaining
- `context` (optional): Additional context for query processing
- `config` (optional): Override default configuration for this request

**Response:**

```json
{
  "response": "Synthesized response from operations",
  "sources": [
    {
      "id": "source_1",
      "title": "Issue PROJ-123",
      "content": "Issue details...",
      "score": 0.95,
      "source_type": "jira",
      "metadata": {
        "operation_type": "JIRA_GET_ISSUE",
        "issue_key": "PROJ-123"
      }
    }
  ],
  "metadata": {
    "total_operations": 3,
    "successful_operations": 3,
    "chain_duration": 4.2,
    "intent": "issue_analysis",
    "confidence": 0.87
  }
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me details for issue PROJ-123 and related documentation",
    "source": "intelligent",
    "config": {
      "max_operations": 5,
      "timeout": 30.0
    }
  }'
```

---

## Intelligent MCP Specific Endpoints

### 2. Preview Operation Plan

Preview the operation plan without executing it.

**Endpoint:** `POST /intelligent/preview`

**Request Body:**

```json
{
  "query": "string",
  "config": {
    "max_operations": 5,
    "confidence_threshold": 0.3
  }
}
```

**Response:**

```json
{
  "plan": {
    "intent": "issue_analysis",
    "confidence": 0.85,
    "estimated_cost": 0.4,
    "estimated_duration": 6.0,
    "operations": [
      {
        "type": "JIRA_GET_ISSUE",
        "description": "Get details for issue PROJ-123",
        "params": { "issue_key": "PROJ-123" },
        "estimated_duration": 2.0,
        "depends_on": []
      },
      {
        "type": "CONFLUENCE_SEARCH",
        "description": "Search for related documentation",
        "params": { "query": "PROJ-123 documentation" },
        "estimated_duration": 4.0,
        "depends_on": ["JIRA_GET_ISSUE"]
      }
    ]
  },
  "preview_only": true,
  "entities": {
    "issue_keys": ["PROJ-123"],
    "projects": ["PROJ"],
    "statuses": [],
    "users": []
  }
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/rag/intelligent/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all high priority bugs in project DEMO",
    "config": {
      "max_operations": 3
    }
  }'
```

### 3. Health Check

Check the health status of intelligent MCP components.

**Endpoint:** `GET /intelligent/health`

**Response:**

```json
{
  "healthy": true,
  "status": "IDLE",
  "agent_id": "intelligent_mcp",
  "orchestrator": {
    "healthy": true,
    "mcp_service_healthy": true,
    "last_health_check": "2024-01-01T12:00:00Z"
  },
  "statistics": {
    "total_queries": 145,
    "successful_chains": 138,
    "failed_chains": 7,
    "avg_chain_duration": 4.2,
    "cache_hit_rate": 0.23
  },
  "version": "1.0.0",
  "uptime": 3600
}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/health"
```

### 4. Performance Statistics

Get detailed performance statistics and metrics.

**Endpoint:** `GET /intelligent/statistics`

**Response:**

```json
{
  "overview": {
    "total_queries_processed": 1250,
    "total_successful_chains": 1187,
    "total_failed_chains": 63,
    "success_rate": 0.9496,
    "avg_response_time": 4.7,
    "cache_hit_rate": 0.32
  },
  "intent_distribution": {
    "issue_analysis": 345,
    "documentation_search": 298,
    "project_overview": 267,
    "troubleshooting": 156,
    "status_check": 98,
    "relationship_mapping": 65,
    "general_search": 21
  },
  "operation_statistics": {
    "JIRA_SEARCH": {
      "total_executions": 892,
      "success_rate": 0.967,
      "avg_duration": 2.3
    },
    "CONFLUENCE_SEARCH": {
      "total_executions": 743,
      "success_rate": 0.945,
      "avg_duration": 3.1
    },
    "JIRA_GET_ISSUE": {
      "total_executions": 456,
      "success_rate": 0.989,
      "avg_duration": 1.2
    }
  },
  "performance_trends": {
    "last_24h": {
      "queries": 45,
      "avg_response_time": 4.2,
      "success_rate": 0.978
    },
    "last_7d": {
      "queries": 298,
      "avg_response_time": 4.5,
      "success_rate": 0.951
    }
  },
  "optimization_recommendations": [
    {
      "category": "performance",
      "severity": "medium",
      "message": "Consider increasing cache TTL to reduce repeated operations",
      "action": "Update cache_ttl in configuration"
    }
  ]
}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/rag/intelligent/statistics"
```

---

## Configuration Management

### 5. Get Configuration

Retrieve current system configuration.

**Endpoint:** `GET /intelligent/config`

**Response:**

```json
{
  "performance": {
    "max_operations": 5,
    "timeout": 30.0,
    "confidence_threshold": 0.3,
    "cost_limit": 1.0,
    "enable_caching": true,
    "cache_ttl": 300,
    "max_concurrent_chains": 10
  },
  "intelligence": {
    "intent_detection_enabled": true,
    "entity_extraction_enabled": true,
    "confidence_scoring_enabled": true,
    "adaptive_planning": true
  },
  "operations": {
    "jira_enabled": true,
    "confluence_enabled": true,
    "parallel_execution": true,
    "retry_failed_operations": true,
    "max_retries": 2
  },
  "templates": {
    "issue_analysis": {...},
    "documentation_search": {...},
    "project_overview": {...}
  }
}
```

### 6. Update Configuration

Update system configuration.

**Endpoint:** `PUT /intelligent/config`

**Request Body:**

```json
{
  "updates": {
    "performance": {
      "max_operations": 7,
      "timeout": 45.0,
      "confidence_threshold": 0.2
    },
    "operations": {
      "parallel_execution": false
    }
  }
}
```

**Response:**

```json
{
  "message": "Configuration updated successfully",
  "updated_fields": [
    "performance.max_operations",
    "performance.timeout",
    "performance.confidence_threshold",
    "operations.parallel_execution"
  ],
  "config": {
    // Updated configuration object
  }
}
```

**Example Request:**

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

---

## Template Management

### 7. List Operation Templates

Get all available operation templates.

**Endpoint:** `GET /intelligent/config/templates`

**Response:**

```json
{
  "templates": [
    {
      "name": "issue_analysis",
      "description": "Deep analysis of specific issues",
      "intent_types": ["issue_analysis"],
      "operations": [
        {
          "type": "JIRA_GET_ISSUE",
          "params": { "issue_key": "{issue_key}" }
        },
        {
          "type": "CONFLUENCE_SEARCH",
          "params": { "query": "{issue_key} documentation" }
        }
      ],
      "conditions": {
        "min_entities": ["issue_keys"],
        "max_operations": 3
      },
      "created_at": "2024-01-01T12:00:00Z",
      "usage_count": 456
    }
  ],
  "total_templates": 7
}
```

### 8. Create Operation Template

Create a new custom operation template.

**Endpoint:** `POST /intelligent/config/templates`

**Request Body:**

```json
{
  "name": "custom_workflow",
  "description": "Custom workflow for release management",
  "intent_types": ["release", "deployment"],
  "operations": [
    {
      "type": "JIRA_SEARCH",
      "params": { "jql": "fixVersion = '{version}' AND status != Done" },
      "description": "Find incomplete issues for version"
    },
    {
      "type": "CONFLUENCE_SEARCH",
      "params": { "query": "release notes {version}" },
      "description": "Find release documentation"
    }
  ],
  "conditions": {
    "min_entities": ["version"],
    "max_operations": 2,
    "confidence_threshold": 0.4
  }
}
```

**Response:**

```json
{
  "message": "Template created successfully",
  "template": {
    "id": "template_12345"
    // Complete template object
  }
}
```

### 9. Delete Operation Template

Delete a custom operation template.

**Endpoint:** `DELETE /intelligent/config/templates/{template_name}`

**Response:**

```json
{
  "message": "Template 'custom_workflow' deleted successfully"
}
```

**Example Request:**

```bash
curl -X DELETE "http://localhost:8000/api/v1/rag/intelligent/config/templates/custom_workflow"
```

---

## Performance Configuration

### 10. Get Performance Settings

Get current performance configuration.

**Endpoint:** `GET /intelligent/config/performance`

**Response:**

```json
{
  "performance": {
    "max_operations": 5,
    "timeout": 30.0,
    "confidence_threshold": 0.3,
    "cost_limit": 1.0,
    "enable_caching": true,
    "cache_ttl": 300,
    "max_concurrent_chains": 10,
    "operation_timeout": 10.0
  },
  "limits": {
    "max_operations_hard_limit": 15,
    "max_timeout": 120.0,
    "min_confidence_threshold": 0.1
  }
}
```

### 11. Update Performance Settings

Update performance configuration.

**Endpoint:** `PUT /intelligent/config/performance`

**Request Body:**

```json
{
  "max_operations": 8,
  "timeout": 60.0,
  "confidence_threshold": 0.25,
  "enable_caching": true,
  "cache_ttl": 600
}
```

**Response:**

```json
{
  "message": "Performance settings updated successfully",
  "performance": {
    // Updated performance configuration
  }
}
```

### 12. Reset Configuration

Reset configuration to default values.

**Endpoint:** `POST /intelligent/config/reset`

**Request Body (optional):**

```json
{
  "sections": ["performance", "operations"],
  "confirm": true
}
```

**Response:**

```json
{
  "message": "Configuration reset to defaults",
  "reset_sections": ["performance", "operations"],
  "config": {
    // Default configuration
  }
}
```

---

## Query Examples by Intent Type

### Issue Analysis Queries

```bash
# Get specific issue details
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me details for issue PROJ-123", "source": "intelligent"}'

# Analyze bug with context
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze bug DEMO-456 and find related issues", "source": "intelligent"}'
```

### Documentation Search Queries

```bash
# Find setup documentation
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Find setup instructions for new developers", "source": "intelligent"}'

# Search API documentation
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Search for API documentation", "source": "intelligent"}'
```

### Project Overview Queries

```bash
# Project status check
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of project DEMO?", "source": "intelligent"}'

# Release overview
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Project overview for Q4 release", "source": "intelligent"}'
```

### Troubleshooting Queries

```bash
# Fix login issues
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to fix login errors", "source": "intelligent"}'

# Database troubleshooting
curl -X POST "http://localhost:8000/api/v1/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Troubleshoot database connection issues", "source": "intelligent"}'
```

## Response Models

### OperationPlan Model

```json
{
  "intent": "issue_analysis",
  "confidence": 0.85,
  "estimated_cost": 0.4,
  "estimated_duration": 6.0,
  "operations": [
    {
      "type": "JIRA_GET_ISSUE",
      "description": "Get issue details",
      "params": { "issue_key": "PROJ-123" },
      "estimated_duration": 2.0,
      "depends_on": []
    }
  ],
  "entities": {
    "issue_keys": ["PROJ-123"],
    "projects": ["PROJ"],
    "statuses": [],
    "users": []
  }
}
```

### ChainResult Model

```json
{
  "success": true,
  "intent": "issue_analysis",
  "confidence": 0.87,
  "total_operations": 3,
  "successful_operations": 3,
  "failed_operations": 0,
  "chain_duration": 4.2,
  "results": [
    {
      "operation": "JIRA_GET_ISSUE",
      "success": true,
      "duration": 1.2,
      "data": {...}
    }
  ],
  "synthesized_response": "Based on the analysis...",
  "sources": [...]
}
```

### RetrievedSource Model

```json
{
  "id": "source_1",
  "title": "Issue PROJ-123: Login Bug",
  "content": "Issue description and details...",
  "score": 0.95,
  "source_type": "jira",
  "metadata": {
    "operation_type": "JIRA_GET_ISSUE",
    "issue_key": "PROJ-123",
    "project": "PROJ",
    "status": "In Progress",
    "priority": "High",
    "created": "2024-01-01T12:00:00Z"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limits**: 100 requests per minute per IP
- **Headers**: Rate limit information in response headers
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Requests remaining in current window
  - `X-RateLimit-Reset`: Window reset time (Unix timestamp)

When rate limited, the API returns HTTP 429 with:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

## Webhooks (Future Feature)

Future versions will support webhooks for:

- Operation completion notifications
- Health status changes
- Performance threshold alerts
- Configuration updates

---

## SDK Examples

### Python SDK Usage

```python
import asyncio
from intelligent_mcp_client import IntelligentMCPClient

client = IntelligentMCPClient("http://localhost:8000/api/v1/rag")

async def main():
    # Process query with intelligent chaining
    result = await client.process_query(
        "Show me details for issue PROJ-123",
        source="intelligent"
    )

    # Preview operation plan
    preview = await client.preview_operations(
        "Find all high priority bugs in project DEMO"
    )

    # Check health
    health = await client.health_check()

    # Update configuration
    await client.update_config({
        "performance": {
            "max_operations": 7,
            "timeout": 45.0
        }
    })

asyncio.run(main())
```

### JavaScript SDK Usage

```javascript
import { IntelligentMCPClient } from "intelligent-mcp-client";

const client = new IntelligentMCPClient("http://localhost:8000/api/v1/rag");

// Process query
const result = await client.processQuery({
  query: "Show me details for issue PROJ-123",
  source: "intelligent",
});

// Preview operations
const preview = await client.previewOperations(
  "Find all high priority bugs in project DEMO"
);

// Get statistics
const stats = await client.getStatistics();
```

---

_This API reference provides complete documentation for integrating with the Intelligent MCP Chaining system. For additional examples and advanced usage patterns, see the User Guide._
 