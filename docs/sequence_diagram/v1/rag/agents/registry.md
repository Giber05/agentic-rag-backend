# Agent Registry

## 📋 Overview

The Agent Registry is a centralized management system for all agents in the RAG pipeline, providing comprehensive lifecycle management, discovery services, and coordination capabilities. It serves as the foundation for the multi-agent architecture, ensuring proper agent instantiation, configuration, and communication.

### Purpose

Provide centralized agent management through:

- **Agent Type Registration**: Register and manage different agent types
- **Instance Creation**: Create and configure agent instances dynamically
- **Lifecycle Management**: Handle agent startup, shutdown, and health monitoring
- **Discovery Services**: Enable agents to find and communicate with each other
- **Resource Management**: Optimize resource allocation and usage
- **Performance Monitoring**: Track agent performance and health metrics

### Key Responsibilities

- **Registration Management**: Maintain registry of available agent types
- **Instance Management**: Create, configure, and destroy agent instances
- **Health Monitoring**: Monitor agent health and handle failures
- **Message Routing**: Route messages between agents efficiently
- **Configuration Management**: Manage agent configurations and updates
- **Performance Tracking**: Collect and analyze agent performance metrics

## 🏗️ Architecture

### Class Structure

```python
class AgentRegistry:
    """
    Centralized registry for agent lifecycle and discovery management.

    Capabilities:
    - Agent type registration and management
    - Dynamic instance creation and configuration
    - Health monitoring and recovery
    - Message routing and communication
    - Performance tracking and analytics
    - Resource optimization and cleanup
    """
```

### Core Components

#### 1. Agent Type Manager

- **Type Registration**: Register agent classes with metadata
- **Capability Discovery**: Track agent capabilities and requirements
- **Version Management**: Handle different agent versions
- **Compatibility Checking**: Ensure agent compatibility

#### 2. Instance Manager

- **Instance Creation**: Create agent instances with proper configuration
- **Instance Pool**: Maintain pool of active agent instances
- **Resource Allocation**: Optimize resource usage across instances
- **Instance Cleanup**: Properly cleanup terminated instances

#### 3. Health Monitor

- **Health Checks**: Periodic health verification for all agents
- **Failure Detection**: Detect and respond to agent failures
- **Recovery Management**: Automatic recovery and restart procedures
- **Health Reporting**: Generate health status reports

#### 4. Message Router

- **Routing Table**: Maintain routing information for agent communication
- **Message Queuing**: Queue messages for asynchronous delivery
- **Load Balancing**: Distribute messages across agent instances
- **Delivery Guarantees**: Ensure reliable message delivery

#### 5. Configuration Manager

- **Configuration Storage**: Store and manage agent configurations
- **Dynamic Updates**: Apply configuration changes without restart
- **Environment Management**: Handle different deployment environments
- **Validation**: Validate configuration changes before application

#### 6. Performance Tracker

- **Metrics Collection**: Collect performance metrics from all agents
- **Analytics Engine**: Analyze performance patterns and trends
- **Alerting System**: Generate alerts for performance issues
- **Reporting**: Generate performance reports and dashboards

## 🔧 Configuration

### Registry Configuration

```python
config = {
    # Instance management
    "max_instances_per_type": 10,          # Maximum instances per agent type
    "instance_pool_size": 5,               # Default instance pool size
    "instance_timeout_seconds": 300,       # Instance idle timeout
    "enable_instance_pooling": True,       # Enable instance reuse

    # Health monitoring
    "health_check_interval_seconds": 30,   # Health check frequency
    "health_check_timeout_seconds": 5,     # Health check timeout
    "max_consecutive_failures": 3,         # Max failures before restart
    "enable_auto_recovery": True,          # Enable automatic recovery

    # Message routing
    "message_queue_size": 1000,            # Message queue capacity
    "message_timeout_seconds": 30,         # Message delivery timeout
    "enable_message_persistence": True,    # Persist messages to disk
    "routing_strategy": "round_robin",     # Message routing strategy

    # Performance monitoring
    "enable_performance_tracking": True,   # Enable performance monitoring
    "metrics_collection_interval": 60,     # Metrics collection frequency (seconds)
    "metrics_retention_days": 30,          # How long to keep metrics
    "enable_alerts": True,                 # Enable performance alerts

    # Resource management
    "enable_resource_optimization": True,  # Enable resource optimization
    "resource_check_interval": 120,        # Resource check frequency (seconds)
    "memory_threshold_mb": 1024,           # Memory usage alert threshold
    "cpu_threshold_percent": 80            # CPU usage alert threshold
}
```

### Environment Variables

```bash
# Registry configuration
AGENT_REGISTRY_ENABLED=true
AGENT_REGISTRY_MAX_INSTANCES=10
AGENT_REGISTRY_HEALTH_CHECK_INTERVAL=30
AGENT_REGISTRY_AUTO_RECOVERY=true

# Message routing
AGENT_REGISTRY_MESSAGE_QUEUE_SIZE=1000
AGENT_REGISTRY_MESSAGE_TIMEOUT=30
AGENT_REGISTRY_ROUTING_STRATEGY=round_robin

# Performance monitoring
AGENT_REGISTRY_PERFORMANCE_TRACKING=true
AGENT_REGISTRY_METRICS_INTERVAL=60
AGENT_REGISTRY_ALERTS_ENABLED=true
```

## 📚 API Reference

### Core Methods

#### Agent Type Management

```python
def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent],
                       metadata: Dict[str, Any] = None) -> bool:
    """
    Register a new agent type with the registry.

    Parameters:
    - agent_type: Unique identifier for the agent type
    - agent_class: Agent class that implements BaseAgent
    - metadata: Additional metadata about the agent type

    Returns:
    - bool: True if registration successful, False otherwise
    """

def unregister_agent_type(self, agent_type: str) -> bool:
    """
    Unregister an agent type from the registry.

    Parameters:
    - agent_type: Agent type identifier to unregister

    Returns:
    - bool: True if unregistration successful, False otherwise
    """

def get_registered_types(self) -> List[str]:
    """
    Get list of all registered agent types.

    Returns:
    - List[str]: List of registered agent type identifiers
    """

def get_agent_metadata(self, agent_type: str) -> Dict[str, Any]:
    """
    Get metadata for a specific agent type.

    Parameters:
    - agent_type: Agent type identifier

    Returns:
    - Dict[str, Any]: Agent metadata including capabilities and requirements
    """
```

#### Instance Management

```python
async def create_agent(self, agent_type: str, agent_id: str = None,
                      config: Dict[str, Any] = None) -> BaseAgent:
    """
    Create a new agent instance.

    Parameters:
    - agent_type: Type of agent to create
    - agent_id: Optional specific ID for the agent
    - config: Optional configuration overrides

    Returns:
    - BaseAgent: Created and configured agent instance
    """

async def get_agent(self, agent_id: str) -> BaseAgent:
    """
    Get an existing agent instance by ID.

    Parameters:
    - agent_id: Agent instance identifier

    Returns:
    - BaseAgent: Agent instance if found, None otherwise
    """

async def destroy_agent(self, agent_id: str) -> bool:
    """
    Destroy an agent instance and cleanup resources.

    Parameters:
    - agent_id: Agent instance identifier

    Returns:
    - bool: True if destruction successful, False otherwise
    """

def list_agents(self, agent_type: str = None,
               status: str = None) -> List[Dict[str, Any]]:
    """
    List agent instances with optional filtering.

    Parameters:
    - agent_type: Optional filter by agent type
    - status: Optional filter by agent status

    Returns:
    - List[Dict[str, Any]]: List of agent information dictionaries
    """
```

#### Health Management

```python
async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
    """
    Check health status of a specific agent.

    Parameters:
    - agent_id: Agent instance identifier

    Returns:
    - Dict[str, Any]: Health status information
    """

async def check_all_health(self) -> Dict[str, Dict[str, Any]]:
    """
    Check health status of all registered agents.

    Returns:
    - Dict[str, Dict[str, Any]]: Health status for all agents
    """

async def recover_agent(self, agent_id: str) -> bool:
    """
    Attempt to recover a failed agent.

    Parameters:
    - agent_id: Agent instance identifier

    Returns:
    - bool: True if recovery successful, False otherwise
    """
```

#### Message Routing

```python
async def send_message(self, from_agent: str, to_agent: str,
                      message: Dict[str, Any]) -> bool:
    """
    Send a message between agents.

    Parameters:
    - from_agent: Sender agent ID
    - to_agent: Recipient agent ID
    - message: Message payload

    Returns:
    - bool: True if message sent successfully, False otherwise
    """

async def broadcast_message(self, from_agent: str,
                           message: Dict[str, Any],
                           agent_type: str = None) -> int:
    """
    Broadcast a message to multiple agents.

    Parameters:
    - from_agent: Sender agent ID
    - message: Message payload
    - agent_type: Optional filter by agent type

    Returns:
    - int: Number of agents that received the message
    """
```

### Registry Lifecycle

#### Startup and Shutdown

```python
async def start(self) -> None:
    """
    Start the agent registry and all its components.

    Operations:
    - Initialize internal components
    - Start health monitoring
    - Initialize message routing
    - Begin performance tracking
    """

async def stop(self) -> None:
    """
    Stop the agent registry and cleanup resources.

    Operations:
    - Stop all agent instances
    - Cleanup message queues
    - Stop monitoring threads
    - Save persistent state
    """

async def restart(self) -> None:
    """
    Restart the registry (stop then start).

    Useful for applying configuration changes that require restart.
    """
```

## 💡 Usage Examples

### Basic Registry Usage

```python
from app.agents.registry import AgentRegistry
from app.agents.query_rewriter import QueryRewritingAgent
from app.agents.context_decision import ContextDecisionAgent

# Initialize registry
registry = AgentRegistry()
await registry.start()

# Register agent types
registry.register_agent_type("query_rewriter", QueryRewritingAgent, {
    "description": "Agent for query rewriting and optimization",
    "capabilities": ["spell_check", "grammar_correction", "query_expansion"],
    "requirements": {"openai_api": True}
})

registry.register_agent_type("context_decision", ContextDecisionAgent, {
    "description": "Agent for context necessity decisions",
    "capabilities": ["similarity_analysis", "query_classification"],
    "requirements": {"embedding_service": True}
})

# Create agent instances
query_rewriter = await registry.create_agent(
    agent_type="query_rewriter",
    agent_id="qr-001",
    config={"enable_expansion": True}
)

context_decision = await registry.create_agent(
    agent_type="context_decision",
    agent_id="cd-001",
    config={"similarity_threshold": 0.7}
)

print(f"Created agents: {[agent.agent_id for agent in [query_rewriter, context_decision]]}")

# List all agents
agents = registry.list_agents()
for agent_info in agents:
    print(f"Agent {agent_info['agent_id']}: {agent_info['status']}")
```

### Health Monitoring

```python
# Check health of all agents
async def monitor_agent_health(registry):
    health_status = await registry.check_all_health()

    for agent_id, health in health_status.items():
        status = health["status"]
        last_seen = health["last_seen"]

        print(f"Agent {agent_id}: {status} (last seen: {last_seen})")

        if status == "unhealthy":
            print(f"  Issues: {health.get('issues', [])}")

            # Attempt recovery
            print(f"  Attempting recovery...")
            recovery_success = await registry.recover_agent(agent_id)
            print(f"  Recovery {'successful' if recovery_success else 'failed'}")

# Run health monitoring
await monitor_agent_health(registry)
```

### Message Routing

```python
# Send messages between agents
async def demonstrate_messaging(registry):
    # Send direct message
    message_sent = await registry.send_message(
        from_agent="qr-001",
        to_agent="cd-001",
        message={
            "type": "query_processed",
            "data": {
                "original_query": "What is AI?",
                "rewritten_query": "What is artificial intelligence?",
                "confidence": 0.89
            }
        }
    )

    print(f"Direct message sent: {message_sent}")

    # Broadcast to all agents of a type
    recipients = await registry.broadcast_message(
        from_agent="coordinator",
        message={
            "type": "config_update",
            "data": {"temperature": 0.2}
        },
        agent_type="query_rewriter"
    )

    print(f"Broadcast message sent to {recipients} agents")

await demonstrate_messaging(registry)
```

### Performance Monitoring

```python
# Get performance metrics
async def analyze_performance(registry):
    # Get overall performance metrics
    metrics = registry.get_performance_metrics()

    print("Overall Performance Metrics:")
    print(f"  Total Agents: {metrics['total_agents']}")
    print(f"  Active Agents: {metrics['active_agents']}")
    print(f"  Average Response Time: {metrics['avg_response_time_ms']:.1f}ms")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")

    # Get per-agent metrics
    agent_metrics = registry.get_agent_metrics()

    print("\nPer-Agent Metrics:")
    for agent_id, agent_metric in agent_metrics.items():
        print(f"  {agent_id}:")
        print(f"    Response Time: {agent_metric['avg_response_time_ms']:.1f}ms")
        print(f"    Success Rate: {agent_metric['success_rate']:.2%}")
        print(f"    Memory Usage: {agent_metric['memory_usage_mb']:.1f}MB")

await analyze_performance(registry)
```

### Dynamic Configuration

```python
# Update agent configuration dynamically
async def update_agent_config(registry, agent_id, new_config):
    agent = await registry.get_agent(agent_id)
    if agent:
        # Update configuration
        await agent.update_config(new_config)
        print(f"Updated configuration for {agent_id}")

        # Verify configuration applied
        current_config = agent.get_config()
        print(f"Current config: {current_config}")
    else:
        print(f"Agent {agent_id} not found")

# Example: Update query rewriter configuration
await update_agent_config(
    registry,
    "qr-001",
    {"enable_expansion": False, "temperature": 0.1}
)
```

## 🎯 Performance Characteristics

### Registry Performance

| Operation                | Average Time | 95th Percentile | Notes                         |
| ------------------------ | ------------ | --------------- | ----------------------------- |
| **Agent Creation**       | 150ms        | 300ms           | Including initialization      |
| **Agent Lookup**         | 1ms          | 3ms             | Memory-based lookup           |
| **Health Check**         | 10ms         | 25ms            | Per agent health verification |
| **Message Routing**      | 5ms          | 15ms            | Local message delivery        |
| **Configuration Update** | 20ms         | 50ms            | Dynamic config changes        |

### Resource Utilization

| Resource         | Baseline | Per Agent  | Notes                           |
| ---------------- | -------- | ---------- | ------------------------------- |
| **Memory Usage** | 50MB     | +15MB      | Base registry + per agent       |
| **CPU Usage**    | 2%       | +1%        | Monitoring and routing overhead |
| **Network**      | <1MB/min | +100KB/min | Health checks and messaging     |
| **Storage**      | 10MB     | +5MB       | Configuration and metrics       |

### Scalability Metrics

| Metric                 | Current | Target | Notes                         |
| ---------------------- | ------- | ------ | ----------------------------- |
| **Max Agents**         | 100     | 500    | Concurrent agent instances    |
| **Messages/sec**       | 1000    | 5000   | Inter-agent messaging rate    |
| **Health Checks/min**  | 200     | 1000   | Health monitoring capacity    |
| **Config Updates/min** | 50      | 200    | Dynamic configuration changes |

## 🚨 Error Handling

### Common Error Scenarios

#### 1. Agent Registration Errors

```python
# Duplicate registration
{"error": "Agent type already registered", "code": "DUPLICATE_REGISTRATION"}

# Invalid agent class
{"error": "Agent class must inherit from BaseAgent", "code": "INVALID_AGENT_CLASS"}

# Missing requirements
{"error": "Agent requirements not met", "code": "REQUIREMENTS_NOT_MET"}
```

#### 2. Instance Management Errors

```python
# Instance limit exceeded
{"error": "Maximum instances exceeded for agent type", "code": "INSTANCE_LIMIT_EXCEEDED"}

# Instance creation failed
{"error": "Failed to create agent instance", "code": "INSTANCE_CREATION_FAILED"}

# Instance not found
{"error": "Agent instance not found", "code": "INSTANCE_NOT_FOUND"}
```

#### 3. Health Monitoring Errors

```python
# Health check timeout
{"error": "Agent health check timeout", "code": "HEALTH_CHECK_TIMEOUT"}

# Recovery failed
{"error": "Agent recovery failed", "code": "RECOVERY_FAILED"}

# Health monitor failure
{"error": "Health monitoring system failure", "code": "HEALTH_MONITOR_ERROR"}
```

### Error Recovery Strategies

#### Automatic Recovery

```python
async def _handle_agent_failure(self, agent_id: str, failure_reason: str):
    """Handle agent failure with automatic recovery."""

    logger.warning(f"Agent {agent_id} failed: {failure_reason}")

    # Get agent information
    agent_info = self.get_agent_info(agent_id)
    if not agent_info:
        return False

    # Attempt recovery based on failure type
    if "timeout" in failure_reason.lower():
        # Simple restart for timeout issues
        return await self._restart_agent(agent_id)
    elif "memory" in failure_reason.lower():
        # Resource cleanup then restart
        await self._cleanup_agent_resources(agent_id)
        return await self._restart_agent(agent_id)
    else:
        # Full recreation for other issues
        return await self._recreate_agent(agent_id, agent_info)

async def _restart_agent(self, agent_id: str) -> bool:
    """Restart an existing agent."""
    try:
        agent = await self.get_agent(agent_id)
        if agent:
            await agent.stop()
            await agent.start()
            return True
    except Exception as e:
        logger.error(f"Failed to restart agent {agent_id}: {e}")
    return False
```

#### Graceful Degradation

```python
def _handle_registry_overload(self):
    """Handle registry overload by implementing graceful degradation."""

    # Reduce health check frequency
    self.config["health_check_interval_seconds"] *= 2

    # Pause non-essential features
    self.config["enable_performance_tracking"] = False

    # Implement circuit breaker for new agent creation
    self._circuit_breaker_active = True

    logger.warning("Registry overload detected, implementing degradation strategies")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. High Memory Usage

**Symptoms**: Registry consuming excessive memory

**Solutions**:

```python
# Enable resource optimization
config = {"enable_resource_optimization": True}

# Reduce instance pool size
config = {"instance_pool_size": 3}

# Implement garbage collection
await registry.cleanup_idle_instances()

# Monitor memory usage
memory_stats = registry.get_memory_usage()
```

#### 2. Agent Creation Failures

**Symptoms**: New agents failing to start

**Solutions**:

```python
# Check agent requirements
requirements = registry.get_agent_metadata(agent_type)["requirements"]

# Verify dependencies
dependency_check = await registry.verify_dependencies(agent_type)

# Check resource availability
resource_status = registry.get_resource_status()

# Enable debug logging
logging.getLogger("app.agents.registry").setLevel(logging.DEBUG)
```

#### 3. Message Delivery Issues

**Symptoms**: Messages not reaching destination agents

**Solutions**:

```python
# Check message queue status
queue_status = registry.get_message_queue_status()

# Verify agent connectivity
connectivity = await registry.check_agent_connectivity()

# Increase message timeout
config = {"message_timeout_seconds": 60}

# Enable message persistence
config = {"enable_message_persistence": True}
```

### Debugging Tools

#### Registry Status Dashboard

```python
def get_registry_status():
    """Get comprehensive registry status for debugging."""

    return {
        "registry_health": {
            "status": registry.get_health_status(),
            "uptime_seconds": registry.get_uptime(),
            "memory_usage_mb": registry.get_memory_usage(),
            "cpu_usage_percent": registry.get_cpu_usage()
        },
        "agent_summary": {
            "total_agents": len(registry.list_agents()),
            "agents_by_type": registry.get_agent_count_by_type(),
            "agents_by_status": registry.get_agent_count_by_status()
        },
        "performance_metrics": {
            "avg_response_time_ms": registry.get_avg_response_time(),
            "success_rate": registry.get_success_rate(),
            "error_rate": registry.get_error_rate()
        },
        "resource_usage": {
            "memory_threshold_exceeded": registry.is_memory_threshold_exceeded(),
            "cpu_threshold_exceeded": registry.is_cpu_threshold_exceeded(),
            "active_connections": registry.get_active_connections()
        }
    }
```

## 🔗 Integration Points

### With Agent Coordinator

```python
# Registry provides agents to coordinator
coordinator = AgentCoordinator(registry)

# Coordinator uses registry for agent management
await coordinator.setup_pipeline([
    ("query_rewriting", "query_rewriter"),
    ("context_decision", "context_decision"),
    ("source_retrieval", "source_retrieval"),
    ("answer_generation", "answer_generation")
])
```

### With External Monitoring

```python
# Export metrics to external monitoring systems
def export_metrics_to_prometheus():
    metrics = registry.get_performance_metrics()

    # Export to Prometheus
    prometheus_client.Gauge('agent_count').set(metrics['total_agents'])
    prometheus_client.Gauge('avg_response_time').set(metrics['avg_response_time_ms'])
    prometheus_client.Gauge('success_rate').set(metrics['success_rate'])
```

## 📊 Monitoring and Metrics

### Key Performance Indicators

```python
{
    "registry_metrics": {
        "total_agents_managed": 45,
        "active_agents": 38,
        "failed_agents": 2,
        "agents_created_today": 15,
        "agents_destroyed_today": 8,
        "average_agent_lifetime_hours": 24.5
    },
    "performance_metrics": {
        "average_agent_creation_time_ms": 145.3,
        "average_health_check_time_ms": 8.7,
        "message_routing_success_rate": 0.998,
        "configuration_update_success_rate": 0.995
    },
    "resource_metrics": {
        "total_memory_usage_mb": 850.2,
        "average_cpu_usage_percent": 12.3,
        "peak_memory_usage_mb": 1200.5,
        "peak_cpu_usage_percent": 45.8
    }
}
```

### Alerting Configuration

```python
ALERT_THRESHOLDS = {
    "agent_failure_rate": 0.05,          # Alert if >5% agents fail
    "memory_usage_mb": 2048,             # Alert if >2GB memory usage
    "cpu_usage_percent": 80,             # Alert if >80% CPU usage
    "message_delivery_failures": 10,     # Alert if >10 delivery failures/hour
    "health_check_failures": 5           # Alert if >5 health check failures/min
}
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from app.agents.registry import AgentRegistry
from app.agents.base import BaseAgent

class MockAgent(BaseAgent):
    async def process(self, input_data):
        return {"result": "mock_processed"}

@pytest.mark.asyncio
async def test_agent_registration():
    registry = AgentRegistry()

    # Test registration
    success = registry.register_agent_type("mock_agent", MockAgent)
    assert success

    # Test duplicate registration
    duplicate = registry.register_agent_type("mock_agent", MockAgent)
    assert not duplicate

    # Test getting registered types
    types = registry.get_registered_types()
    assert "mock_agent" in types

@pytest.mark.asyncio
async def test_agent_creation():
    registry = AgentRegistry()
    await registry.start()

    registry.register_agent_type("mock_agent", MockAgent)

    # Create agent
    agent = await registry.create_agent("mock_agent", "test-agent-1")
    assert agent is not None
    assert agent.agent_id == "test-agent-1"

    # Get agent
    retrieved = await registry.get_agent("test-agent-1")
    assert retrieved == agent

    # Destroy agent
    destroyed = await registry.destroy_agent("test-agent-1")
    assert destroyed

    await registry.stop()

@pytest.mark.asyncio
async def test_health_monitoring():
    registry = AgentRegistry()
    await registry.start()

    registry.register_agent_type("mock_agent", MockAgent)
    agent = await registry.create_agent("mock_agent", "health-test")

    # Check health
    health = await registry.check_agent_health("health-test")
    assert health["status"] in ["healthy", "starting"]

    # Check all health
    all_health = await registry.check_all_health()
    assert "health-test" in all_health

    await registry.stop()
```

---

_The Agent Registry serves as the foundational infrastructure for the multi-agent RAG system, providing robust lifecycle management, health monitoring, and communication capabilities that enable seamless coordination between specialized agents._
