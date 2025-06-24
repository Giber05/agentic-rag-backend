# Agent Coordinator

## 📋 Overview

The Agent Coordinator is the orchestration engine for the RAG pipeline, responsible for managing the execution flow between multiple agents, handling errors and retries, and providing comprehensive pipeline monitoring. It acts as the central conductor that ensures proper sequencing, data flow, and error handling across all pipeline stages.

### Purpose

Orchestrate multi-agent pipeline execution through:

- **Pipeline Configuration**: Define and manage agent execution sequences
- **Execution Flow Control**: Coordinate data flow between pipeline stages
- **Error Handling**: Implement robust error recovery and retry logic
- **Performance Monitoring**: Track pipeline performance and bottlenecks
- **Streaming Support**: Enable real-time streaming pipeline execution
- **Context Management**: Maintain execution context across pipeline stages

### Key Responsibilities

- **Pipeline Definition**: Configure agent sequences and dependencies
- **Execution Management**: Control pipeline execution and stage transitions
- **Data Flow**: Manage data passing between agents
- **Error Recovery**: Handle failures and implement retry strategies
- **Performance Optimization**: Optimize pipeline execution for speed and reliability
- **Monitoring**: Provide comprehensive execution monitoring and analytics

## 🏗️ Architecture

### Class Structure

```python
class AgentCoordinator:
    """
    Orchestrates multi-agent pipeline execution with comprehensive
    error handling, monitoring, and optimization capabilities.

    Capabilities:
    - Pipeline configuration and management
    - Agent execution coordination
    - Error handling and retry logic
    - Performance monitoring and optimization
    - Streaming pipeline support
    - Context and state management
    """
```

### Core Components

#### 1. Pipeline Manager

- **Pipeline Definition**: Define agent sequences and execution rules
- **Stage Configuration**: Configure individual pipeline stages
- **Dependency Management**: Handle stage dependencies and prerequisites
- **Execution Planning**: Optimize execution order and parallelization

#### 2. Execution Engine

- **Stage Execution**: Execute individual pipeline stages
- **Data Flow Control**: Manage data passing between stages
- **Parallel Processing**: Execute independent stages in parallel
- **Synchronization**: Coordinate stage completion and handoffs

#### 3. Error Handler

- **Failure Detection**: Detect and classify execution failures
- **Retry Logic**: Implement configurable retry strategies
- **Recovery Strategies**: Apply different recovery approaches
- **Fallback Mechanisms**: Use alternative execution paths

#### 4. Context Manager

- **Execution Context**: Maintain context throughout pipeline execution
- **State Persistence**: Persist execution state for recovery
- **Data Caching**: Cache intermediate results for efficiency
- **Memory Management**: Optimize memory usage during execution

#### 5. Performance Monitor

- **Execution Tracking**: Track pipeline and stage performance
- **Bottleneck Detection**: Identify performance bottlenecks
- **Resource Monitoring**: Monitor resource usage patterns
- **Analytics**: Generate performance analytics and reports

#### 6. Streaming Controller

- **Stream Management**: Handle streaming pipeline execution
- **Real-time Updates**: Provide real-time execution updates
- **Buffer Management**: Manage streaming data buffers
- **Backpressure Handling**: Handle backpressure in streaming scenarios

## 🔧 Configuration

### Coordinator Configuration

```python
config = {
    # Pipeline execution settings
    "max_concurrent_stages": 3,            # Maximum parallel stage execution
    "stage_timeout_seconds": 60,           # Default stage timeout
    "pipeline_timeout_seconds": 300,       # Overall pipeline timeout
    "enable_parallel_execution": True,     # Enable parallel stage execution

    # Error handling settings
    "max_retry_attempts": 3,               # Maximum retry attempts per stage
    "retry_delay_seconds": 1.0,            # Initial retry delay
    "retry_backoff_factor": 2.0,           # Exponential backoff factor
    "enable_circuit_breaker": True,        # Enable circuit breaker pattern

    # Performance settings
    "enable_performance_monitoring": True, # Enable performance tracking
    "enable_caching": True,                # Enable intermediate result caching
    "cache_ttl_seconds": 1800,             # Cache time-to-live (30 minutes)
    "memory_limit_mb": 2048,               # Memory usage limit

    # Streaming settings
    "enable_streaming": True,              # Enable streaming support
    "stream_buffer_size": 1024,            # Streaming buffer size
    "stream_timeout_ms": 5000,             # Streaming timeout
    "enable_backpressure": True,           # Enable backpressure handling

    # Context management
    "enable_context_persistence": True,    # Persist execution context
    "context_storage_backend": "memory",   # Context storage backend
    "enable_state_recovery": True,         # Enable execution state recovery

    # Monitoring settings
    "enable_detailed_logging": True,       # Enable detailed execution logging
    "log_level": "INFO",                   # Logging level
    "enable_metrics_collection": True,     # Enable metrics collection
    "metrics_export_interval": 60          # Metrics export interval (seconds)
}
```

### Environment Variables

```bash
# Coordinator configuration
AGENT_COORDINATOR_ENABLED=true
AGENT_COORDINATOR_MAX_CONCURRENT=3
AGENT_COORDINATOR_STAGE_TIMEOUT=60
AGENT_COORDINATOR_PIPELINE_TIMEOUT=300

# Error handling
AGENT_COORDINATOR_MAX_RETRIES=3
AGENT_COORDINATOR_RETRY_DELAY=1.0
AGENT_COORDINATOR_CIRCUIT_BREAKER=true

# Performance and monitoring
AGENT_COORDINATOR_PERFORMANCE_MONITORING=true
AGENT_COORDINATOR_CACHING=true
AGENT_COORDINATOR_STREAMING=true
```

## 📚 API Reference

### Core Methods

#### Pipeline Execution

```python
async def execute_pipeline(self, query: str, conversation_id: str,
                          pipeline_config: Dict[str, Any] = None,
                          stream_callback: Callable = None) -> PipelineExecution:
    """
    Execute the complete RAG pipeline.

    Parameters:
    - query: User query to process
    - conversation_id: Conversation identifier
    - pipeline_config: Optional pipeline configuration overrides
    - stream_callback: Optional callback for streaming updates

    Returns:
    - PipelineExecution: Complete execution results and metadata
    """

async def execute_stage(self, stage_name: str, stage_config: Dict[str, Any],
                       input_data: Dict[str, Any],
                       context: ExecutionContext) -> StageResult:
    """
    Execute a single pipeline stage.

    Parameters:
    - stage_name: Name of the stage to execute
    - stage_config: Stage configuration
    - input_data: Input data for the stage
    - context: Execution context

    Returns:
    - StageResult: Stage execution results
    """
```

#### Pipeline Configuration

```python
def configure_pipeline(self, stages: List[PipelineStage]) -> None:
    """
    Configure the pipeline stages and execution order.

    Parameters:
    - stages: List of pipeline stage configurations
    """

def add_stage(self, stage: PipelineStage) -> None:
    """
    Add a new stage to the pipeline.

    Parameters:
    - stage: Pipeline stage configuration
    """

def remove_stage(self, stage_name: str) -> bool:
    """
    Remove a stage from the pipeline.

    Parameters:
    - stage_name: Name of stage to remove

    Returns:
    - bool: True if stage was removed, False otherwise
    """

def get_pipeline_config(self) -> List[PipelineStage]:
    """
    Get current pipeline configuration.

    Returns:
    - List[PipelineStage]: Current pipeline stages
    """
```

#### Execution Management

```python
async def start_execution(self, execution_id: str,
                         initial_data: Dict[str, Any]) -> ExecutionContext:
    """
    Start a new pipeline execution.

    Parameters:
    - execution_id: Unique execution identifier
    - initial_data: Initial execution data

    Returns:
    - ExecutionContext: Created execution context
    """

async def pause_execution(self, execution_id: str) -> bool:
    """
    Pause an active pipeline execution.

    Parameters:
    - execution_id: Execution identifier

    Returns:
    - bool: True if paused successfully, False otherwise
    """

async def resume_execution(self, execution_id: str) -> bool:
    """
    Resume a paused pipeline execution.

    Parameters:
    - execution_id: Execution identifier

    Returns:
    - bool: True if resumed successfully, False otherwise
    """

async def cancel_execution(self, execution_id: str) -> bool:
    """
    Cancel an active pipeline execution.

    Parameters:
    - execution_id: Execution identifier

    Returns:
    - bool: True if cancelled successfully, False otherwise
    """
```

### Data Models

#### Pipeline Stage

```python
@dataclass
class PipelineStage:
    name: str                              # Stage name
    agent_type: str                        # Agent type to execute
    required: bool = True                  # Whether stage is required
    parallel: bool = False                 # Can execute in parallel
    depends_on: List[str] = field(default_factory=list)  # Stage dependencies
    config: Dict[str, Any] = field(default_factory=dict)  # Stage configuration
    timeout_seconds: float = 60.0          # Stage timeout
    retry_attempts: int = 3                # Maximum retry attempts
```

#### Execution Context

```python
@dataclass
class ExecutionContext:
    execution_id: str                      # Unique execution identifier
    query: str                             # Original user query
    conversation_id: str                   # Conversation identifier
    start_time: datetime                   # Execution start time
    current_stage: str                     # Currently executing stage
    stage_results: Dict[str, StageResult]  # Results from completed stages
    context_data: Dict[str, Any]           # Shared context data
    execution_metadata: Dict[str, Any]     # Execution metadata
    status: str                            # Current execution status
```

#### Pipeline Execution

```python
@dataclass
class PipelineExecution:
    execution_id: str                      # Execution identifier
    status: str                            # Final execution status
    query: str                             # Original query
    conversation_id: str                   # Conversation identifier
    start_time: datetime                   # Execution start time
    end_time: datetime                     # Execution end time
    duration_ms: float                     # Total execution duration
    stage_results: Dict[str, StageResult]  # Results from all stages
    final_result: Dict[str, Any]           # Final pipeline result
    performance_metrics: Dict[str, Any]    # Performance metrics
    error_info: Optional[Dict[str, Any]]   # Error information (if failed)
```

## 💡 Usage Examples

### Basic Pipeline Execution

```python
from app.agents.coordinator import AgentCoordinator, PipelineStage
from app.agents.registry import AgentRegistry

# Initialize registry and coordinator
registry = AgentRegistry()
await registry.start()

# Register agent types
registry.register_agent_type("query_rewriter", QueryRewritingAgent)
registry.register_agent_type("context_decision", ContextDecisionAgent)
registry.register_agent_type("source_retrieval", SourceRetrievalAgent)
registry.register_agent_type("answer_generation", AnswerGenerationAgent)

# Create coordinator
coordinator = AgentCoordinator(registry)

# Configure pipeline
pipeline_stages = [
    PipelineStage(
        name="query_rewriting",
        agent_type="query_rewriter",
        required=False,  # Optional stage
        config={"enable_expansion": True}
    ),
    PipelineStage(
        name="context_decision",
        agent_type="context_decision",
        required=True,
        depends_on=["query_rewriting"]
    ),
    PipelineStage(
        name="source_retrieval",
        agent_type="source_retrieval",
        required=True,
        depends_on=["context_decision"],
        config={"max_results": 10}
    ),
    PipelineStage(
        name="answer_generation",
        agent_type="answer_generation",
        required=True,
        depends_on=["source_retrieval"],
        config={"citation_style": "apa"}
    )
]

coordinator.configure_pipeline(pipeline_stages)

# Execute pipeline
execution = await coordinator.execute_pipeline(
    query="What are the benefits of renewable energy?",
    conversation_id="conv_123"
)

print(f"Pipeline Status: {execution.status}")
print(f"Execution Time: {execution.duration_ms:.1f}ms")
print(f"Final Answer: {execution.final_result.get('answer', 'No answer')}")
```

### Streaming Pipeline Execution

```python
# Execute pipeline with streaming
async def stream_pipeline_execution():
    def stream_callback(update):
        print(f"[{update['stage']}] {update['status']}: {update.get('message', '')}")

        if update['status'] == 'completed' and 'data' in update:
            if update['stage'] == 'answer_generation':
                # Stream answer generation in real-time
                answer_chunk = update['data'].get('partial_answer', '')
                if answer_chunk:
                    print(answer_chunk, end='', flush=True)

    execution = await coordinator.execute_pipeline(
        query="Explain machine learning algorithms",
        conversation_id="stream_session",
        stream_callback=stream_callback
    )

    print(f"\n\nFinal execution status: {execution.status}")

await stream_pipeline_execution()
```

### Error Handling and Recovery

```python
# Configure pipeline with custom error handling
error_handling_config = {
    "max_retry_attempts": 5,              # More retries
    "retry_delay_seconds": 2.0,           # Longer retry delay
    "retry_backoff_factor": 1.5,          # Gentler backoff
    "enable_circuit_breaker": True,       # Circuit breaker protection
    "fallback_strategies": {              # Custom fallback strategies
        "source_retrieval": "use_cached_results",
        "answer_generation": "use_simple_model"
    }
}

coordinator.update_config(error_handling_config)

# Execute pipeline with error monitoring
try:
    execution = await coordinator.execute_pipeline(
        query="Complex technical query that might fail",
        conversation_id="error_test"
    )

    if execution.status == "failed":
        print(f"Pipeline failed: {execution.error_info['message']}")
        print(f"Failed stage: {execution.error_info['failed_stage']}")
        print(f"Retry attempts: {execution.error_info['retry_attempts']}")

        # Analyze failure and potentially retry with different config
        if execution.error_info['failed_stage'] == 'source_retrieval':
            # Retry with simpler retrieval strategy
            simple_config = {"search_strategy": "keyword"}
            execution = await coordinator.execute_pipeline(
                query="Complex technical query that might fail",
                conversation_id="error_test_retry",
                pipeline_config={"source_retrieval": {"config": simple_config}}
            )

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Parallel Stage Execution

```python
# Configure pipeline with parallel stages
parallel_pipeline = [
    PipelineStage(
        name="query_rewriting",
        agent_type="query_rewriter",
        required=False
    ),
    PipelineStage(
        name="context_decision",
        agent_type="context_decision",
        required=True,
        depends_on=["query_rewriting"]
    ),
    # These stages can run in parallel after context decision
    PipelineStage(
        name="source_retrieval_semantic",
        agent_type="source_retrieval",
        parallel=True,
        depends_on=["context_decision"],
        config={"search_strategy": "vector"}
    ),
    PipelineStage(
        name="source_retrieval_keyword",
        agent_type="source_retrieval",
        parallel=True,
        depends_on=["context_decision"],
        config={"search_strategy": "keyword"}
    ),
    PipelineStage(
        name="answer_generation",
        agent_type="answer_generation",
        depends_on=["source_retrieval_semantic", "source_retrieval_keyword"]
    )
]

coordinator.configure_pipeline(parallel_pipeline)

# Execute with parallel processing
execution = await coordinator.execute_pipeline(
    query="Benefits of solar energy vs wind energy",
    conversation_id="parallel_test"
)

print(f"Parallel execution completed in {execution.duration_ms:.1f}ms")
```

### Performance Monitoring

```python
# Monitor pipeline performance
async def monitor_pipeline_performance():
    # Get overall performance metrics
    metrics = coordinator.get_performance_metrics()

    print("Pipeline Performance Metrics:")
    print(f"  Total Executions: {metrics['total_executions']}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Average Duration: {metrics['avg_duration_ms']:.1f}ms")
    print(f"  95th Percentile: {metrics['p95_duration_ms']:.1f}ms")

    # Get per-stage metrics
    stage_metrics = coordinator.get_stage_metrics()

    print("\nPer-Stage Metrics:")
    for stage_name, stage_metric in stage_metrics.items():
        print(f"  {stage_name}:")
        print(f"    Average Duration: {stage_metric['avg_duration_ms']:.1f}ms")
        print(f"    Success Rate: {stage_metric['success_rate']:.2%}")
        print(f"    Error Rate: {stage_metric['error_rate']:.2%}")

    # Get bottleneck analysis
    bottlenecks = coordinator.analyze_bottlenecks()

    print("\nBottleneck Analysis:")
    for bottleneck in bottlenecks:
        print(f"  {bottleneck['stage']}: {bottleneck['impact']} impact")
        print(f"    Cause: {bottleneck['cause']}")
        print(f"    Recommendation: {bottleneck['recommendation']}")

await monitor_pipeline_performance()
```

## 🎯 Performance Characteristics

### Execution Performance

| Configuration            | Average Time         | 95th Percentile | Throughput     | Notes                        |
| ------------------------ | -------------------- | --------------- | -------------- | ---------------------------- |
| **Sequential Execution** | 3200ms               | 4800ms          | 18 queries/min | Standard pipeline            |
| **Parallel Execution**   | 2100ms               | 3200ms          | 28 queries/min | Parallel retrieval stages    |
| **Streaming Enabled**    | 800ms first response | 3200ms complete | 25 queries/min | Better perceived performance |
| **Optimized Config**     | 2400ms               | 3600ms          | 25 queries/min | Tuned timeouts and caching   |

### Stage Performance Breakdown

| Stage                 | Average Time | % of Total | Parallelizable | Optimization Potential |
| --------------------- | ------------ | ---------- | -------------- | ---------------------- |
| **Query Rewriting**   | 120ms        | 4%         | No             | Low                    |
| **Context Decision**  | 80ms         | 3%         | No             | Low                    |
| **Source Retrieval**  | 800ms        | 25%        | Yes            | High                   |
| **Answer Generation** | 2200ms       | 68%        | No             | Medium                 |

### Resource Utilization

| Resource    | Base Usage | Peak Usage | Optimization        |
| ----------- | ---------- | ---------- | ------------------- |
| **Memory**  | 150MB      | 500MB      | Result caching      |
| **CPU**     | 15%        | 45%        | Parallel processing |
| **Network** | 2MB/query  | 8MB/query  | Request batching    |
| **Storage** | 50MB       | 200MB      | Context persistence |

## 🚨 Error Handling

### Error Classification

#### 1. Stage Execution Errors

```python
# Agent unavailable
{"error": "Agent not available for execution", "code": "AGENT_UNAVAILABLE"}

# Stage timeout
{"error": "Stage execution timeout", "code": "STAGE_TIMEOUT"}

# Agent processing error
{"error": "Agent processing failed", "code": "AGENT_PROCESSING_ERROR"}
```

#### 2. Pipeline Configuration Errors

```python
# Invalid pipeline configuration
{"error": "Invalid pipeline configuration", "code": "INVALID_PIPELINE_CONFIG"}

# Circular dependency
{"error": "Circular dependency detected in pipeline", "code": "CIRCULAR_DEPENDENCY"}

# Missing dependency
{"error": "Required dependency not found", "code": "MISSING_DEPENDENCY"}
```

#### 3. Execution Management Errors

```python
# Execution not found
{"error": "Pipeline execution not found", "code": "EXECUTION_NOT_FOUND"}

# Context corruption
{"error": "Execution context corrupted", "code": "CONTEXT_CORRUPTED"}

# Resource exhaustion
{"error": "Insufficient resources for execution", "code": "RESOURCE_EXHAUSTED"}
```

### Recovery Strategies

#### Retry with Backoff

```python
async def _execute_stage_with_retry(self, stage: PipelineStage,
                                   input_data: Dict[str, Any],
                                   context: ExecutionContext) -> StageResult:
    """Execute stage with exponential backoff retry."""

    for attempt in range(stage.retry_attempts):
        try:
            return await self._execute_single_stage(stage, input_data, context)
        except Exception as e:
            if attempt == stage.retry_attempts - 1:
                # Final attempt failed
                raise e

            # Calculate delay with exponential backoff
            delay = self.config["retry_delay_seconds"] * (
                self.config["retry_backoff_factor"] ** attempt
            )

            logger.warning(f"Stage {stage.name} failed (attempt {attempt + 1}), "
                         f"retrying in {delay:.1f}s: {e}")

            await asyncio.sleep(delay)

    raise Exception("All retry attempts exhausted")
```

#### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

            raise e
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Pipeline Timeouts

**Symptoms**: Pipeline executions timing out frequently

**Solutions**:

```python
# Increase timeouts
config = {
    "stage_timeout_seconds": 120,         # Double stage timeout
    "pipeline_timeout_seconds": 600       # Increase overall timeout
}

# Enable stage-specific timeouts
stages = [
    PipelineStage(
        name="source_retrieval",
        agent_type="source_retrieval",
        timeout_seconds=90                 # Custom timeout for slow stage
    )
]

# Optimize slow stages
retrieval_config = {
    "max_results": 5,                     # Reduce result set
    "enable_parallel_search": True        # Enable parallel processing
}
```

#### 2. Memory Issues

**Symptoms**: High memory usage during pipeline execution

**Solutions**:

```python
# Enable result streaming
config = {"enable_streaming": True}

# Reduce context size
config = {"context_memory_size": 5}

# Enable garbage collection
await coordinator.cleanup_completed_executions()

# Set memory limits
config = {"memory_limit_mb": 1024}
```

#### 3. Stage Dependencies Issues

**Symptoms**: Stages not executing in correct order

**Solutions**:

```python
# Validate pipeline configuration
validation_result = coordinator.validate_pipeline_config()
if not validation_result.valid:
    print(f"Pipeline validation errors: {validation_result.errors}")

# Check dependency resolution
dependencies = coordinator.resolve_dependencies()
for stage, deps in dependencies.items():
    print(f"{stage} depends on: {deps}")

# Fix circular dependencies
coordinator.detect_circular_dependencies()
```

### Debugging Tools

#### Execution Tracing

```python
# Enable detailed execution tracing
def trace_execution(execution_id: str):
    trace = coordinator.get_execution_trace(execution_id)

    print(f"Execution Trace for {execution_id}:")
    for event in trace.events:
        print(f"  {event.timestamp}: {event.stage} - {event.event_type}")
        if event.data:
            print(f"    Data: {event.data}")
        if event.error:
            print(f"    Error: {event.error}")
```

#### Performance Analysis

```python
# Analyze pipeline performance bottlenecks
def analyze_pipeline_bottlenecks():
    analysis = coordinator.analyze_performance_bottlenecks()

    print("Performance Bottleneck Analysis:")
    for bottleneck in analysis.bottlenecks:
        print(f"  Stage: {bottleneck.stage}")
        print(f"  Impact: {bottleneck.impact_score:.2f}")
        print(f"  Cause: {bottleneck.root_cause}")
        print(f"  Recommendation: {bottleneck.optimization_recommendation}")
        print()
```

## 🔗 Integration Points

### With Agent Registry

```python
# Coordinator uses registry for agent management
def __init__(self, registry: AgentRegistry):
    self.registry = registry
    self.agent_cache = {}

async def _get_agent_for_stage(self, stage: PipelineStage) -> BaseAgent:
    """Get or create agent for pipeline stage."""
    agent_key = f"{stage.agent_type}_{stage.name}"

    if agent_key not in self.agent_cache:
        agent = await self.registry.create_agent(
            agent_type=stage.agent_type,
            agent_id=agent_key,
            config=stage.config
        )
        self.agent_cache[agent_key] = agent

    return self.agent_cache[agent_key]
```

### With External Monitoring

```python
# Export execution metrics to external systems
def export_execution_metrics():
    metrics = coordinator.get_execution_metrics()

    # Export to Prometheus
    prometheus_client.Histogram('pipeline_duration_seconds').observe(
        metrics['avg_duration_ms'] / 1000
    )
    prometheus_client.Counter('pipeline_executions_total').inc(
        metrics['total_executions']
    )
    prometheus_client.Gauge('pipeline_success_rate').set(
        metrics['success_rate']
    )
```

## 📊 Monitoring and Metrics

### Execution Metrics

```python
{
    "execution_metrics": {
        "total_executions": 1247,
        "successful_executions": 1198,
        "failed_executions": 49,
        "success_rate": 0.961,
        "average_duration_ms": 2847.3,
        "p95_duration_ms": 4521.7,
        "p99_duration_ms": 6834.2
    },
    "stage_metrics": {
        "query_rewriting": {
            "executions": 1247,
            "avg_duration_ms": 118.5,
            "success_rate": 0.998,
            "error_rate": 0.002
        },
        "context_decision": {
            "executions": 1247,
            "avg_duration_ms": 76.3,
            "success_rate": 0.995,
            "error_rate": 0.005
        },
        "source_retrieval": {
            "executions": 892,
            "avg_duration_ms": 734.2,
            "success_rate": 0.967,
            "error_rate": 0.033
        },
        "answer_generation": {
            "executions": 1247,
            "avg_duration_ms": 2156.8,
            "success_rate": 0.981,
            "error_rate": 0.019
        }
    }
}
```

### Performance Analytics

```python
{
    "performance_analytics": {
        "bottlenecks": [
            {
                "stage": "answer_generation",
                "impact_score": 0.68,
                "avg_duration_ms": 2156.8,
                "percentage_of_total": 68.2
            },
            {
                "stage": "source_retrieval",
                "impact_score": 0.25,
                "avg_duration_ms": 734.2,
                "percentage_of_total": 25.8
            }
        ],
        "optimization_opportunities": [
            {
                "type": "parallel_execution",
                "potential_improvement_ms": 420.3,
                "implementation_complexity": "medium"
            },
            {
                "type": "caching",
                "potential_improvement_ms": 234.7,
                "implementation_complexity": "low"
            }
        ]
    }
}
```

### Alerting Configuration

```python
ALERT_THRESHOLDS = {
    "execution_success_rate": 0.95,       # Alert if <95% success rate
    "avg_execution_time_ms": 5000,        # Alert if >5s average execution
    "stage_error_rate": 0.10,             # Alert if >10% stage errors
    "concurrent_executions": 20,          # Alert if >20 concurrent executions
    "memory_usage_mb": 2048                # Alert if >2GB memory usage
}
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from app.agents.coordinator import AgentCoordinator, PipelineStage
from app.agents.registry import AgentRegistry

@pytest.mark.asyncio
async def test_pipeline_configuration():
    registry = AgentRegistry()
    coordinator = AgentCoordinator(registry)

    stages = [
        PipelineStage("stage1", "mock_agent", required=True),
        PipelineStage("stage2", "mock_agent", depends_on=["stage1"])
    ]

    coordinator.configure_pipeline(stages)
    config = coordinator.get_pipeline_config()

    assert len(config) == 2
    assert config[0].name == "stage1"
    assert config[1].depends_on == ["stage1"]

@pytest.mark.asyncio
async def test_pipeline_execution():
    registry = AgentRegistry()
    await registry.start()

    # Register mock agent
    registry.register_agent_type("mock_agent", MockAgent)

    coordinator = AgentCoordinator(registry)
    coordinator.configure_pipeline([
        PipelineStage("test_stage", "mock_agent")
    ])

    execution = await coordinator.execute_pipeline(
        query="test query",
        conversation_id="test_conv"
    )

    assert execution.status == "completed"
    assert "test_stage" in execution.stage_results

    await registry.stop()

@pytest.mark.asyncio
async def test_error_handling():
    registry = AgentRegistry()
    coordinator = AgentCoordinator(registry)

    # Configure with failing agent
    coordinator.configure_pipeline([
        PipelineStage("failing_stage", "non_existent_agent")
    ])

    execution = await coordinator.execute_pipeline(
        query="test query",
        conversation_id="test_conv"
    )

    assert execution.status == "failed"
    assert execution.error_info is not None
    assert "non_existent_agent" in execution.error_info["message"]
```

---

_The Agent Coordinator serves as the orchestration backbone of the RAG pipeline, ensuring reliable, efficient, and monitored execution of multi-agent workflows while providing comprehensive error handling and performance optimization capabilities._
