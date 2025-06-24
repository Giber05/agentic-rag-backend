# Answer Generation Agent

## 📋 Overview

The Answer Generation Agent is the fourth and final required stage in the RAG pipeline, responsible for generating comprehensive, well-cited responses using the retrieved sources and conversation context. It combines advanced language models with sophisticated citation management to produce high-quality answers.

### Purpose

Generate high-quality responses through:

- **Context-Aware Generation**: Synthesize information from multiple sources
- **Citation Management**: Provide accurate citations in multiple academic formats
- **Response Optimization**: Ensure clarity, coherence, and completeness
- **Quality Assessment**: Evaluate response quality and accuracy
- **Multi-Format Support**: Generate responses in various formats (text, markdown, structured)
- **Streaming Support**: Enable real-time response streaming for better UX

### When Used

- **Pipeline Position**: Fourth stage (required)
- **Input Sources**: Query, retrieved sources, conversation context
- **Trigger Condition**: Always executed (with or without retrieved sources)
- **Output**: Formatted response with citations and metadata

## 🏗️ Architecture

### Class Structure

```python
class AnswerGenerationAgent(BaseAgent):
    """
    Agent responsible for generating comprehensive answers with proper citations.

    Capabilities:
    - Multi-source information synthesis
    - Citation generation and management
    - Response quality assessment
    - Multiple output formats
    - Streaming response generation
    - Quality control and verification
    """
```

### Core Components

#### 1. Context Synthesizer

- **Information Integration**: Combine insights from multiple sources
- **Relevance Filtering**: Focus on most relevant source content
- **Coherence Building**: Create logical flow between source materials
- **Gap Identification**: Identify missing information or contradictions

#### 2. Response Generator

- **LLM Integration**: Use advanced language models for generation
- **Prompt Engineering**: Optimize prompts for different response types
- **Temperature Control**: Adjust creativity vs accuracy balance
- **Length Management**: Control response length based on requirements

#### 3. Citation Manager

- **Source Attribution**: Track which sources contribute to each claim
- **Citation Formatting**: Support multiple academic citation styles
- **Reference Management**: Maintain accurate reference lists
- **Link Preservation**: Keep source URLs and metadata intact

#### 4. Quality Controller

- **Accuracy Assessment**: Verify claims against source material
- **Completeness Check**: Ensure all aspects of query are addressed
- **Coherence Evaluation**: Assess logical flow and readability
- **Citation Verification**: Confirm all claims are properly cited

#### 5. Format Processor

- **Output Formatting**: Convert responses to requested formats
- **Markdown Generation**: Create well-structured markdown output
- **JSON Structuring**: Provide structured data responses
- **HTML Rendering**: Generate web-ready formatted responses

#### 6. Streaming Handler

- **Real-time Generation**: Support streaming response delivery
- **Partial Citation**: Handle citations in streaming context
- **Buffer Management**: Optimize streaming performance
- **Error Recovery**: Handle streaming interruptions gracefully

## 🔧 Configuration

### Agent Configuration

```python
config = {
    # Generation settings
    "model": "gpt-4",                       # Primary generation model
    "fallback_model": "gpt-3.5-turbo",     # Fallback model
    "temperature": 0.2,                     # Creativity vs accuracy balance
    "max_tokens": 2000,                     # Maximum response length

    # Citation settings
    "citation_style": "apa",                # Default citation format
    "enable_inline_citations": True,        # Include [1], [2] style citations
    "enable_reference_list": True,          # Include full reference list
    "citation_formats": ["apa", "mla", "chicago"],  # Supported formats

    # Quality control
    "enable_quality_check": True,           # Enable quality assessment
    "min_confidence_score": 0.7,           # Minimum confidence threshold
    "require_citations": True,              # Require citations for claims
    "max_unsupported_claims": 2,           # Max claims without citations

    # Response formatting
    "default_format": "markdown",           # Default output format
    "include_metadata": True,               # Include generation metadata
    "enable_structured_output": True,       # Support structured responses

    # Streaming settings
    "enable_streaming": True,               # Enable streaming responses
    "stream_buffer_size": 50,               # Streaming buffer size (tokens)
    "stream_timeout_ms": 5000,              # Streaming timeout

    # Performance settings
    "timeout_seconds": 30.0,                # Processing timeout
    "cache_ttl": 1800,                      # Cache time-to-live (30 minutes)
    "parallel_processing": True             # Enable parallel operations
}
```

### Environment Variables

```bash
# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_PRIMARY=gpt-4
OPENAI_MODEL_FALLBACK=gpt-3.5-turbo

# Answer generation settings
ANSWER_GENERATION_ENABLED=true
ANSWER_GENERATION_MAX_TOKENS=2000
ANSWER_GENERATION_TEMPERATURE=0.2
ANSWER_GENERATION_STREAMING=true

# Citation settings
CITATION_STYLE_DEFAULT=apa
CITATION_INLINE_ENABLED=true
CITATION_FORMATS=apa,mla,chicago
```

## 📚 API Reference

### Core Methods

#### `process(input_data: Dict[str, Any]) -> AgentResult`

Main processing method that generates comprehensive answers.

**Parameters:**

```python
input_data = {
    "query": str,                           # Required: User query
    "sources": List[Dict[str, Any]],        # Optional: Retrieved sources
    "conversation_id": str,                 # Required: Conversation context
    "conversation_history": List[Dict],     # Optional: Previous exchanges
    "response_format": str,                 # Optional: Output format
    "citation_style": str,                  # Optional: Citation format
    "max_tokens": int,                      # Optional: Response length limit
    "config": Dict[str, Any]                # Optional: Configuration overrides
}
```

**Returns:**

```python
{
    "answer": str,                          # Generated response
    "citations": {                          # Citation information
        "inline_citations": List[str],      # [1], [2] style citations used
        "references": List[Dict[str, Any]],  # Full reference list
        "citation_map": Dict[str, List[int]] # Claim to citation mapping
    },
    "sources_used": List[str],              # Source IDs referenced in answer
    "quality_metrics": {                    # Quality assessment results
        "confidence_score": float,          # Overall confidence (0.0-1.0)
        "completeness_score": float,        # Query coverage assessment
        "accuracy_score": float,            # Source accuracy alignment
        "citation_score": float,            # Citation quality score
        "coherence_score": float            # Response coherence rating
    },
    "generation_metadata": {                # Generation process details
        "model_used": str,                  # LLM model used
        "tokens_generated": int,            # Number of tokens in response
        "generation_time_ms": float,        # Time to generate response
        "prompt_tokens": int,               # Tokens in input prompt
        "sources_processed": int,           # Number of sources used
        "fallback_used": bool               # Whether fallback model was used
    },
    "structured_data": Dict[str, Any],      # Structured response data (optional)
    "alternatives": List[str]               # Alternative phrasings (optional)
}
```

### Generation Pipeline

#### 1. Context Preparation

```python
def _prepare_generation_context(self, query: str, sources: List[Dict],
                               conversation_history: List[Dict] = None) -> str:
    """
    Prepare context for answer generation.

    Process:
    - Organize sources by relevance
    - Extract key information from each source
    - Format conversation history appropriately
    - Create structured context for LLM

    Returns formatted context string for generation.
    """
```

#### 2. Prompt Construction

```python
def _build_generation_prompt(self, query: str, context: str,
                            citation_style: str = "apa") -> str:
    """
    Construct optimized prompt for answer generation.

    Components:
    - System instructions for response quality
    - Citation format specifications
    - Query and context integration
    - Output format requirements

    Returns complete prompt for LLM generation.
    """
```

#### 3. Response Generation

```python
async def _generate_response(self, prompt: str,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate response using language model.

    Process:
    - Send prompt to configured LLM
    - Handle streaming if enabled
    - Apply fallback model if needed
    - Parse and validate response

    Returns generated response with metadata.
    """
```

#### 4. Citation Processing

```python
def _process_citations(self, response: str, sources: List[Dict],
                      citation_style: str = "apa") -> Dict[str, Any]:
    """
    Extract and format citations from generated response.

    Process:
    - Identify citation markers in response ([1], [2], etc.)
    - Match citations to source materials
    - Generate formatted references
    - Create citation mapping

    Returns citation data with formatted references.
    """
```

#### 5. Quality Assessment

```python
def _assess_response_quality(self, query: str, response: str,
                           sources: List[Dict],
                           citations: Dict[str, Any]) -> Dict[str, float]:
    """
    Assess the quality of generated response.

    Metrics:
    - Confidence: Overall reliability assessment
    - Completeness: Query coverage evaluation
    - Accuracy: Source alignment verification
    - Citation Quality: Citation appropriateness
    - Coherence: Logical flow and readability

    Returns quality scores (0.0-1.0 for each metric).
    """
```

#### 6. Format Processing

```python
def _format_response(self, response: str, format_type: str,
                    citations: Dict[str, Any],
                    metadata: Dict[str, Any]) -> str:
    """
    Format response according to requested output format.

    Supported formats:
    - markdown: Well-structured markdown with citations
    - json: Structured JSON response
    - html: Web-ready HTML with formatted citations
    - plain: Clean text without formatting

    Returns formatted response string.
    """
```

### Streaming Methods

#### Stream Handler

```python
async def _stream_generate_response(self, prompt: str,
                                   callback: Callable[[str], None]) -> str:
    """
    Generate response with real-time streaming.

    Process:
    - Initialize streaming connection
    - Process tokens as they're generated
    - Call callback for each token batch
    - Handle citations in streaming context

    Returns complete response when streaming finishes.
    """
```

## 💡 Usage Examples

### Basic Usage

```python
from app.agents.answer_generation import AnswerGenerationAgent

# Initialize agent
agent = AnswerGenerationAgent(
    agent_id="answer-gen-1",
    config={
        "model": "gpt-4",
        "citation_style": "apa",
        "enable_streaming": False
    }
)

# Start agent
await agent.start()

# Generate answer with sources
input_data = {
    "query": "What are the main applications of machine learning in healthcare?",
    "sources": [
        {
            "id": "src1",
            "title": "ML in Medical Diagnosis",
            "content": "Machine learning algorithms can analyze medical images...",
            "url": "https://example.com/ml-diagnosis",
            "metadata": {"author": "Dr. Smith", "year": "2023"}
        },
        {
            "id": "src2",
            "title": "Predictive Analytics in Healthcare",
            "content": "Predictive models help identify patients at risk...",
            "url": "https://example.com/predictive-analytics",
            "metadata": {"author": "Dr. Johnson", "year": "2023"}
        }
    ],
    "conversation_id": "conv_123"
}

result = await agent.process(input_data)

print("Generated Answer:")
print(result.data["answer"])
print(f"\nQuality Score: {result.data['quality_metrics']['confidence_score']:.2f}")
print(f"Sources Used: {len(result.data['sources_used'])}")

# Expected output:
# Generated Answer:
# Machine learning has several key applications in healthcare:
#
# 1. **Medical Diagnosis**: ML algorithms can analyze medical images
#    to assist in diagnostic processes [1].
#
# 2. **Predictive Analytics**: Predictive models help identify patients
#    at risk for various conditions [2].
#
# ## References
# [1] Dr. Smith. (2023). ML in Medical Diagnosis. https://example.com/ml-diagnosis
# [2] Dr. Johnson. (2023). Predictive Analytics in Healthcare. https://example.com/predictive-analytics
#
# Quality Score: 0.89
# Sources Used: 2
```

### Streaming Response

```python
# Generate streaming response
async def generate_streaming_answer(agent, input_data):
    response_chunks = []

    def stream_callback(chunk: str):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)

    # Enable streaming
    input_data["config"] = {"enable_streaming": True}

    print("Streaming response:")
    result = await agent.process(input_data, stream_callback=stream_callback)
    print("\n\nStreaming complete!")

    return result

# Usage
result = await generate_streaming_answer(agent, input_data)
```

### Multiple Citation Formats

```python
# Generate response with multiple citation formats
async def generate_multi_format_citations(agent, input_data):
    formats = ["apa", "mla", "chicago"]
    responses = {}

    for citation_format in formats:
        format_input = {**input_data, "citation_style": citation_format}
        result = await agent.process(format_input)
        responses[citation_format] = result.data["citations"]["references"]

    return responses

# Compare citation formats
citations = await generate_multi_format_citations(agent, input_data)
for format_name, refs in citations.items():
    print(f"\n{format_name.upper()} Format:")
    for ref in refs:
        print(f"  {ref}")
```

### Quality-Controlled Generation

```python
# Generate response with quality validation
async def generate_quality_controlled_response(agent, input_data,
                                               min_quality: float = 0.8):
    max_attempts = 3
    attempt = 1

    while attempt <= max_attempts:
        result = await agent.process(input_data)

        quality_score = result.data["quality_metrics"]["confidence_score"]

        if quality_score >= min_quality:
            print(f"✅ Quality threshold met (score: {quality_score:.2f})")
            return result

        print(f"⚠️  Attempt {attempt}: Quality below threshold ({quality_score:.2f})")

        # Adjust configuration for better quality
        if attempt < max_attempts:
            input_data["config"] = {
                "temperature": max(0.1, input_data.get("config", {}).get("temperature", 0.2) - 0.05),
                "max_tokens": min(3000, input_data.get("config", {}).get("max_tokens", 2000) + 500)
            }

        attempt += 1

    print(f"❌ Failed to meet quality threshold after {max_attempts} attempts")
    return result

# Usage
quality_result = await generate_quality_controlled_response(agent, input_data, 0.85)
```

## 🎯 Performance Characteristics

### Generation Performance

| Configuration             | Average Time      | 95th Percentile | Token Rate    | Notes                |
| ------------------------- | ----------------- | --------------- | ------------- | -------------------- |
| **GPT-4 Standard**        | 2800ms            | 4200ms          | 35 tokens/sec | Highest quality      |
| **GPT-3.5-turbo**         | 1200ms            | 2100ms          | 85 tokens/sec | Good quality, faster |
| **With Streaming**        | 600ms first token | 2800ms complete | 45 tokens/sec | Better UX            |
| **Quality Check Enabled** | +400ms            | +600ms          | -10%          | Quality overhead     |

### Quality Metrics

| Metric                 | GPT-4 | GPT-3.5-turbo | Notes                           |
| ---------------------- | ----- | ------------- | ------------------------------- |
| **Accuracy Score**     | 0.91  | 0.84          | Source alignment accuracy       |
| **Completeness Score** | 0.89  | 0.82          | Query coverage completeness     |
| **Citation Quality**   | 0.94  | 0.87          | Citation accuracy and relevance |
| **Coherence Score**    | 0.92  | 0.86          | Response logical flow           |
| **User Satisfaction**  | 4.6/5 | 4.2/5         | Human evaluation ratings        |

### Cache Performance

| Cache Type             | Hit Rate | TTL        | Impact                |
| ---------------------- | -------- | ---------- | --------------------- |
| **Complete Responses** | 45%      | 30 minutes | -95% generation time  |
| **Partial Contexts**   | 68%      | 1 hour     | -40% preparation time |
| **Citation Formats**   | 85%      | 2 hours    | -90% citation time    |

## 🚨 Error Handling

### Common Error Scenarios

#### 1. Generation Failures

```python
# Model unavailable
{"error": "Language model service unavailable", "code": "LLM_SERVICE_ERROR"}

# Token limit exceeded
{"error": "Response exceeds maximum token limit", "code": "TOKEN_LIMIT_EXCEEDED"}

# Generation timeout
{"error": "Response generation timeout", "code": "GENERATION_TIMEOUT"}
```

#### 2. Quality Issues

```python
# Low confidence response
{"warning": "Generated response below confidence threshold", "code": "LOW_CONFIDENCE"}

# Missing citations
{"warning": "Response contains unsupported claims", "code": "MISSING_CITATIONS"}

# Poor source alignment
{"warning": "Response poorly aligned with source material", "code": "SOURCE_MISALIGNMENT"}
```

### Error Recovery Strategies

#### Model Fallback

```python
async def _generate_with_fallback(self, prompt: str) -> Dict[str, Any]:
    """Generate response with automatic model fallback."""

    # Try primary model
    try:
        return await self._call_llm(prompt, self.config["model"])
    except Exception as e:
        logger.warning(f"Primary model failed: {e}")

        # Fallback to secondary model
        try:
            fallback_model = self.config.get("fallback_model", "gpt-3.5-turbo")
            result = await self._call_llm(prompt, fallback_model)
            result["metadata"]["fallback_used"] = True
            return result
        except Exception as e2:
            logger.error(f"Fallback model also failed: {e2}")
            raise e2
```

#### Quality Recovery

```python
def _handle_low_quality_response(self, response: str, sources: List[Dict],
                                quality_scores: Dict[str, float]) -> str:
    """Improve low-quality responses through post-processing."""

    if quality_scores["citation_score"] < 0.7:
        # Add missing citations
        response = self._add_missing_citations(response, sources)

    if quality_scores["completeness_score"] < 0.7:
        # Add clarifying information
        response = self._enhance_completeness(response, sources)

    if quality_scores["coherence_score"] < 0.7:
        # Improve structure and flow
        response = self._improve_coherence(response)

    return response
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Poor Citation Quality

**Symptoms**: Missing citations, incorrect references, broken citation formatting

**Solutions**:

```python
# Strengthen citation requirements
config = {
    "require_citations": True,
    "max_unsupported_claims": 0,          # No unsupported claims allowed
    "citation_verification": "strict"     # Strict citation validation
}

# Improve citation prompting
citation_prompt_addition = """
CRITICAL: Every factual claim must be cited with [1], [2], etc.
Each citation number must correspond to a source in the provided materials.
Claims without proper citations will be rejected.
"""
```

#### 2. Inconsistent Response Quality

**Symptoms**: Quality varies significantly between similar queries

**Solutions**:

```python
# Reduce temperature for consistency
config = {"temperature": 0.1}

# Enable quality validation
config = {
    "enable_quality_check": True,
    "min_confidence_score": 0.8,
    "quality_retry_attempts": 2
}

# Use more structured prompting
structured_prompt = True
deterministic_seed = 42
```

#### 3. Slow Response Generation

**Symptoms**: Generation taking longer than acceptable (>5 seconds)

**Solutions**:

```python
# Switch to faster model
config = {"model": "gpt-3.5-turbo"}

# Reduce maximum tokens
config = {"max_tokens": 1000}

# Enable aggressive caching
config = {"cache_ttl": 3600}

# Use streaming for perceived performance
config = {"enable_streaming": True}
```

#### 4. High API Costs

**Symptoms**: OpenAI usage costs higher than expected

**Solutions**:

```python
# Optimize token usage
config = {
    "max_tokens": 1500,                    # Reduce max response length
    "temperature": 0.1,                    # Reduce variability
    "model": "gpt-3.5-turbo"              # Use cheaper model
}

# Implement smart caching
config = {
    "cache_ttl": 7200,                     # 2-hour cache
    "cache_similar_queries": True          # Cache similar query responses
}

# Batch process when possible
def should_batch_generation(queries: List[str]) -> bool:
    return len(queries) > 1 and all(len(q.split()) < 10 for q in queries)
```

### Debugging Tools

#### Response Analysis

```python
# Analyze response quality factors
async def analyze_response_quality(agent, input_data):
    result = await agent.process(input_data)

    quality = result.data["quality_metrics"]
    metadata = result.data["generation_metadata"]

    print("Quality Analysis:")
    print(f"  Overall Confidence: {quality['confidence_score']:.3f}")
    print(f"  Completeness: {quality['completeness_score']:.3f}")
    print(f"  Accuracy: {quality['accuracy_score']:.3f}")
    print(f"  Citation Quality: {quality['citation_score']:.3f}")
    print(f"  Coherence: {quality['coherence_score']:.3f}")

    print(f"\nGeneration Metadata:")
    print(f"  Model Used: {metadata['model_used']}")
    print(f"  Tokens Generated: {metadata['tokens_generated']}")
    print(f"  Generation Time: {metadata['generation_time_ms']:.1f}ms")
    print(f"  Sources Processed: {metadata['sources_processed']}")

    return quality, metadata
```

#### Citation Verification

```python
# Verify citation accuracy
def verify_citations(response: str, sources: List[Dict],
                    citations: Dict[str, Any]):
    citation_issues = []

    # Check for uncited claims
    sentences = response.split('.')
    cited_sentences = [s for s in sentences if '[' in s and ']' in s]
    uncited_sentences = [s for s in sentences if '[' not in s and ']' not in s]

    if len(uncited_sentences) > len(cited_sentences):
        citation_issues.append("Many sentences lack citations")

    # Verify citation numbers exist
    citation_numbers = re.findall(r'\[(\d+)\]', response)
    max_citation = max([int(n) for n in citation_numbers], default=0)

    if max_citation > len(sources):
        citation_issues.append(f"Citation [{max_citation}] exceeds source count")

    return citation_issues
```

## 🔗 Integration Points

### With Other Agents

#### Source Retrieval Agent

```python
# Receives sources from retrieval stage
retrieval_result = pipeline_context.get_result("source_retrieval")
if retrieval_result and retrieval_result.success:
    sources = retrieval_result.data["sources"]
    generation_input["sources"] = sources
```

#### Validation & Refinement Agent (Future)

```python
# Passes generated answer for validation
validation_input = {
    "answer": generation_result.data["answer"],
    "query": original_query,
    "sources": sources_used,
    "quality_metrics": generation_result.data["quality_metrics"]
}
```

### External Services

#### OpenAI Integration

- **Models**: GPT-4, GPT-3.5-turbo for generation
- **Streaming**: Real-time response streaming
- **Rate Limiting**: Handled through service layer
- **Cost Optimization**: Intelligent caching and model selection

#### Citation Services

- **Format Libraries**: Support for multiple citation standards
- **DOI Resolution**: Automatic DOI lookup for academic sources
- **URL Validation**: Verify source URL accessibility

## 📊 Monitoring and Metrics

### Generation Metrics

```python
{
    "generation_operations": {
        "total_responses_generated": 4521,
        "gpt4_generations": 2847,
        "gpt35_generations": 1674,
        "streaming_responses": 3102,
        "fallback_responses": 87
    },
    "performance": {
        "average_generation_time_ms": 2341.7,
        "average_tokens_per_response": 1247,
        "token_generation_rate": 42.3,
        "streaming_first_token_ms": 543.2
    },
    "quality_metrics": {
        "average_confidence_score": 0.84,
        "average_citation_quality": 0.89,
        "responses_above_threshold": 0.91,
        "user_satisfaction_rating": 4.3
    }
}
```

### Citation Analytics

```python
{
    "citation_usage": {
        "responses_with_citations": 0.87,
        "average_citations_per_response": 3.2,
        "citation_accuracy_rate": 0.94,
        "most_used_citation_style": "apa"
    },
    "citation_formats": {
        "apa_usage": 0.52,
        "mla_usage": 0.31,
        "chicago_usage": 0.17
    }
}
```

### Alerting Configuration

```python
ALERT_THRESHOLDS = {
    "generation_time_ms": 5000,         # Alert if >5s average
    "success_rate": 0.95,               # Alert if <95% success
    "quality_score": 0.75,              # Alert if <0.75 average quality
    "citation_accuracy": 0.85,          # Alert if <85% citation accuracy
    "api_error_rate": 0.05              # Alert if >5% API errors
}
```

## 🧪 Testing

### Unit Tests

```python
import pytest
from app.agents.answer_generation import AnswerGenerationAgent

@pytest.mark.asyncio
async def test_basic_answer_generation():
    agent = AnswerGenerationAgent("test-agent")
    await agent.start()

    input_data = {
        "query": "What is machine learning?",
        "sources": [
            {
                "id": "src1",
                "title": "Introduction to ML",
                "content": "Machine learning is a subset of artificial intelligence...",
                "metadata": {"author": "Test Author"}
            }
        ],
        "conversation_id": "test-conv"
    }

    result = await agent.process(input_data)

    assert result.success
    assert "machine learning" in result.data["answer"].lower()
    assert len(result.data["citations"]["references"]) > 0
    assert result.data["quality_metrics"]["confidence_score"] > 0.0

    await agent.stop()

@pytest.mark.asyncio
async def test_citation_generation():
    agent = AnswerGenerationAgent("test-agent")
    await agent.start()

    input_data = {
        "query": "Explain neural networks",
        "sources": [
            {"id": "src1", "title": "Neural Networks Basics", "content": "Neural networks are..."},
            {"id": "src2", "title": "Deep Learning", "content": "Deep learning uses multiple layers..."}
        ],
        "conversation_id": "test-conv",
        "citation_style": "apa"
    }

    result = await agent.process(input_data)

    # Verify citations exist
    assert result.success
    assert len(result.data["citations"]["references"]) == 2
    assert "[1]" in result.data["answer"] or "[2]" in result.data["answer"]

    await agent.stop()

@pytest.mark.asyncio
async def test_quality_assessment():
    agent = AnswerGenerationAgent("test-agent")
    await agent.start()

    # Test with good sources
    input_data = {
        "query": "Benefits of renewable energy",
        "sources": [
            {
                "id": "src1",
                "title": "Renewable Energy Benefits",
                "content": "Solar and wind energy provide clean electricity without emissions...",
                "metadata": {"author": "Energy Expert"}
            }
        ],
        "conversation_id": "test-conv"
    }

    result = await agent.process(input_data)

    quality = result.data["quality_metrics"]
    assert quality["confidence_score"] > 0.7
    assert quality["completeness_score"] > 0.0
    assert quality["accuracy_score"] > 0.0
    assert quality["citation_score"] > 0.0

    await agent.stop()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_pipeline_answer_generation():
    from app.agents.coordinator import AgentCoordinator
    from app.agents.registry import AgentRegistry

    registry = AgentRegistry()
    registry.register_agent_type("answer_generation", AnswerGenerationAgent)

    coordinator = AgentCoordinator(registry)

    execution = await coordinator.execute_pipeline(
        query="How does photosynthesis work?",
        conversation_id="integration-test"
    )

    assert execution.status == "completed"
    answer_result = execution.step_results["answer_generation"]
    assert answer_result.success

    answer_data = answer_result.data
    assert "photosynthesis" in answer_data["answer"].lower()
    assert "quality_metrics" in answer_data
    assert "citations" in answer_data

    # Verify quality standards
    assert answer_data["quality_metrics"]["confidence_score"] > 0.6
```

---

_The Answer Generation Agent serves as the synthesis engine of the RAG pipeline, combining retrieved knowledge with advanced language generation to produce comprehensive, well-cited, and high-quality responses._
