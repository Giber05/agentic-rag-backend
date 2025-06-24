"""
Answer Generation Agent for generating responses with source citations.

This agent handles:
- Response generation using retrieved context
- Source citation and attribution
- Response formatting and structuring
- Context integration strategies
- Response quality assessment
- Streaming response support
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib

from .base import BaseAgent
from ..utils.language_detector import detect_query_language, get_language_instruction_for_query
from ..core.openai_config import OpenAIModels

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Enumeration for response formats."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    JSON = "json"


class CitationStyle(Enum):
    """Enumeration for citation styles."""
    NUMBERED = "numbered"
    BRACKETED = "bracketed"
    FOOTNOTE = "footnote"
    INLINE = "inline"


class ResponseQuality:
    """Class for managing response quality metrics."""
    
    def __init__(
        self,
        relevance_score: float = 0.0,
        coherence_score: float = 0.0,
        completeness_score: float = 0.0,
        citation_accuracy: float = 0.0,
        factual_accuracy: float = 0.0
    ):
        self.relevance_score = relevance_score
        self.coherence_score = coherence_score
        self.completeness_score = completeness_score
        self.citation_accuracy = citation_accuracy
        self.factual_accuracy = factual_accuracy
    
    @property
    def overall_quality(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'relevance': 0.3,
            'coherence': 0.2,
            'completeness': 0.2,
            'citation': 0.15,
            'factual': 0.15
        }
        
        return (
            self.relevance_score * weights['relevance'] +
            self.coherence_score * weights['coherence'] +
            self.completeness_score * weights['completeness'] +
            self.citation_accuracy * weights['citation'] +
            self.factual_accuracy * weights['factual']
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            'relevance_score': self.relevance_score,
            'coherence_score': self.coherence_score,
            'completeness_score': self.completeness_score,
            'citation_accuracy': self.citation_accuracy,
            'factual_accuracy': self.factual_accuracy,
            'overall_quality': self.overall_quality
        }


class GeneratedResponse:
    """Class representing a generated response."""
    
    def __init__(
        self,
        content: str,
        citations: List[Dict[str, Any]],
        quality: ResponseQuality,
        format_type: ResponseFormat = ResponseFormat.MARKDOWN,
        metadata: Optional[Dict[str, Any]] = None,
        streaming_chunks: Optional[List[str]] = None
    ):
        self.content = content
        self.citations = citations
        self.quality = quality
        self.format_type = format_type
        self.metadata = metadata or {}
        self.streaming_chunks = streaming_chunks or []
        self.generated_at = datetime.utcnow()
        self.word_count = len(content.split())
        self.character_count = len(content)
    
    @property
    def response_hash(self) -> str:
        """Generate hash for response deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'content': self.content,
            'citations': self.citations,
            'quality': self.quality.to_dict(),
            'format_type': self.format_type.value,
            'metadata': self.metadata,
            'generated_at': self.generated_at.isoformat(),
            'word_count': self.word_count,
            'character_count': self.character_count,
            'response_hash': self.response_hash
        }


class AnswerGenerationAgent(BaseAgent):
    """
    Agent responsible for generating responses with source citations.
    
    Capabilities:
    - Response generation using LLM with retrieved context
    - Source citation and attribution in multiple styles
    - Response formatting and structuring
    - Context integration strategies
    - Response quality assessment
    - Streaming response support
    - Response caching and optimization
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "answer_generation",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Answer Generation Agent."""
        super().__init__(agent_id, agent_type, config)
        
        # Configuration
        self.max_response_length = config.get("max_response_length", 2000) if config else 600
        self.min_response_length = config.get("min_response_length", 50) if config else 50
        self.citation_style = CitationStyle(config.get("citation_style", "numbered")) if config else CitationStyle.NUMBERED
        self.response_format = ResponseFormat(config.get("response_format", "markdown")) if config else ResponseFormat.MARKDOWN
        self.enable_streaming = config.get("enable_streaming", True) if config else True
        self.quality_threshold = config.get("quality_threshold", 0.7) if config else 0.7
        self.max_citations = config.get("max_citations", 10) if config else 10
        self.enable_quality_assessment = config.get("enable_quality_assessment", True) if config else True
        self.temperature = config.get("temperature", 0.7) if config else 0.7
        self.model_name = config.get("model_name", OpenAIModels.GPT_4_1_MINI) if config else OpenAIModels.GPT_4_1_NANO
        
        # Services (will be initialized on start)
        self.openai_service = None
        
        # Cache for recent responses
        self.response_cache = {}
        self.cache_ttl = 600  # 10 minutes
        
        # Performance tracking
        self.generation_stats = {
            'total_generations': 0,
            'cache_hits': 0,
            'streaming_responses': 0,
            'avg_response_length': 0.0,
            'avg_generation_time': 0.0,
            'avg_quality_score': 0.0,
            'citation_accuracy': 0.0
        }
        
        logger.info(f"Answer Generation Agent {self.agent_id} initialized")
    
    async def _on_start(self) -> None:
        """Initialize services when agent starts."""
        try:
            # Import here to avoid circular imports
            from ..core.dependencies import get_openai_service
            
            self.openai_service = get_openai_service()
            
            logger.info(f"Answer Generation Agent {self.agent_id} connected to services")
        except Exception as e:
            logger.warning(f"Could not connect to OpenAI service: {str(e)}")
            # Don't raise the exception, just log it and continue
            self.openai_service = None
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate response with citations based on retrieved sources.
        
        Args:
            input_data: Contains query, sources, and generation parameters
            
        Returns:
            Dictionary with generated response and metadata
        """
        query = input_data.get("query", "").strip()
        sources = input_data.get("sources", [])
        conversation_history = input_data.get("conversation_history", [])
        generation_config = input_data.get("generation_config", {})
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, sources, generation_config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.generation_stats['cache_hits'] += 1
            return cached_result
        
        # Generate response
        if self.enable_streaming and generation_config.get("stream", False):
            response = await self._generate_streaming_response(
                query, sources, conversation_history, generation_config
            )
        else:
            response = await self._generate_response(
                query, sources, conversation_history, generation_config
            )
        
        # Assess quality if enabled
        if self.enable_quality_assessment:
            response.quality = await self._assess_response_quality(
                query, response, sources
            )
        
        # Prepare result
        result = {
            "query": query,
            "response": response.to_dict(),
            "sources_used": len(sources),
            "generation_metadata": {
                "processing_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "model_used": self.model_name,
                "cache_hit": False,
                "performance_stats": self._get_performance_stats()
            }
        }
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        # Update statistics
        self._update_stats(response)
        
        logger.debug(
            f"Generated response ({response.word_count} words) with "
            f"{len(response.citations)} citations for query: '{query[:50]}...'"
        )
        
        return result
    
    async def _generate_response(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        generation_config: Dict[str, Any]
    ) -> GeneratedResponse:
        """Generate a complete response using LLM."""
        
        # Extract config with fallbacks to instance defaults
        model_name = generation_config.get("model", self.model_name)
        temperature = generation_config.get("temperature", self.temperature)
        max_tokens = generation_config.get("max_tokens", self.max_response_length)
        
        # Prepare context from sources
        context = self._prepare_context(sources)
        
        # Build prompt
        prompt = self._build_generation_prompt(
            query, context, conversation_history, generation_config
        )
        
        # Generate response using OpenAI
        if self.openai_service:
            try:
                response = await self.openai_service.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,  # Use config override
                    temperature=temperature,  # Use config override
                    max_tokens=max_tokens  # Use config override
                )
                # Extract the actual text content from the ChatCompletion object
                response_text = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI generation failed: {str(e)}")
                response_text = self._generate_fallback_response(query, sources)
        else:
            response_text = self._generate_fallback_response(query, sources)
        
        # Extract citations and format response
        formatted_response, citations = self._format_response_with_citations(
            response_text, sources
        )
        
        # Convert to requested format
        formatted_response = self._convert_response_format(formatted_response, self.response_format)
        
        # Create response object
        response = GeneratedResponse(
            content=formatted_response,
            citations=citations,
            quality=ResponseQuality(),  # Will be assessed later if enabled
            format_type=self.response_format,
            metadata={
                "model_used": model_name,  # Use actual model used
                "temperature": temperature,  # Use actual temperature used
                "sources_count": len(sources)
            }
        )
        
        return response
    
    async def _generate_streaming_response(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        generation_config: Dict[str, Any]
    ) -> GeneratedResponse:
        """Generate a streaming response using LLM."""
        
        # Extract config with fallbacks to instance defaults
        model_name = generation_config.get("model", self.model_name)
        temperature = generation_config.get("temperature", self.temperature)
        max_tokens = generation_config.get("max_tokens", self.max_response_length)
        
        # Prepare context from sources
        context = self._prepare_context(sources)
        
        # Build prompt
        prompt = self._build_generation_prompt(
            query, context, conversation_history, generation_config
        )
        
        # Generate streaming response
        chunks = []
        full_response = ""
        
        if self.openai_service:
            try:
                async for chunk in self.openai_service.create_chat_completion_stream(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,  # Use config override
                    temperature=temperature,  # Use config override
                    max_tokens=max_tokens  # Use config override
                ):
                    chunks.append(chunk)
                    full_response += chunk
            except Exception as e:
                logger.error(f"OpenAI streaming failed: {str(e)}")
                full_response = self._generate_fallback_response(query, sources)
                chunks = [full_response]
        else:
            full_response = self._generate_fallback_response(query, sources)
            chunks = [full_response]
        
        # Extract citations and format response
        formatted_response, citations = self._format_response_with_citations(
            full_response, sources
        )
        
        # Convert to requested format
        formatted_response = self._convert_response_format(formatted_response, self.response_format)
        
        # Create response object
        response = GeneratedResponse(
            content=formatted_response,
            citations=citations,
            quality=ResponseQuality(),
            format_type=self.response_format,
            metadata={
                "model_used": model_name,  # Use actual model used
                "temperature": temperature,  # Use actual temperature used
                "sources_count": len(sources),
                "streaming": True
            },
            streaming_chunks=chunks
        )
        
        self.generation_stats['streaming_responses'] += 1
        
        return response
    
    def _prepare_context(self, sources: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved sources."""
        
        if not sources:
            return "No relevant sources found."
        
        context_parts = []
        
        for i, source in enumerate(sources[:self.max_citations], 1):
            source_content = source.get('content', '')
            source_title = source.get('document_title', f'Source {i}')
            source_url = source.get('url', '')
            relevance_score = source.get('relevance_score', {}).get('combined_score', 0.0)
            
            # Format source with metadata
            source_text = f"[Source {i}] {source_title}"
            if source_url:
                source_text += f" ({source_url})"
            source_text += f" [Relevance: {relevance_score:.2f}]\n{source_content}\n"
            
            context_parts.append(source_text)
        
        return "\n".join(context_parts)
    
    def _build_generation_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, Any]],
        generation_config: Dict[str, Any]
    ) -> str:
        """Build the prompt for response generation."""
        
        # Detect user's language and get appropriate instruction
        detected_language, confidence = detect_query_language(query)
        language_instruction = get_language_instruction_for_query(query)
        
        # Base system prompt with language instruction
        system_prompt = f"""You are an expert AI assistant that provides accurate, well-researched answers with proper source citations.

{language_instruction}

Your task is to:
1. Answer the user's question comprehensively using the provided context
2. Include proper citations in your response using [Source X] format
3. Ensure all claims are supported by the provided sources
4. Maintain a helpful, professional tone
5. Structure your response clearly with appropriate formatting

Guidelines:
- Use only information from the provided sources
- Cite sources immediately after making claims: "This is a fact [Source 1]."
- If sources conflict, acknowledge the disagreement
- If sources don't fully answer the question, state what information is available
- Keep responses focused and relevant to the question
- Respond in the same language as the user's question"""
        
        # Add conversation context if available
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages
            conversation_context = "\n\nConversation Context:\n"
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]  # Truncate long messages
                conversation_context += f"{role.title()}: {content}\n"
        
        # Build full prompt
        prompt = f"""{system_prompt}

{conversation_context}

Context Sources:
{context}

User Question: {query}

Please provide a comprehensive answer with proper citations:"""
        
        return prompt
    
    def _format_response_with_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format response with proper citations and extract citation metadata."""
        
        citations = []
        formatted_response = response_text
        
        # Extract and format citations based on style
        if self.citation_style == CitationStyle.NUMBERED:
            formatted_response, citations = self._format_numbered_citations(
                response_text, sources
            )
        elif self.citation_style == CitationStyle.BRACKETED:
            formatted_response, citations = self._format_bracketed_citations(
                response_text, sources
            )
        elif self.citation_style == CitationStyle.FOOTNOTE:
            formatted_response, citations = self._format_footnote_citations(
                response_text, sources
            )
        elif self.citation_style == CitationStyle.INLINE:
            formatted_response, citations = self._format_inline_citations(
                response_text, sources
            )
        
        return formatted_response, citations
    
    def _format_numbered_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format citations in numbered style [1], [2], etc."""
        
        citations = []
        citation_map = {}
        
        # Find all [Source X] patterns and replace with numbers
        pattern = r'\[Source (\d+)\]'
        matches = re.finditer(pattern, response_text)
        
        for match in matches:
            source_num = int(match.group(1)) - 1  # Convert to 0-based index
            
            if 0 <= source_num < len(sources):
                source = sources[source_num]
                
                if source_num not in citation_map:
                    citation_id = len(citations) + 1
                    citation_map[source_num] = citation_id
                    
                    citations.append({
                        'id': citation_id,
                        'source_id': source.get('source_id', ''),
                        'title': source.get('document_title', f'Source {citation_id}'),
                        'url': source.get('url', ''),
                        'relevance_score': source.get('relevance_score', {}).get('combined_score', 0.0),
                        'content_snippet': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', '')
                    })
        
        # Replace [Source X] with [citation_id]
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if source_num in citation_map:
                return f"[{citation_map[source_num]}]"
            return match.group(0)
        
        formatted_response = re.sub(pattern, replace_citation, response_text)
        
        # Add citations list at the end
        if citations:
            formatted_response += "\n\n**Sources:**\n"
            for citation in citations:
                formatted_response += f"[{citation['id']}] {citation['title']}"
                if citation['url']:
                    formatted_response += f" - {citation['url']}"
                formatted_response += "\n"
        
        return formatted_response, citations
    
    def _format_bracketed_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format citations in bracketed style [Title, Year]."""
        
        citations = []
        
        # Simple implementation - convert [Source X] to [Title]
        pattern = r'\[Source (\d+)\]'
        
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if 0 <= source_num < len(sources):
                source = sources[source_num]
                title = source.get('document_title', f'Source {source_num + 1}')
                
                # Add to citations if not already present
                if not any(c['title'] == title for c in citations):
                    citations.append({
                        'id': len(citations) + 1,
                        'source_id': source.get('source_id', ''),
                        'title': title,
                        'url': source.get('url', ''),
                        'relevance_score': source.get('relevance_score', {}).get('combined_score', 0.0),
                        'content_snippet': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', '')
                    })
                
                return f"[{title}]"
            return match.group(0)
        
        formatted_response = re.sub(pattern, replace_citation, response_text)
        
        return formatted_response, citations
    
    def _format_footnote_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format citations in footnote style with superscript numbers."""
        
        citations = []
        citation_map = {}
        
        pattern = r'\[Source (\d+)\]'
        matches = re.finditer(pattern, response_text)
        
        for match in matches:
            source_num = int(match.group(1)) - 1
            
            if 0 <= source_num < len(sources) and source_num not in citation_map:
                source = sources[source_num]
                citation_id = len(citations) + 1
                citation_map[source_num] = citation_id
                
                citations.append({
                    'id': citation_id,
                    'source_id': source.get('source_id', ''),
                    'title': source.get('document_title', f'Source {citation_id}'),
                    'url': source.get('url', ''),
                    'relevance_score': source.get('relevance_score', {}).get('combined_score', 0.0),
                    'content_snippet': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', '')
                })
        
        # Replace with superscript numbers
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if source_num in citation_map:
                return f"^{citation_map[source_num]}"
            return match.group(0)
        
        formatted_response = re.sub(pattern, replace_citation, response_text)
        
        # Add footnotes at the end
        if citations:
            formatted_response += "\n\n**References:**\n"
            for citation in citations:
                formatted_response += f"^{citation['id']} {citation['title']}"
                if citation['url']:
                    formatted_response += f" - {citation['url']}"
                formatted_response += "\n"
        
        return formatted_response, citations
    
    def _format_inline_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format citations inline with full source information."""
        
        citations = []
        
        pattern = r'\[Source (\d+)\]'
        
        def replace_citation(match):
            source_num = int(match.group(1)) - 1
            if 0 <= source_num < len(sources):
                source = sources[source_num]
                title = source.get('document_title', f'Source {source_num + 1}')
                url = source.get('url', '')
                
                # Add to citations
                if not any(c['title'] == title for c in citations):
                    citations.append({
                        'id': len(citations) + 1,
                        'source_id': source.get('source_id', ''),
                        'title': title,
                        'url': url,
                        'relevance_score': source.get('relevance_score', {}).get('combined_score', 0.0),
                        'content_snippet': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', '')
                    })
                
                # Return inline citation
                if url:
                    return f"([{title}]({url}))"
                else:
                    return f"({title})"
            return match.group(0)
        
        formatted_response = re.sub(pattern, replace_citation, response_text)
        
        return formatted_response, citations
    
    def _convert_response_format(self, content: str, format_type: ResponseFormat) -> str:
        """Convert response content to the specified format."""
        
        try:
            if format_type == ResponseFormat.MARKDOWN:
                # Content is already in markdown format
                return content
            
            elif format_type == ResponseFormat.PLAIN_TEXT:
                # Remove markdown formatting
                import re
                # Remove headers
                content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
                # Remove bold/italic
                content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
                content = re.sub(r'\*(.*?)\*', r'\1', content)
                # Remove links but keep text
                content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
                # Remove citations formatting
                content = re.sub(r'\[(\d+)\]', r'[\1]', content)
                return content.strip()
            
            elif format_type == ResponseFormat.HTML:
                # Convert markdown to HTML
                import re
                html_content = content
                # Convert headers
                html_content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
                html_content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
                # Convert bold/italic
                html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
                html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
                # Convert links
                html_content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html_content)
                # Convert line breaks
                html_content = html_content.replace('\n\n', '</p><p>')
                html_content = html_content.replace('\n', '<br>')
                # Wrap in paragraphs
                html_content = f'<p>{html_content}</p>'
                return html_content
            
            elif format_type == ResponseFormat.JSON:
                # Convert to structured JSON
                import re
                
                # Extract sections
                sections = []
                current_section = {"type": "paragraph", "content": ""}
                
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_section["content"]:
                            sections.append(current_section)
                            current_section = {"type": "paragraph", "content": ""}
                        continue
                    
                    # Check for headers
                    if line.startswith('###'):
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"type": "header", "level": 3, "content": line[3:].strip()}
                        sections.append(current_section)
                        current_section = {"type": "paragraph", "content": ""}
                    elif line.startswith('##'):
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"type": "header", "level": 2, "content": line[2:].strip()}
                        sections.append(current_section)
                        current_section = {"type": "paragraph", "content": ""}
                    elif line.startswith('#'):
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {"type": "header", "level": 1, "content": line[1:].strip()}
                        sections.append(current_section)
                        current_section = {"type": "paragraph", "content": ""}
                    else:
                        if current_section["content"]:
                            current_section["content"] += " " + line
                        else:
                            current_section["content"] = line
                
                if current_section["content"]:
                    sections.append(current_section)
                
                return json.dumps({
                    "format": "structured_response",
                    "sections": sections,
                    "raw_content": content
                }, indent=2)
            
            else:
                # Default to original content
                return content
        
        except Exception as e:
            logger.warning(f"Format conversion failed for {format_type.value}: {str(e)}")
            # Return original content if conversion fails
            return content
    
    def _generate_fallback_response(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Generate a fallback response when OpenAI is unavailable."""
        
        if not sources:
            return f"I apologize, but I don't have enough information to answer your question about '{query}'. No relevant sources were found."
        
        # Create a simple response based on sources
        response = f"Based on the available sources, here's what I found regarding '{query}':\n\n"
        
        for i, source in enumerate(sources[:3], 1):  # Use top 3 sources
            content = source.get('content', '')[:300]  # First 300 chars
            title = source.get('document_title', f'Source {i}')
            
            response += f"According to {title} [Source {i}]: {content}...\n\n"
        
        response += "Please note: This response was generated using a fallback method due to service limitations."
        
        return response
    
    async def _assess_response_quality(
        self,
        query: str,
        response: GeneratedResponse,
        sources: List[Dict[str, Any]]
    ) -> ResponseQuality:
        """Assess the quality of the generated response."""
        
        # Basic quality assessment (can be enhanced with ML models)
        relevance_score = self._assess_relevance(query, response.content)
        coherence_score = self._assess_coherence(response.content)
        completeness_score = self._assess_completeness(query, response.content)
        citation_accuracy = self._assess_citation_accuracy(response.content, sources)
        factual_accuracy = self._assess_factual_accuracy(response.content, sources)
        
        return ResponseQuality(
            relevance_score=relevance_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            citation_accuracy=citation_accuracy,
            factual_accuracy=factual_accuracy
        )
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess how relevant the response is to the query."""
        
        # Simple keyword overlap assessment
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = query_words.intersection(response_words)
        return len(overlap) / len(query_words)
    
    def _assess_coherence(self, response: str) -> float:
        """Assess the coherence and readability of the response."""
        
        # Simple coherence metrics
        sentences = response.split('.')
        
        if len(sentences) < 2:
            return 0.5
        
        # Check for reasonable sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            coherence = 0.8
        elif 5 <= avg_sentence_length <= 35:
            coherence = 0.6
        else:
            coherence = 0.4
        
        return min(1.0, coherence)
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess how complete the response is."""
        
        # Check response length relative to query complexity
        query_complexity = len(query.split())
        response_length = len(response.split())
        
        # Expect roughly 10-50 words per query word for complete answers
        expected_min = query_complexity * 5
        expected_max = query_complexity * 30
        
        if expected_min <= response_length <= expected_max:
            return 0.9
        elif response_length >= expected_min * 0.7:
            return 0.7
        else:
            return 0.5
    
    def _assess_citation_accuracy(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Assess the accuracy of citations in the response."""
        
        # Count citation patterns
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\[Source \d+\]',  # [Source 1], etc.
            r'\([^)]+\)',  # (Title), etc.
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            total_citations += len(re.findall(pattern, response))
        
        if total_citations == 0:
            return 0.0 if sources else 1.0  # No citations needed if no sources
        
        # Simple heuristic: good citation rate is 1-3 citations per 100 words
        words = len(response.split())
        expected_citations = max(1, words // 100)
        
        if total_citations >= expected_citations:
            return min(1.0, total_citations / (expected_citations * 2))
        else:
            return total_citations / expected_citations
    
    def _assess_factual_accuracy(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Assess factual accuracy based on source alignment."""
        
        # Simple heuristic: check if response content aligns with sources
        if not sources:
            return 0.5  # Neutral score if no sources to verify against
        
        response_words = set(response.lower().split())
        source_words = set()
        
        for source in sources:
            content = source.get('content', '')
            source_words.update(content.lower().split())
        
        if not source_words:
            return 0.5
        
        # Calculate overlap
        overlap = response_words.intersection(source_words)
        return min(1.0, len(overlap) / len(response_words) * 2)  # Scale up the score
    
    def _generate_cache_key(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> str:
        """Generate cache key for response caching."""
        
        # Create a hash based on query, source IDs, and config
        source_ids = [s.get('source_id', '') for s in sources]
        key_data = f"{query}:{':'.join(source_ids)}:{str(sorted(config.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response result if still valid."""
        
        if cache_key not in self.response_cache:
            return None
        
        cached_data = self.response_cache[cache_key]
        cache_time = cached_data.get('cached_at', 0)
        
        if datetime.utcnow().timestamp() - cache_time > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        result = cached_data['result'].copy()
        result['generation_metadata']['cache_hit'] = True
        return result
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache response result."""
        
        self.response_cache[cache_key] = {
            'result': result.copy(),
            'cached_at': datetime.utcnow().timestamp()
        }
        
        # Clean old cache entries
        current_time = datetime.utcnow().timestamp()
        expired_keys = [
            key for key, data in self.response_cache.items()
            if current_time - data['cached_at'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
    
    def _update_stats(self, response: GeneratedResponse) -> None:
        """Update generation statistics."""
        
        self.generation_stats['total_generations'] += 1
        
        # Update average response length
        total = self.generation_stats['total_generations']
        current_avg = self.generation_stats['avg_response_length']
        self.generation_stats['avg_response_length'] = (
            (current_avg * (total - 1) + response.word_count) / total
        )
        
        # Update average quality score
        if response.quality:
            current_quality_avg = self.generation_stats['avg_quality_score']
            self.generation_stats['avg_quality_score'] = (
                (current_quality_avg * (total - 1) + response.quality.overall_quality) / total
            )
            
            # Update citation accuracy
            current_citation_avg = self.generation_stats['citation_accuracy']
            self.generation_stats['citation_accuracy'] = (
                (current_citation_avg * (total - 1) + response.quality.citation_accuracy) / total
            )
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        
        return {
            'total_generations': self.generation_stats['total_generations'],
            'cache_hit_rate': (
                self.generation_stats['cache_hits'] / max(1, self.generation_stats['total_generations'])
            ),
            'streaming_rate': (
                self.generation_stats['streaming_responses'] / max(1, self.generation_stats['total_generations'])
            ),
            'avg_response_length': self.generation_stats['avg_response_length'],
            'avg_quality_score': self.generation_stats['avg_quality_score'],
            'citation_accuracy': self.generation_stats['citation_accuracy']
        }
    
    async def stream_response(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None,
        generation_config: Dict[str, Any] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response generation in real-time."""
        
        if not self.enable_streaming:
            raise ValueError("Streaming is not enabled for this agent")
        
        conversation_history = conversation_history or []
        generation_config = generation_config or {"stream": True}
        
        # Prepare context and prompt
        context = self._prepare_context(sources)
        prompt = self._build_generation_prompt(
            query, context, conversation_history, generation_config
        )
        
        # Stream response
        if self.openai_service:
            try:
                async for chunk in self.openai_service.create_chat_completion_stream(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_response_length
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming failed: {str(e)}")
                yield self._generate_fallback_response(query, sources)
        else:
            yield self._generate_fallback_response(query, sources) 