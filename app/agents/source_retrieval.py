"""
Source Retrieval Agent for context retrieval from knowledge base.

This agent handles:
- Semantic search using Supabase pgvector
- Dynamic source selection logic
- Relevance scoring and ranking system
- Multi-source retrieval (database, APIs, web)
- Adaptive retrieval strategies
- Result deduplication and filtering
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib

from .base import BaseAgent

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enumeration for retrieval strategies."""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    ADAPTIVE = "adaptive"


class SourceType(Enum):
    """Enumeration for source types."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    WEB = "web"
    API = "api"
    CACHED = "cached"


class RelevanceScore:
    """Class for managing relevance scores."""
    
    def __init__(
        self,
        semantic_score: float = 0.0,
        keyword_score: float = 0.0,
        recency_score: float = 0.0,
        authority_score: float = 0.0,
        context_score: float = 0.0
    ):
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.recency_score = recency_score
        self.authority_score = authority_score
        self.context_score = context_score
    
    @property
    def combined_score(self) -> float:
        """Calculate weighted combined relevance score."""
        weights = {
            'semantic': 0.4,
            'keyword': 0.2,
            'recency': 0.1,
            'authority': 0.2,
            'context': 0.1
        }
        
        return (
            self.semantic_score * weights['semantic'] +
            self.keyword_score * weights['keyword'] +
            self.recency_score * weights['recency'] +
            self.authority_score * weights['authority'] +
            self.context_score * weights['context']
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'recency_score': self.recency_score,
            'authority_score': self.authority_score,
            'context_score': self.context_score,
            'combined_score': self.combined_score
        }


class RetrievedSource:
    """Class representing a retrieved source."""
    
    def __init__(
        self,
        source_id: str,
        content: str,
        source_type: SourceType,
        relevance_score: RelevanceScore,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: Optional[int] = None,
        document_title: Optional[str] = None,
        url: Optional[str] = None
    ):
        self.source_id = source_id
        self.content = content
        self.source_type = source_type
        self.relevance_score = relevance_score
        self.metadata = metadata or {}
        self.chunk_index = chunk_index
        self.document_title = document_title
        self.url = url
        self.retrieved_at = datetime.utcnow()
    
    @property
    def content_hash(self) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'source_id': self.source_id,
            'content': self.content,
            'source_type': self.source_type.value,
            'relevance_score': self.relevance_score.to_dict(),
            'metadata': self.metadata,
            'chunk_index': self.chunk_index,
            'document_title': self.document_title,
            'url': self.url,
            'retrieved_at': self.retrieved_at.isoformat(),
            'content_hash': self.content_hash
        }


class SourceRetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant sources from knowledge base.
    
    Capabilities:
    - Semantic search using vector embeddings
    - Hybrid search combining semantic and keyword matching
    - Dynamic source selection and ranking
    - Multi-source retrieval from different data sources
    - Adaptive retrieval strategies based on query type
    - Result deduplication and filtering
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "source_retrieval",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Source Retrieval Agent."""
        super().__init__(agent_id, agent_type, config)
        
        # Configuration
        self.max_results = config.get("max_results", 5) if config else 3
        self.min_relevance_threshold = config.get("min_relevance_threshold", 0.1) if config else 0.1
        self.semantic_weight = config.get("semantic_weight", 0.7) if config else 0.7
        self.keyword_weight = config.get("keyword_weight", 0.3) if config else 0.3
        self.enable_deduplication = config.get("enable_deduplication", True) if config else True
        self.similarity_threshold = config.get("similarity_threshold", 0.85) if config else 0.85
        self.default_strategy = RetrievalStrategy(config.get("default_strategy", "adaptive")) if config else RetrievalStrategy.ADAPTIVE
        self.enable_multi_source = config.get("enable_multi_source", True) if config else True
        
        # Services (will be initialized on start)
        self.openai_service = None
        self.vector_search_service = None
        self.supabase_client = None
        
        # Cache for recent retrievals
        self.retrieval_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'semantic_searches': 0,
            'keyword_searches': 0,
            'hybrid_searches': 0,
            'avg_retrieval_time': 0.0,
            'avg_results_count': 0.0
        }
        
        logger.info(f"Source Retrieval Agent {self.agent_id} initialized")
    
    async def _on_start(self) -> None:
        """Initialize services when agent starts."""
        try:
            # Import here to avoid circular imports
            from ..core.dependencies import (
                get_openai_service,
                get_vector_search_service,
                get_supabase_client
            )
            
            self.openai_service = get_openai_service()
            self.vector_search_service = get_vector_search_service()
            self.supabase_client = get_supabase_client()
            
            logger.info(f"Source Retrieval Agent {self.agent_id} connected to services")
        except Exception as e:
            logger.warning(f"Could not connect to all services: {str(e)}")
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant sources for a query.
        
        Args:
            input_data: Contains the query, context decision, and optional parameters
            
        Returns:
            Dictionary with retrieved sources and metadata
        """
        query = input_data.get("query", "").strip()
        context_decision = input_data.get("context_decision", {})
        conversation_history = input_data.get("conversation_history", [])
        retrieval_config = input_data.get("retrieval_config", {})
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, retrieval_config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.retrieval_stats['cache_hits'] += 1
            return cached_result
        
        # Determine retrieval strategy
        strategy = self._determine_strategy(query, context_decision, retrieval_config)
        
        # Perform retrieval based on strategy
        logger.info(f"Retrieving sources for query: {query} with strategy: {strategy} and retrieval config: {retrieval_config}")
        retrieved_sources = await self._retrieve_sources(
            query, strategy, conversation_history, retrieval_config
        )
        logger.info(f"Retrieved sources: {retrieved_sources}")
        
        # Apply post-processing
        processed_sources = await self._post_process_sources(
            retrieved_sources, query, context_decision
        )
        
        # Prepare result
        result = {
            "query": query,
            "strategy_used": strategy.value,
            "sources": [source.to_dict() for source in processed_sources],
            "total_sources": len(processed_sources),
            "retrieval_metadata": {
                "processing_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "strategy": strategy.value,
                "cache_hit": False,
                "performance_stats": self._get_performance_stats()
            }
        }
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        # Update statistics
        self._update_stats(strategy, len(processed_sources))
        
        logger.debug(
            f"Retrieved {len(processed_sources)} sources using {strategy.value} "
            f"for query: '{query[:50]}...'"
        )
        
        return result
    
    def _determine_strategy(
        self,
        query: str,
        context_decision: Dict[str, Any],
        retrieval_config: Dict[str, Any]
    ) -> RetrievalStrategy:
        """Determine the best retrieval strategy for the query."""
        
        # Check for explicit strategy override
        if "strategy" in retrieval_config:
            try:
                return RetrievalStrategy(retrieval_config["strategy"])
            except ValueError:
                logger.warning(f"Invalid strategy: {retrieval_config['strategy']}")
        
        # Use adaptive strategy selection
        if self.default_strategy == RetrievalStrategy.ADAPTIVE:
            return self._adaptive_strategy_selection(query, context_decision)
        
        return self.default_strategy
    
    def _adaptive_strategy_selection(
        self,
        query: str,
        context_decision: Dict[str, Any]
    ) -> RetrievalStrategy:
        """Adaptively select retrieval strategy based on query characteristics."""
        
        query_lower = query.lower()
        
        # Check for specific keywords that suggest keyword search
        keyword_indicators = [
            'define', 'definition', 'what is', 'who is', 'when did',
            'where is', 'how many', 'list', 'name', 'identify'
        ]
        
        if any(indicator in query_lower for indicator in keyword_indicators):
            return RetrievalStrategy.KEYWORD
        
        # Check for complex queries that benefit from hybrid search
        complex_indicators = [
            'compare', 'difference', 'similar', 'contrast', 'relationship',
            'explain', 'analyze', 'evaluate', 'discuss'
        ]
        
        if any(indicator in query_lower for indicator in complex_indicators):
            return RetrievalStrategy.HYBRID
        
        # Check context decision confidence
        decision_confidence = context_decision.get("confidence", 0.5)
        if decision_confidence > 0.8:
            return RetrievalStrategy.SEMANTIC_ONLY
        
        # Default to hybrid for balanced results
        return RetrievalStrategy.HYBRID
    
    async def _retrieve_sources(
        self,
        query: str,
        strategy: RetrievalStrategy,
        conversation_history: List[Dict[str, Any]],
        retrieval_config: Dict[str, Any]
    ) -> List[RetrievedSource]:
        """Retrieve sources based on the selected strategy."""
        
        sources = []
        
        if strategy == RetrievalStrategy.SEMANTIC_ONLY:
            sources = await self._semantic_search(query, retrieval_config)
            self.retrieval_stats['semantic_searches'] += 1
            
        elif strategy == RetrievalStrategy.KEYWORD:
            sources = await self._keyword_search(query, retrieval_config)
            self.retrieval_stats['keyword_searches'] += 1
            
        elif strategy == RetrievalStrategy.HYBRID:
            semantic_sources = await self._semantic_search(query, retrieval_config)
            keyword_sources = await self._keyword_search(query, retrieval_config)
            sources = self._merge_search_results(semantic_sources, keyword_sources)
            self.retrieval_stats['hybrid_searches'] += 1
            
        elif strategy == RetrievalStrategy.ADAPTIVE:
            # Adaptive strategy combines multiple approaches
            sources = await self._adaptive_retrieval(query, conversation_history, retrieval_config)
        
        return sources
    
    async def _semantic_search(
        self,
        query: str,
        retrieval_config: Dict[str, Any]
    ) -> List[RetrievedSource]:
        """Perform semantic search using vector embeddings."""
        
        if not self.vector_search_service:
            logger.warning("Vector search service not available")
            return []
        
        try:
            # Perform semantic search using the correct method
            search_results, metrics = await self.vector_search_service.semantic_search(
                query=query,
                config=None,  # Use default config for now
                document_ids=retrieval_config.get("document_ids"),
                user_id=retrieval_config.get("user_id")
            )
            
            sources = []
            for result in search_results:
                relevance_score = RelevanceScore(
                    semantic_score=result.similarity,
                    recency_score=self._calculate_recency_score(result.metadata.get('created_at')),
                    authority_score=self._calculate_authority_score(result.metadata)
                )
                
                source = RetrievedSource(
                    source_id=result.chunk_id,
                    content=result.chunk_text,
                    source_type=SourceType.CHUNK,
                    relevance_score=relevance_score,
                    metadata=result.metadata,
                    chunk_index=result.chunk_index,
                    document_title=result.filename,
                    url=result.metadata.get('url')
                )
                sources.append(source)
            
            logger.info(f"Semantic search found {len(sources)} sources for query: '{query[:50]}...'")
            return sources
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        retrieval_config: Dict[str, Any]
    ) -> List[RetrievedSource]:
        """Perform keyword-based search."""
        
        if not self.supabase_client:
            logger.warning("Supabase client not available")
            return []
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Build search query
            search_query = " | ".join(keywords)  # OR search
            
            # Use the stored procedure for keyword search
            # Note: Using the actual function signature in the database
            response = self.supabase_client.rpc(
                'search_keywords',
                {
                    'search_query': search_query,
                    'match_count': retrieval_config.get('max_results', self.max_results),
                    'document_ids': retrieval_config.get('document_ids'),
                    'filter_user_id': retrieval_config.get('user_id')
                }
            ).execute()
            
            sources = []
            for result in response.data:
                # Use the rank score from the stored procedure
                keyword_score = float(result.get('rank', 0.0))
                
                relevance_score = RelevanceScore(
                    keyword_score=keyword_score,
                    recency_score=self._calculate_recency_score(result.get('created_at')),
                    authority_score=self._calculate_authority_score(result.get('metadata', {}))
                )
                
                source = RetrievedSource(
                    source_id=result['chunk_id'],
                    content=result['chunk_text'],
                    source_type=SourceType.CHUNK,
                    relevance_score=relevance_score,
                    metadata=result.get('metadata', {}),
                    chunk_index=result.get('chunk_index'),
                    document_title=result.get('filename'),
                    url=result.get('metadata', {}).get('url')
                )
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _merge_search_results(
        self,
        semantic_sources: List[RetrievedSource],
        keyword_sources: List[RetrievedSource]
    ) -> List[RetrievedSource]:
        """Merge and rank results from semantic and keyword searches."""
        
        # Create a map to track sources by ID
        source_map = {}
        
        # Add semantic sources
        for source in semantic_sources:
            source_map[source.source_id] = source
        
        # Merge keyword sources, combining scores for duplicates
        for source in keyword_sources:
            if source.source_id in source_map:
                # Combine scores for duplicate sources
                existing = source_map[source.source_id]
                existing.relevance_score.keyword_score = max(
                    existing.relevance_score.keyword_score,
                    source.relevance_score.keyword_score
                )
            else:
                source_map[source.source_id] = source
        
        # Convert back to list and sort by combined score
        merged_sources = list(source_map.values())
        merged_sources.sort(key=lambda x: x.relevance_score.combined_score, reverse=True)
        
        return merged_sources[:self.max_results]
    
    async def _adaptive_retrieval(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        retrieval_config: Dict[str, Any]
    ) -> List[RetrievedSource]:
        """Perform adaptive retrieval using multiple strategies."""
        
        # Start with hybrid search as base
        sources = await self._retrieve_sources(
            query, RetrievalStrategy.HYBRID, conversation_history, retrieval_config
        )
        
        # If we don't have enough high-quality results, try expanding
        if len(sources) < self.max_results // 2:
            # Try expanding query with conversation context
            expanded_query = self._expand_query_with_context(query, conversation_history)
            if expanded_query != query:
                additional_sources = await self._semantic_search(expanded_query, retrieval_config)
                sources.extend(additional_sources)
        
        # Remove duplicates and re-rank
        sources = self._deduplicate_sources(sources)
        sources.sort(key=lambda x: x.relevance_score.combined_score, reverse=True)
        
        return sources[:self.max_results]
    
    async def _post_process_sources(
        self,
        sources: List[RetrievedSource],
        query: str,
        context_decision: Dict[str, Any]
    ) -> List[RetrievedSource]:
        """Apply post-processing to retrieved sources."""
        
        # Filter by minimum relevance threshold
        filtered_sources = [
            source for source in sources
            if source.relevance_score.combined_score >= self.min_relevance_threshold
        ]
        
        # Apply deduplication if enabled
        if self.enable_deduplication:
            filtered_sources = self._deduplicate_sources(filtered_sources)
        
        # Enhance relevance scores with context
        for source in filtered_sources:
            source.relevance_score.context_score = self._calculate_context_score(
                source, query, context_decision
            )
        
        # Re-sort by updated combined scores
        filtered_sources.sort(key=lambda x: x.relevance_score.combined_score, reverse=True)
        
        # Apply final result limit
        return filtered_sources[:self.max_results]
    
    def _deduplicate_sources(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Remove duplicate sources based on content similarity."""
        
        if not self.enable_deduplication:
            return sources
        
        unique_sources = []
        seen_hashes = set()
        
        for source in sources:
            content_hash = source.content_hash
            
            # Check for exact duplicates
            if content_hash in seen_hashes:
                continue
            
            # Check for near-duplicates
            is_duplicate = False
            for existing in unique_sources:
                similarity = self._calculate_content_similarity(source.content, existing.content)
                if similarity > self.similarity_threshold:
                    # Keep the one with higher relevance score
                    if source.relevance_score.combined_score > existing.relevance_score.combined_score:
                        unique_sources.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_sources.append(source)
                seen_hashes.add(content_hash)
        
        return unique_sources
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'how', 'what', 'when', 'where', 'why', 'who', 'which'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """Calculate keyword relevance score."""
        
        query_keywords = set(self._extract_keywords(query))
        content_keywords = set(self._extract_keywords(content))
        
        if not query_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_keywords.intersection(content_keywords)
        union = query_keywords.union(content_keywords)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_recency_score(self, created_at: Optional[str]) -> float:
        """Calculate recency score based on creation date."""
        
        if not created_at:
            return 0.5  # Neutral score for unknown dates
        
        try:
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.utcnow().replace(tzinfo=created_date.tzinfo)
            
            # Calculate days since creation
            days_old = (now - created_date).days
            
            # Exponential decay with 30-day half-life
            return max(0.1, 2 ** (-days_old / 30))
            
        except Exception:
            return 0.5
    
    def _calculate_authority_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate authority score based on source metadata."""
        
        score = 0.5  # Base score
        
        # Boost for official sources
        if metadata.get('source_type') == 'official':
            score += 0.3
        
        # Boost for verified sources
        if metadata.get('verified', False):
            score += 0.2
        
        # Boost based on citation count
        citations = metadata.get('citations', 0)
        if citations > 0:
            score += min(0.2, citations / 100)
        
        return min(1.0, score)
    
    def _calculate_context_score(
        self,
        source: RetrievedSource,
        query: str,
        context_decision: Dict[str, Any]
    ) -> float:
        """Calculate context relevance score."""
        
        score = 0.5  # Base score
        
        # Boost if context decision indicates high necessity
        decision = context_decision.get("decision", "optional")
        if decision == "required":
            score += 0.3
        elif decision == "optional":
            score += 0.1
        
        # Boost based on decision confidence
        confidence = context_decision.get("confidence", 0.5)
        score += confidence * 0.2
        
        return min(1.0, score)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _expand_query_with_context(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Expand query with relevant context from conversation history."""
        
        if not conversation_history:
            return query
        
        # Get recent relevant messages
        recent_messages = conversation_history[-3:]
        context_terms = []
        
        for message in recent_messages:
            content = message.get('content', '')
            if content and len(content) > 10:
                # Extract key terms
                terms = self._extract_keywords(content)
                context_terms.extend(terms[:3])  # Top 3 terms per message
        
        # Add unique context terms to query
        query_terms = set(self._extract_keywords(query))
        new_terms = [term for term in context_terms if term not in query_terms]
        
        if new_terms:
            expanded_query = f"{query} {' '.join(new_terms[:3])}"
            return expanded_query
        
        return query
    
    def _generate_cache_key(self, query: str, config: Dict[str, Any]) -> str:
        """Generate cache key for retrieval results."""
        
        key_data = f"{query}:{str(sorted(config.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached retrieval result if still valid."""
        
        if cache_key not in self.retrieval_cache:
            return None
        
        cached_data = self.retrieval_cache[cache_key]
        cache_time = cached_data.get('cached_at', 0)
        
        if datetime.utcnow().timestamp() - cache_time > self.cache_ttl:
            del self.retrieval_cache[cache_key]
            return None
        
        result = cached_data['result'].copy()
        result['retrieval_metadata']['cache_hit'] = True
        return result
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache retrieval result."""
        
        self.retrieval_cache[cache_key] = {
            'result': result.copy(),
            'cached_at': datetime.utcnow().timestamp()
        }
        
        # Clean old cache entries
        current_time = datetime.utcnow().timestamp()
        expired_keys = [
            key for key, data in self.retrieval_cache.items()
            if current_time - data['cached_at'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.retrieval_cache[key]
    
    def _update_stats(self, strategy: RetrievalStrategy, result_count: int) -> None:
        """Update retrieval statistics."""
        
        self.retrieval_stats['total_retrievals'] += 1
        
        # Update strategy distribution
        if strategy == RetrievalStrategy.SEMANTIC_ONLY:
            self.retrieval_stats['semantic_searches'] += 1
        elif strategy == RetrievalStrategy.KEYWORD:
            self.retrieval_stats['keyword_searches'] += 1
        elif strategy == RetrievalStrategy.HYBRID:
            self.retrieval_stats['hybrid_searches'] += 1
        
        # Update average result count
        total = self.retrieval_stats['total_retrievals']
        current_avg = self.retrieval_stats['avg_results_count']
        self.retrieval_stats['avg_results_count'] = (
            (current_avg * (total - 1) + result_count) / total
        )
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        
        return {
            'total_retrievals': self.retrieval_stats['total_retrievals'],
            'cache_hit_rate': (
                self.retrieval_stats['cache_hits'] / max(1, self.retrieval_stats['total_retrievals'])
            ),
            'avg_results_count': self.retrieval_stats['avg_results_count'],
            'strategy_distribution': {
                'semantic_searches': self.retrieval_stats['semantic_searches'],
                'keyword_searches': self.retrieval_stats['keyword_searches'],
                'hybrid_searches': self.retrieval_stats['hybrid_searches']
            }
        } 