"""
Vector search service for semantic and hybrid search using Supabase pgvector.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from supabase import Client
import numpy as np

from ..core.config import settings
from ..models.document_models import SearchResult, SearchRequest
from ..services.openai_service import OpenAIService

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search supported by the vector search service."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    similarity_threshold: float = 0.5  # Lowered from 0.7 for better recall
    max_results: int = 5
    boost_recent: bool = True
    boost_factor: float = 1.2
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7


@dataclass
class SearchMetrics:
    """Metrics for search performance tracking."""
    query_time: float
    embedding_time: float
    search_time: float
    total_results: int
    filtered_results: int
    avg_similarity: float


class VectorSearchService:
    """
    Advanced vector search service with semantic and hybrid search capabilities.
    """
    
    def __init__(self, supabase_client: Client, openai_service: OpenAIService):
        self.supabase = supabase_client
        self.openai_service = openai_service
        self.search_stats = {
            "total_searches": 0,
            "avg_query_time": 0.0,
            "cache_hits": 0,
            "popular_queries": {}
        }
    
    async def semantic_search(
        self,
        query: str,
        config: Optional[SearchConfig] = None,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Perform semantic vector search using embeddings.
        
        Args:
            query: Search query text
            config: Search configuration parameters
            document_ids: Optional list of document IDs to search within
            user_id: Optional user ID for personalization
            
        Returns:
            Tuple of search results and metrics
        """
        start_time = time.time()
        config = config or SearchConfig()
        
        try:
            # Generate embedding for query
            embedding_start = time.time()
            query_embedding = await self.openai_service.create_embedding(query)
            embedding_time = time.time() - embedding_start
            
            # Perform vector search
            search_start = time.time()
            results, metrics = await self._execute_vector_search(
                query_embedding=query_embedding,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
            search_time = time.time() - search_start
            
            # Apply post-processing and ranking
            processed_results = await self._post_process_results(
                results=results,
                query=query,
                config=config
            )
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=total_time,
                embedding_time=embedding_time,
                search_time=search_time,
                total_results=len(results),
                filtered_results=len(processed_results),
                avg_similarity=np.mean([r.similarity for r in processed_results]) if processed_results else 0.0
            )
            
            # Update search statistics
            await self._update_search_stats(query, total_time)
            
            logger.info(
                f"Semantic search completed - query_length: {len(query)}, "
                f"results_count: {len(processed_results)}, query_time: {total_time}, "
                f"avg_similarity: {metrics.avg_similarity}"
            )
            
            return processed_results, metrics
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise e
    
    async def keyword_search(
        self,
        query: str,
        config: Optional[SearchConfig] = None,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Perform keyword-based text search.
        
        Args:
            query: Search query text
            config: Search configuration parameters
            document_ids: Optional list of document IDs to search within
            user_id: Optional user ID for filtering
            
        Returns:
            Tuple of search results and metrics
        """
        start_time = time.time()
        config = config or SearchConfig()
        
        try:
            # Prepare search terms
            search_terms = await self._prepare_search_terms(query)
            
            # Execute keyword search
            search_start = time.time()
            results = await self._execute_keyword_search(
                search_terms=search_terms,
                max_results=config.max_results,
                document_ids=document_ids,
                user_id=user_id
            )
            search_time = time.time() - search_start
            
            # Calculate relevance scores
            scored_results = await self._calculate_keyword_scores(
                results=results,
                query=query,
                search_terms=search_terms
            )
            
            # Calculate metrics
            total_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=total_time,
                embedding_time=0.0,  # No embedding for keyword search
                search_time=search_time,
                total_results=len(results),
                filtered_results=len(scored_results),
                avg_similarity=np.mean([r.similarity for r in scored_results]) if scored_results else 0.0
            )
            
            logger.info(
                f"Keyword search completed - query_length: {len(query)}, "
                f"results_count: {len(scored_results)}, query_time: {total_time}"
            )
            
            return scored_results, metrics
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            raise e
    
    async def hybrid_search(
        self,
        query: str,
        config: Optional[SearchConfig] = None,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query text
            config: Search configuration parameters
            document_ids: Optional list of document IDs to search within
            user_id: Optional user ID for personalization
            
        Returns:
            Tuple of search results and metrics
        """
        start_time = time.time()
        config = config or SearchConfig()
        
        try:
            # Perform both semantic and keyword searches
            semantic_results, semantic_metrics = await self.semantic_search(
                query=query,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
            
            keyword_results, keyword_metrics = await self.keyword_search(
                query=query,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
            
            # Combine and rank results
            combined_results = await self._combine_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=config.semantic_weight,
                keyword_weight=config.keyword_weight
            )
            
            # Limit to max results
            final_results = combined_results[:config.max_results]
            
            # Calculate combined metrics
            total_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=total_time,
                embedding_time=semantic_metrics.embedding_time,
                search_time=semantic_metrics.search_time + keyword_metrics.search_time,
                total_results=len(semantic_results) + len(keyword_results),
                filtered_results=len(final_results),
                avg_similarity=np.mean([r.similarity for r in final_results]) if final_results else 0.0
            )
            
            logger.info(
                f"Hybrid search completed - query_length: {len(query)}, "
                f"semantic_results: {len(semantic_results)}, keyword_results: {len(keyword_results)}, "
                f"final_results: {len(final_results)}, query_time: {total_time}"
            )
            
            return final_results, metrics
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise e
    
    async def _execute_vector_search(
        self,
        query_embedding: List[float],
        config: Optional[SearchConfig] = None,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Execute vector search using Supabase stored procedure."""
        
        if not config:
            config = SearchConfig()
        
        start_time = time.time()
        
        try:
            # Validate and convert user_id to UUID format
            filter_user_id = self._validate_user_id(user_id)
            
            # Call the stored procedure with correct parameter names
            response = self.supabase.rpc(
                "search_embeddings",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": config.similarity_threshold,
                    "match_count": config.max_results,
                    "document_ids": document_ids,
                    "filter_user_id": filter_user_id  # Use validated UUID or None
                }
            ).execute()
            
            search_time = time.time() - start_time
            
            # Process results
            results = []
            for item in response.data:
                result = SearchResult(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    chunk_text=item["chunk_text"],
                    similarity=float(item["similarity"]),
                    chunk_index=item.get("chunk_index", 0),
                    metadata=item.get("metadata", {}),
                    filename=item.get("filename", "")
                )
                results.append(result)
            
            # Create metrics
            metrics = SearchMetrics(
                query_time=search_time,
                embedding_time=0.0,
                search_time=search_time,
                total_results=len(results),
                filtered_results=len(results),
                avg_similarity=np.mean([r.similarity for r in results]) if results else 0.0
            )
            
            logger.info(f"Vector search completed: {len(results)} results in {search_time:.3f}s")
            return results, metrics
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Vector search failed: {str(e)}")
            
            # Return empty results with error metrics
            metrics = SearchMetrics(
                query_time=search_time,
                embedding_time=0.0,
                search_time=search_time,
                total_results=0,
                filtered_results=0,
                avg_similarity=0.0
            )
            
            return [], metrics
    
    async def _execute_keyword_search(
        self,
        search_terms: List[str],
        max_results: int,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute keyword-based search using PostgreSQL full-text search."""
        try:
            # Ensure search_terms is a list
            if not isinstance(search_terms, list):
                logger.error(f"search_terms is not a list: {type(search_terms)}, value: {search_terms}")
                search_terms = [str(search_terms)]
            
            if not search_terms:
                logger.warning("Empty search terms provided")
                return []
            
            # Build the search query
            search_query = " | ".join(search_terms)  # OR search
            
            # Validate and convert user_id to UUID format
            filter_user_id = self._validate_user_id(user_id)
            
            # Use the database function for keyword search
            # Note: Using the actual function signature in the database
            result = self.supabase.rpc(
                'search_keywords',
                {
                    'document_ids': document_ids,
                    'filter_user_id': filter_user_id,  # Use validated UUID or None
                    'match_count': max_results,
                    'search_query': search_query
                }
            ).execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error executing keyword search: {e}")
            raise e
    
    async def _post_process_results(
        self,
        results: List[SearchResult],
        query: str,
        config: SearchConfig
    ) -> List[SearchResult]:
        """Apply post-processing to search results."""
        
        if not results:
            return []
        
        processed_results = []
        
        for result in results:
            # Apply recency boost if enabled
            if config.boost_recent:
                boosted_similarity = await self._apply_recency_boost(
                    result, config.boost_factor
                )
                # Create new result with boosted similarity
                processed_result = SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    filename=result.filename,
                    chunk_text=result.chunk_text,
                    similarity=boosted_similarity,
                    chunk_index=result.chunk_index,
                    metadata=result.metadata
                )
            else:
                processed_result = result
            
            processed_results.append(processed_result)
        
        # Sort by similarity score (descending)
        processed_results.sort(key=lambda x: x.similarity, reverse=True)
        
        # Apply similarity threshold filter
        filtered_results = [
            r for r in processed_results 
            if r.similarity >= config.similarity_threshold
        ]
        
        # Limit results
        return filtered_results[:config.max_results]
    
    async def _prepare_search_terms(self, query: str) -> List[str]:
        """Prepare search terms for keyword search."""
        # Basic text preprocessing
        terms = query.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        # If no terms remain after filtering, use the original query
        if not terms:
            terms = [query.lower()]
        
        return terms
    
    async def _calculate_keyword_scores(
        self,
        results: List[Dict[str, Any]],
        query: str,
        search_terms: List[str]
    ) -> List[SearchResult]:
        """Calculate relevance scores for keyword search results."""
        scored_results = []
        
        for result in results:
            # The database function already returns the rank score
            search_result = SearchResult(
                chunk_id=result['chunk_id'],
                document_id=result['document_id'],
                filename=result['filename'],
                chunk_text=result['chunk_text'],
                similarity=result['rank'],  # Use the rank from the database function
                chunk_index=result['chunk_index'],
                metadata=result.get('metadata', {})
            )
            
            scored_results.append(search_result)
        
        return scored_results
    
    async def _combine_search_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """Combine and rank semantic and keyword search results."""
        # Create a dictionary to merge results by chunk_id
        combined_scores = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_scores[result.chunk_id] = {
                'result': result,
                'semantic_score': result.similarity,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            if result.chunk_id in combined_scores:
                combined_scores[result.chunk_id]['keyword_score'] = result.similarity
            else:
                combined_scores[result.chunk_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': result.similarity
                }
        
        # Calculate combined scores
        final_results = []
        for chunk_id, data in combined_scores.items():
            combined_score = (
                data['semantic_score'] * semantic_weight +
                data['keyword_score'] * keyword_weight
            )
            
            result = data['result']
            result.similarity = combined_score
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x.similarity, reverse=True)
        
        return final_results
    
    async def _apply_recency_boost(
        self,
        result: SearchResult,
        boost_factor: float
    ) -> float:
        """Apply recency boost to search results."""
        # This is a simplified implementation
        # In practice, you'd use document creation/update timestamps
        return result.similarity * boost_factor
    
    async def _update_search_stats(self, query: str, query_time: float):
        """Update search statistics for analytics."""
        self.search_stats["total_searches"] += 1
        
        # Update average query time
        current_avg = self.search_stats["avg_query_time"]
        total_searches = self.search_stats["total_searches"]
        self.search_stats["avg_query_time"] = (
            (current_avg * (total_searches - 1) + query_time) / total_searches
        )
        
        # Track popular queries
        query_key = query.lower().strip()
        if query_key in self.search_stats["popular_queries"]:
            self.search_stats["popular_queries"][query_key] += 1
        else:
            self.search_stats["popular_queries"][query_key] = 1
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics."""
        # Get top 10 popular queries
        popular_queries = sorted(
            self.search_stats["popular_queries"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_searches": self.search_stats["total_searches"],
            "avg_query_time": self.search_stats["avg_query_time"],
            "cache_hits": self.search_stats["cache_hits"],
            "popular_queries": popular_queries,
            "performance_metrics": {
                "avg_query_time_ms": self.search_stats["avg_query_time"] * 1000,
                "searches_per_minute": self.search_stats["total_searches"] / max(1, time.time() / 60)
            }
        }
    
    async def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """Generate query suggestions based on popular searches."""
        partial_lower = partial_query.lower()
        suggestions = []
        
        for query, count in self.search_stats["popular_queries"].items():
            if partial_lower in query and query != partial_lower:
                suggestions.append(query)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def _validate_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Validate and convert user_id to proper UUID format or None."""
        if not user_id:
            return None
        
        # If it's already a valid UUID, return it
        try:
            import uuid
            uuid.UUID(user_id)
            return user_id
        except (ValueError, TypeError):
            # If it's not a valid UUID (like "test-user"), return None
            # This will make the search work without user filtering
            logger.warning(f"Invalid UUID format for user_id: {user_id}, proceeding without user filter")
            return None 