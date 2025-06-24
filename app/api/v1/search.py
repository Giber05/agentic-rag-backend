"""
Search API endpoints for semantic, keyword, and hybrid search.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ...core.dependencies import get_supabase_client, get_openai_service, get_vector_search_service
from ...models.document_models import SearchRequest, SearchResponse, SearchResult
from ...services.vector_search_service import VectorSearchService, SearchConfig, SearchType
from ...models.base import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> SearchResponse:
    """
    Perform semantic vector search using embeddings.
    
    This endpoint uses OpenAI embeddings to find semantically similar content
    in the document database using pgvector similarity search.
    """
    try:
        # Create search configuration
        config = SearchConfig(
            similarity_threshold=request.similarity_threshold,
            max_results=request.max_results,
            boost_recent=request.boost_recent
        )
        
        # Perform semantic search
        results, metrics = await search_service.semantic_search(
            query=request.query,
            config=config,
            document_ids=request.document_ids,
            user_id=request.user_id
        )
        
        logger.info(
            f"Semantic search completed - query: {request.query}, "
            f"results_count: {len(results)}, query_time: {metrics.query_time}"
        )
        
        return SearchResponse(
            query=request.query,
            search_type=SearchType.SEMANTIC.value,
            results=results,
            total_results=metrics.total_results,
            filtered_results=metrics.filtered_results,
            query_time=metrics.query_time,
            avg_similarity=metrics.avg_similarity,
            metadata={
                "embedding_time": metrics.embedding_time,
                "search_time": metrics.search_time,
                "similarity_threshold": config.similarity_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(
    request: SearchRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> SearchResponse:
    """
    Perform keyword-based text search.
    
    This endpoint uses PostgreSQL full-text search to find documents
    containing the specified keywords.
    """
    try:
        # Create search configuration
        config = SearchConfig(
            max_results=request.max_results,
            boost_recent=request.boost_recent
        )
        
        # Perform keyword search
        results, metrics = await search_service.keyword_search(
            query=request.query,
            config=config,
            document_ids=request.document_ids,
            user_id=request.user_id
        )
        
        logger.info(
            f"Keyword search completed - query: {request.query}, "
            f"results_count: {len(results)}, query_time: {metrics.query_time}"
        )
        
        return SearchResponse(
            query=request.query,
            search_type=SearchType.KEYWORD.value,
            results=results,
            total_results=metrics.total_results,
            filtered_results=metrics.filtered_results,
            query_time=metrics.query_time,
            avg_similarity=metrics.avg_similarity,
            metadata={
                "search_time": metrics.search_time,
                "search_method": "postgresql_fulltext"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Keyword search failed: {str(e)}"
        )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> SearchResponse:
    """
    Perform hybrid search combining semantic and keyword search.
    
    This endpoint combines the results of both semantic vector search
    and keyword-based search, providing the best of both approaches.
    """
    try:
        # Create search configuration
        config = SearchConfig(
            similarity_threshold=request.similarity_threshold,
            max_results=request.max_results,
            boost_recent=request.boost_recent,
            semantic_weight=request.semantic_weight or 0.7,
            keyword_weight=request.keyword_weight or 0.3
        )
        
        # Perform hybrid search
        results, metrics = await search_service.hybrid_search(
            query=request.query,
            config=config,
            document_ids=request.document_ids,
            user_id=request.user_id
        )
        
        logger.info(
            f"Hybrid search completed - query: {request.query}, "
            f"results_count: {len(results)}, query_time: {metrics.query_time}"
        )
        
        return SearchResponse(
            query=request.query,
            search_type=SearchType.HYBRID.value,
            results=results,
            total_results=metrics.total_results,
            filtered_results=metrics.filtered_results,
            query_time=metrics.query_time,
            avg_similarity=metrics.avg_similarity,
            metadata={
                "embedding_time": metrics.embedding_time,
                "search_time": metrics.search_time,
                "semantic_weight": config.semantic_weight,
                "keyword_weight": config.keyword_weight,
                "similarity_threshold": config.similarity_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.get("/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of suggestions"),
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> List[str]:
    """
    Get search query suggestions based on popular searches.
    
    This endpoint provides autocomplete suggestions based on
    previously executed search queries.
    """
    try:
        suggestions = await search_service.suggest_queries(query, limit)
        
        logger.info(
            f"Search suggestions generated - partial_query: {query}, "
            f"suggestions_count: {len(suggestions)}"
        )
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating search suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.get("/analytics")
async def get_search_analytics(
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> dict:
    """
    Get search analytics and performance metrics.
    
    This endpoint provides insights into search usage patterns,
    performance metrics, and popular queries.
    """
    try:
        analytics = search_service.get_search_analytics()
        
        logger.info("Search analytics retrieved")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving search analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@router.get("/health")
async def search_service_health(
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> dict:
    """
    Check search service health and connectivity.
    
    This endpoint verifies that the search service can connect to
    the database and OpenAI API.
    """
    try:
        # Test database connectivity
        test_result = search_service.supabase.table('documents').select('count').limit(1).execute()
        db_healthy = test_result is not None
        
        # Test OpenAI connectivity (if API key is configured)
        openai_healthy = search_service.openai_service.client is not None
        
        overall_health = db_healthy and openai_healthy
        
        health_status = {
            "status": "healthy" if overall_health else "unhealthy",
            "database_connection": "healthy" if db_healthy else "unhealthy",
            "openai_connection": "healthy" if openai_healthy else "unhealthy",
            "search_stats": search_service.get_search_analytics(),
            "service": "vector_search_service",
            "version": "1.0.0"
        }
        
        logger.info(
            f"Search service health check - status: {health_status['status']}, "
            f"db_healthy: {db_healthy}, openai_healthy: {openai_healthy}"
        )
        
        if not overall_health:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in search service health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


# Advanced search endpoints

@router.post("/advanced")
async def advanced_search(
    query: str = Query(..., description="Search query"),
    search_type: SearchType = Query(SearchType.HYBRID, description="Type of search to perform"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    max_results: int = Query(5, ge=1, le=100, description="Maximum number of results"),
    document_ids: Optional[List[str]] = Query(None, description="Filter by specific document IDs"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    boost_recent: bool = Query(True, description="Boost recent documents"),
    semantic_weight: float = Query(0.7, ge=0.0, le=1.0, description="Weight for semantic search in hybrid mode"),
    keyword_weight: float = Query(0.3, ge=0.0, le=1.0, description="Weight for keyword search in hybrid mode"),
    search_service: VectorSearchService = Depends(get_vector_search_service)
) -> SearchResponse:
    """
    Advanced search with full parameter control.
    
    This endpoint provides fine-grained control over all search parameters
    and supports all search types (semantic, keyword, hybrid).
    """
    try:
        # Validate weights for hybrid search
        if search_type == SearchType.HYBRID and abs(semantic_weight + keyword_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Semantic weight and keyword weight must sum to 1.0"
            )
        
        # Create search configuration
        config = SearchConfig(
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            boost_recent=boost_recent,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        # Perform search based on type
        if search_type == SearchType.SEMANTIC:
            results, metrics = await search_service.semantic_search(
                query=query,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
        elif search_type == SearchType.KEYWORD:
            results, metrics = await search_service.keyword_search(
                query=query,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
        else:  # HYBRID
            results, metrics = await search_service.hybrid_search(
                query=query,
                config=config,
                document_ids=document_ids,
                user_id=user_id
            )
        
        logger.info(
            f"Advanced search completed - query: {query}, "
            f"search_type: {search_type.value}, results_count: {len(results)}, "
            f"query_time: {metrics.query_time}"
        )
        
        return SearchResponse(
            query=query,
            search_type=search_type.value,
            results=results,
            total_results=metrics.total_results,
            filtered_results=metrics.filtered_results,
            query_time=metrics.query_time,
            avg_similarity=metrics.avg_similarity,
            metadata={
                "embedding_time": metrics.embedding_time,
                "search_time": metrics.search_time,
                "similarity_threshold": similarity_threshold,
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                "boost_recent": boost_recent
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search failed: {str(e)}"
        ) 