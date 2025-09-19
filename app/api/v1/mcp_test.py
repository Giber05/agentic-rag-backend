"""
Test endpoints for MCP Jira integration.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ...core.dependencies import security_dependencies
from ...services.mcp_atlassian_service import MCPAtlassianService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Testing"])

@router.get("/jira/health")
async def test_jira_health(
    current_user = Depends(security_dependencies)
) -> Dict[str, Any]:
    """Test MCP Jira service health."""
    try:
        service = MCPAtlassianService()
        health_status = await service.health_check()
        
        return {
            "success": True,
            "health_status": health_status,
            "message": "MCP Jira health check completed"
        }
        
    except Exception as e:
        logger.error(f"MCP Jira health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP Jira health check failed: {str(e)}"
        )

@router.get("/jira/projects")
async def test_jira_projects(
    current_user = Depends(security_dependencies)
) -> Dict[str, Any]:
    """Test retrieving Jira projects."""
    try:
        service = MCPAtlassianService()
        await service.initialize()
        
        result = await service.get_jira_projects()
        
        return {
            "success": result["success"],
            "projects": result["data"] if result["success"] else None,
            "error": result.get("error"),
            "stats": service.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Jira projects: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Jira projects: {str(e)}"
        )

@router.get("/jira/search")
async def test_jira_search(
    query: str,
    max_results: int = 5,
    current_user = Depends(security_dependencies)
) -> Dict[str, Any]:
    """Test searching Jira issues."""
    try:
        service = MCPAtlassianService()
        await service.initialize()
        
        result = await service.search_jira_issues(
            query=query,
            max_results=max_results
        )
        
        return {
            "success": result["success"],
            "query": query,
            "issues": result["data"] if result["success"] else None,
            "metadata": result.get("metadata", {}),
            "error": result.get("error"),
            "stats": service.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to search Jira issues: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search Jira issues: {str(e)}"
        )

@router.post("/test/source-retrieval")
async def test_source_retrieval(
    query: str,
    source: str = "db",
    current_user = Depends(security_dependencies)
) -> Dict[str, Any]:
    """Test enhanced source retrieval with different sources."""
    try:
        from ...agents.enhanced_source_retrieval import EnhancedSourceRetrievalAgent
        
        # Create and initialize agent
        agent = EnhancedSourceRetrievalAgent(
            config={
                "jira_enabled": True,
                "jira_max_results": 5
            }
        )
        await agent.start()
        
        # Process query
        result = await agent.process({
            "query": query,
            "source": source,
            "conversation_history": [],
            "retrieval_config": {"max_results": 10}
        })
        
        return {
            "success": True,
            "query": query,
            "source": source,
            "result": result,
            "agent_health": await agent.health_check()
        }
        
    except Exception as e:
        logger.error(f"Enhanced source retrieval test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced source retrieval test failed: {str(e)}"
        )
    finally:
        try:
            await agent.stop()
        except:
            pass 