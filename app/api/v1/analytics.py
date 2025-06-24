"""
Analytics API endpoints for token usage and cost tracking.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Dict, Any
from datetime import datetime, date

from ...services.token_tracker import token_tracker
from ...models.rag_models import BaseAPIModel
from ...core.dependencies import authenticated_security_dependencies, require_admin

router = APIRouter(prefix="/analytics", tags=["analytics"])


class TokenAnalyticsResponse(BaseAPIModel):
    """Response model for token analytics."""
    request_id: str
    query: str
    query_tokens: int
    total_api_calls: int
    total_tokens: int
    total_cost: float
    pipeline_type: str
    optimization_savings: float
    breakdown: Dict[str, Any]
    timestamp: str


class DailyStatsResponse(BaseAPIModel):
    """Response model for daily statistics."""
    date: str
    total_requests: int
    total_tokens: int
    total_cost: float
    optimized_requests: int
    full_requests: int
    optimization_savings: float


class CostPatternsResponse(BaseAPIModel):
    """Response model for cost pattern analysis."""
    period: str
    total_requests: int
    total_tokens: int
    total_cost: float
    total_savings: float
    avg_cost_per_request: float
    avg_tokens_per_request: float
    optimization_rate: float
    daily_breakdown: list


@router.get("/token-usage/{request_id}", response_model=TokenAnalyticsResponse)
async def get_request_token_analysis(
    request_id: str,
    current_user = Depends(authenticated_security_dependencies)
):
    """Get detailed token analysis for a specific request."""
    analysis = token_tracker.get_request_analysis(request_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    return TokenAnalyticsResponse(**analysis.to_dict())


@router.get("/daily-stats", response_model=DailyStatsResponse)
async def get_daily_stats(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    current_user = Depends(require_admin)  # Only admins can view usage stats
):
    """Get daily token usage statistics."""
    stats = token_tracker.get_daily_stats(date)
    
    if not stats:
        # Return empty stats for the requested date
        target_date = date or datetime.utcnow().date().isoformat()
        return DailyStatsResponse(
            date=target_date,
            total_requests=0,
            total_tokens=0,
            total_cost=0.0,
            optimized_requests=0,
            full_requests=0,
            optimization_savings=0.0
        )
    
    return DailyStatsResponse(
        date=date or datetime.utcnow().date().isoformat(),
        **stats
    )


@router.get("/cost-patterns", response_model=CostPatternsResponse)
async def get_cost_patterns(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    current_user = Depends(require_admin)
):
    """Get cost pattern analysis over recent days."""
    patterns = token_tracker.analyze_cost_patterns(days=days)
    return CostPatternsResponse(**patterns)


@router.get("/recent-requests")
async def get_recent_requests(limit: int = Query(10, ge=1, le=50, description="Number of recent requests")):
    """Get recent request analyses."""
    requests = token_tracker.get_recent_requests(limit=limit)
    return [TokenAnalyticsResponse(**req.to_dict()) for req in requests]


@router.get("/monthly-projection")
async def get_monthly_projection():
    """Get estimated monthly cost projection."""
    projection = token_tracker.estimate_monthly_cost()
    return projection


@router.get("/model-pricing")
async def get_model_pricing():
    """Get current model pricing information."""
    return {
        "pricing": token_tracker.MODEL_PRICING,
        "note": "Prices are per 1K tokens in USD",
        "last_updated": "2024-01-01"
    }


@router.get("/export")
async def export_analytics(format: str = Query("json", regex="^(json)$")):
    """Export comprehensive analytics data."""
    try:
        data = token_tracker.export_analytics(format=format)
        
        if format == "json":
            import json
            return json.loads(data)
        else:
            return {"data": data}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/cost-breakdown")
async def get_cost_breakdown():
    """Get detailed cost breakdown by model and operation type."""
    recent_requests = token_tracker.get_recent_requests(limit=100)
    
    breakdown = {
        "by_model": {},
        "by_operation": {},
        "by_pipeline_type": {"optimized": 0, "full": 0},
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_requests": len(recent_requests)
    }
    
    for req in recent_requests:
        breakdown["total_cost"] += req.total_cost
        breakdown["total_tokens"] += req.total_tokens
        breakdown["by_pipeline_type"][req.pipeline_type] += req.total_cost
        
        for operation, usage in req.breakdown.items():
            # By model
            if usage.model not in breakdown["by_model"]:
                breakdown["by_model"][usage.model] = {"cost": 0.0, "tokens": 0, "calls": 0}
            breakdown["by_model"][usage.model]["cost"] += usage.cost
            breakdown["by_model"][usage.model]["tokens"] += usage.total_tokens
            breakdown["by_model"][usage.model]["calls"] += 1
            
            # By operation
            if operation not in breakdown["by_operation"]:
                breakdown["by_operation"][operation] = {"cost": 0.0, "tokens": 0, "calls": 0}
            breakdown["by_operation"][operation]["cost"] += usage.cost
            breakdown["by_operation"][operation]["tokens"] += usage.total_tokens
            breakdown["by_operation"][operation]["calls"] += 1
    
    return breakdown


@router.get("/optimization-impact")
async def get_optimization_impact():
    """Get analysis of optimization impact and savings."""
    patterns = token_tracker.analyze_cost_patterns(days=30)
    
    if patterns["total_requests"] == 0:
        return {
            "message": "No data available for analysis",
            "total_savings": 0.0,
            "optimization_rate": 0.0
        }
    
    # Calculate what costs would have been without optimization
    estimated_full_cost = patterns["total_cost"] + patterns["total_savings"]
    savings_percentage = (patterns["total_savings"] / estimated_full_cost * 100) if estimated_full_cost > 0 else 0
    
    return {
        "period": patterns["period"],
        "actual_cost": patterns["total_cost"],
        "estimated_full_cost": estimated_full_cost,
        "total_savings": patterns["total_savings"],
        "savings_percentage": savings_percentage,
        "optimization_rate": patterns["optimization_rate"],
        "avg_cost_per_request": patterns["avg_cost_per_request"],
        "avg_tokens_per_request": patterns["avg_tokens_per_request"],
        "recommendations": [
            "Continue using optimized pipeline for simple queries",
            "Monitor complex query patterns for further optimization",
            "Consider caching frequently asked questions"
        ]
    }


@router.post("/reset-stats")
async def reset_statistics():
    """Reset all token tracking statistics (use with caution)."""
    try:
        token_tracker.request_analyses.clear()
        token_tracker.daily_stats.clear()
        return {"message": "Statistics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}") 