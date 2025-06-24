"""
Token tracking service for analyzing OpenAI API usage and costs.
"""

import logging
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float
    timestamp: datetime
    request_type: str  # 'chat', 'embedding'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RequestTokenAnalysis:
    """Complete token analysis for a RAG request."""
    request_id: str
    query: str
    query_tokens: int
    total_api_calls: int
    total_tokens: int
    total_cost: float
    pipeline_type: str  # 'optimized', 'full'
    optimization_savings: float
    breakdown: Dict[str, TokenUsage]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'breakdown': {k: v.to_dict() for k, v in self.breakdown.items()}
        }


class TokenTracker:
    """Service for tracking and analyzing token usage across the RAG pipeline."""
    
    # Model pricing per 1K tokens (input/output)
    MODEL_PRICING = {
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0001},
        'text-embedding-3-small': {'input': 0.00002, 'output': 0.00002},
        'text-embedding-3-large': {'input': 0.00013, 'output': 0.00013},
        'gpt-4.1-nano': {'input': 0.0001, 'output': 0.0001},
        'gpt-4.1-mini': {'input': 0.0001, 'output': 0.0001},
    }
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self.request_analyses: Dict[str, RequestTokenAnalysis] = {}
        self.daily_stats = defaultdict(lambda: {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'optimized_requests': 0,
            'full_requests': 0,
            'optimization_savings': 0.0
        })
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def calculate_cost(self, tokens: int, model: str, is_input: bool = True) -> float:
        """Calculate cost for token usage."""
        if model not in self.MODEL_PRICING:
            logger.warning(f"Unknown model pricing for {model}, using GPT-3.5-turbo")
            model = 'gpt-3.5-turbo'
        
        pricing = self.MODEL_PRICING[model]
        rate = pricing['input'] if is_input else pricing['output']
        return (tokens / 1000) * rate
    
    def start_request_tracking(self, request_id: str, query: str, pipeline_type: str) -> None:
        """Start tracking a new RAG request."""
        query_tokens = self.count_tokens(query)
        
        self.request_analyses[request_id] = RequestTokenAnalysis(
            request_id=request_id,
            query=query,
            query_tokens=query_tokens,
            total_api_calls=0,
            total_tokens=0,
            total_cost=0.0,
            pipeline_type=pipeline_type,
            optimization_savings=0.0,
            breakdown={},
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Started tracking request {request_id}: {query_tokens} query tokens")
    
    def track_api_call(
        self,
        request_id: str,
        call_type: str,  # 'query_rewriting', 'context_decision', 'embedding', 'answer_generation'
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        prompt_text: Optional[str] = None,
        completion_text: Optional[str] = None
    ) -> TokenUsage:
        """Track an individual API call within a request."""
        
        # If tokens not provided, calculate from text
        if prompt_text and prompt_tokens == 0:
            prompt_tokens = self.count_tokens(prompt_text)
        if completion_text and completion_tokens == 0:
            completion_tokens = self.count_tokens(completion_text)
        
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate costs
        input_cost = self.calculate_cost(prompt_tokens, model, is_input=True)
        output_cost = self.calculate_cost(completion_tokens, model, is_input=False)
        total_cost = input_cost + output_cost
        
        # Determine request type
        request_type = 'embedding' if 'embedding' in model else 'chat'
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            cost=total_cost,
            timestamp=datetime.utcnow(),
            request_type=request_type
        )
        
        # Update request analysis
        if request_id in self.request_analyses:
            analysis = self.request_analyses[request_id]
            analysis.total_api_calls += 1
            analysis.total_tokens += total_tokens
            analysis.total_cost += total_cost
            analysis.breakdown[call_type] = usage
        
        logger.info(f"Tracked {call_type} call for {request_id}: {total_tokens} tokens, ${total_cost:.4f}")
        return usage
    
    def calculate_optimization_savings(self, request_id: str, baseline_cost: float) -> float:
        """Calculate how much was saved through optimization."""
        if request_id not in self.request_analyses:
            return 0.0
        
        analysis = self.request_analyses[request_id]
        savings = baseline_cost - analysis.total_cost
        analysis.optimization_savings = max(0.0, savings)
        
        return analysis.optimization_savings
    
    def finish_request_tracking(self, request_id: str) -> Optional[RequestTokenAnalysis]:
        """Finish tracking a request and update daily stats."""
        if request_id not in self.request_analyses:
            logger.warning(f"Request {request_id} not found in tracking")
            return None
        
        analysis = self.request_analyses[request_id]
        today = datetime.utcnow().date().isoformat()
        
        # Update daily stats
        stats = self.daily_stats[today]
        stats['total_requests'] += 1
        stats['total_tokens'] += analysis.total_tokens
        stats['total_cost'] += analysis.total_cost
        stats['optimization_savings'] += analysis.optimization_savings
        
        if analysis.pipeline_type == 'optimized':
            stats['optimized_requests'] += 1
        else:
            stats['full_requests'] += 1
        
        logger.info(f"Finished tracking {request_id}: {analysis.total_tokens} tokens, ${analysis.total_cost:.4f}")
        return analysis
    
    def get_request_analysis(self, request_id: str) -> Optional[RequestTokenAnalysis]:
        """Get analysis for a specific request."""
        return self.request_analyses.get(request_id)
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily statistics."""
        if date is None:
            date = datetime.utcnow().date().isoformat()
        
        return dict(self.daily_stats.get(date, {}))
    
    def get_recent_requests(self, limit: int = 10) -> List[RequestTokenAnalysis]:
        """Get recent request analyses."""
        analyses = list(self.request_analyses.values())
        analyses.sort(key=lambda x: x.timestamp, reverse=True)
        return analyses[:limit]
    
    def analyze_cost_patterns(self, days: int = 7) -> Dict[str, Any]:
        """Analyze cost patterns over recent days."""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days-1)
        
        total_cost = 0.0
        total_tokens = 0
        total_requests = 0
        optimized_requests = 0
        total_savings = 0.0
        daily_breakdown = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            stats = self.daily_stats.get(date_str, {})
            
            daily_breakdown.append({
                'date': date_str,
                'requests': stats.get('total_requests', 0),
                'tokens': stats.get('total_tokens', 0),
                'cost': stats.get('total_cost', 0.0),
                'optimized_requests': stats.get('optimized_requests', 0),
                'savings': stats.get('optimization_savings', 0.0)
            })
            
            total_cost += stats.get('total_cost', 0.0)
            total_tokens += stats.get('total_tokens', 0)
            total_requests += stats.get('total_requests', 0)
            optimized_requests += stats.get('optimized_requests', 0)
            total_savings += stats.get('optimization_savings', 0.0)
            
            current_date += timedelta(days=1)
        
        avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0
        avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0.0
        optimization_rate = optimized_requests / total_requests if total_requests > 0 else 0.0
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_requests': total_requests,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'total_savings': total_savings,
            'avg_cost_per_request': avg_cost_per_request,
            'avg_tokens_per_request': avg_tokens_per_request,
            'optimization_rate': optimization_rate,
            'daily_breakdown': daily_breakdown
        }
    
    def estimate_monthly_cost(self) -> Dict[str, Any]:
        """Estimate monthly cost based on recent usage."""
        # Get last 7 days of data
        recent_stats = self.analyze_cost_patterns(days=7)
        
        if recent_stats['total_requests'] == 0:
            return {
                'estimated_monthly_cost': 0.0,
                'estimated_monthly_requests': 0,
                'estimated_monthly_tokens': 0,
                'confidence': 'low'
            }
        
        # Calculate daily averages
        daily_avg_cost = recent_stats['total_cost'] / 7
        daily_avg_requests = recent_stats['total_requests'] / 7
        daily_avg_tokens = recent_stats['total_tokens'] / 7
        
        # Project to monthly (30 days)
        monthly_cost = daily_avg_cost * 30
        monthly_requests = daily_avg_requests * 30
        monthly_tokens = daily_avg_tokens * 30
        
        # Determine confidence based on data volume
        confidence = 'high' if recent_stats['total_requests'] >= 50 else 'medium' if recent_stats['total_requests'] >= 10 else 'low'
        
        return {
            'estimated_monthly_cost': monthly_cost,
            'estimated_monthly_requests': int(monthly_requests),
            'estimated_monthly_tokens': int(monthly_tokens),
            'confidence': confidence,
            'based_on_days': 7,
            'recent_daily_avg': {
                'cost': daily_avg_cost,
                'requests': daily_avg_requests,
                'tokens': daily_avg_tokens
            }
        }
    
    def export_analytics(self, format: str = 'json') -> str:
        """Export analytics data."""
        data = {
            'summary': self.analyze_cost_patterns(days=30),
            'monthly_projection': self.estimate_monthly_cost(),
            'recent_requests': [r.to_dict() for r in self.get_recent_requests(20)],
            'model_pricing': self.MODEL_PRICING,
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return str(data)


# Global instance
token_tracker = TokenTracker() 