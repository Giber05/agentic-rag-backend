"""
Performance Monitoring and Optimization Utilities

This module provides performance monitoring, metrics collection, and
optimization recommendations for intelligent MCP operations.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """
    Performance monitoring and optimization for intelligent MCP operations.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Performance thresholds
        self.thresholds = {
            "max_operation_time": 30.0,  # seconds
            "max_chain_length": 5,  # operations
            "max_concurrent_operations": 10
        }
        
        # Active operation tracking
        self.active_operations: Dict[str, datetime] = {}
        self.concurrent_count = 0
    
    @asynccontextmanager
    async def monitor_operation(
        self, 
        operation_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for monitoring individual operations.
        
        Usage:
            async with monitor.monitor_operation("confluence_search") as metrics:
                result = await some_operation()
                metrics["metadata"] = {"result_count": len(result)}
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = datetime.utcnow()
        
        # Track active operation
        self.active_operations[operation_id] = start_time
        self.concurrent_count += 1
        
        metrics_container = {}
        
        try:
            yield metrics_container
            
            # Operation completed successfully
            end_time = datetime.utcnow()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                success=True,
                metadata=metadata or metrics_container.get("metadata")
            )
            
            self._record_metrics(metrics)
            
        except Exception as e:
            # Operation failed
            end_time = datetime.utcnow()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds(),
                success=False,
                error_message=str(e),
                metadata=metadata or metrics_container.get("metadata")
            )
            
            self._record_metrics(metrics)
            raise
            
        finally:
            # Clean up tracking
            self.active_operations.pop(operation_id, None)
            self.concurrent_count = max(0, self.concurrent_count - 1)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        self.operation_stats[metrics.operation_name].append(metrics)
        
        # Keep operation stats within limits
        if len(self.operation_stats[metrics.operation_name]) > self.max_history // 10:
            self.operation_stats[metrics.operation_name] = \
                self.operation_stats[metrics.operation_name][-(self.max_history // 10):]
        
        # Log warnings for performance issues
        self._check_performance_warnings(metrics)
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """Check for performance issues and log warnings."""
        if metrics.duration > self.thresholds["max_operation_time"]:
            logger.warning(
                f"Operation '{metrics.operation_name}' took {metrics.duration:.2f}s "
                f"(threshold: {self.thresholds['max_operation_time']}s)"
            )
        
        if self.concurrent_count > self.thresholds["max_concurrent_operations"]:
            logger.warning(
                f"High concurrent operations: {self.concurrent_count} "
                f"(threshold: {self.thresholds['max_concurrent_operations']})"
            )
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type."""
        if operation_name not in self.operation_stats:
            return {"error": f"No statistics available for operation '{operation_name}'"}
        
        metrics_list = self.operation_stats[operation_name]
        successful_metrics = [m for m in metrics_list if m.success]
        failed_metrics = [m for m in metrics_list if not m.success]
        
        if not metrics_list:
            return {"error": "No metrics available"}
        
        durations = [m.duration for m in successful_metrics]
        
        return {
            "operation_name": operation_name,
            "total_operations": len(metrics_list),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(metrics_list) if metrics_list else 0,
            "duration_stats": {
                "avg": sum(durations) / len(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "median": sorted(durations)[len(durations) // 2] if durations else 0
            },
            "recent_errors": [
                m.error_message for m in failed_metrics[-5:] if m.error_message
            ]
        }
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Calculate overall stats
        total_operations = len(self.metrics_history)
        successful_operations = sum(1 for m in self.metrics_history if m.success)
        failed_operations = total_operations - successful_operations
        
        # Duration statistics for successful operations
        successful_durations = [m.duration for m in self.metrics_history if m.success]
        
        return {
            "overall_stats": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / total_operations if total_operations else 0,
                "avg_duration": sum(successful_durations) / len(successful_durations) if successful_durations else 0
            },
            "active_operations": len(self.active_operations),
            "concurrent_count": self.concurrent_count,
            "operation_breakdown": {
                op_name: len(metrics) for op_name, metrics in self.operation_stats.items()
            },
            "thresholds": self.thresholds.copy()
        }
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation (compatibility method)."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = datetime.utcnow()
        
        # Track active operation
        self.active_operations[operation_id] = start_time
        self.concurrent_count += 1
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """End tracking an operation (compatibility method)."""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation ID not found: {operation_id}")
            return
        
        start_time = self.active_operations.pop(operation_id)
        end_time = datetime.utcnow()
        
        # Extract operation name from ID
        operation_name = operation_id.rsplit('_', 1)[0]
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            success=success,
            error_message=error_message,
            metadata=metadata
        )
        
        self._record_metrics(metrics)
        self.concurrent_count = max(0, self.concurrent_count - 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics (compatibility method)."""
        stats = self.get_overall_statistics()
        if "error" in stats:
            return stats
        
        overall = stats["overall_stats"]
        return {
            "total_operations": overall["total_operations"],
            "success_rate": overall["success_rate"],
            "avg_duration": overall["avg_duration"],
            "active_operations": stats["active_operations"],
            "operation_breakdown": stats["operation_breakdown"]
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics_history:
            return [{"type": "info", "message": "No metrics available for analysis"}]
        
        # Analyze recent performance
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 operations
        
        # Check for slow operations
        slow_operations = [m for m in recent_metrics if m.duration > self.thresholds["max_operation_time"]]
        if slow_operations:
            slow_ops_by_type = defaultdict(list)
            for op in slow_operations:
                slow_ops_by_type[op.operation_name].append(op.duration)
            
            for op_name, durations in slow_ops_by_type.items():
                avg_duration = sum(durations) / len(durations)
                recommendations.append({
                    "type": "performance",
                    "severity": "high" if avg_duration > self.thresholds["max_operation_time"] * 1.5 else "medium",
                    "operation": op_name,
                    "message": f"Operation '{op_name}' is running slowly (avg: {avg_duration:.2f}s)",
                    "suggestion": "Consider optimizing query parameters or adding result filtering"
                })
        
        # Check for high failure rates
        operation_failure_rates = {}
        for op_name, metrics in self.operation_stats.items():
            if len(metrics) >= 5:  # Only analyze operations with sufficient data
                failure_rate = sum(1 for m in metrics[-20:] if not m.success) / min(20, len(metrics))
                if failure_rate > 0.2:  # 20% failure rate
                    operation_failure_rates[op_name] = failure_rate
        
        for op_name, failure_rate in operation_failure_rates.items():
            recommendations.append({
                "type": "reliability",
                "severity": "high" if failure_rate > 0.5 else "medium",
                "operation": op_name,
                "message": f"Operation '{op_name}' has high failure rate: {failure_rate:.1%}",
                "suggestion": "Review error logs and consider adding retry logic or input validation"
            })
        
        # Check for too many concurrent operations
        if self.concurrent_count > self.thresholds["max_concurrent_operations"]:
            recommendations.append({
                "type": "concurrency",
                "severity": "medium",
                "message": f"High concurrent operations: {self.concurrent_count}",
                "suggestion": "Consider implementing operation queuing or reducing max_concurrent_chains"
            })
        
        # Generate positive recommendations if performance is good
        if not recommendations:
            overall_stats = self.get_overall_statistics()
            if overall_stats.get("overall_stats", {}).get("success_rate", 0) > 0.95:
                recommendations.append({
                    "type": "optimization",
                    "severity": "low",
                    "message": "System performance is excellent",
                    "suggestion": "Consider increasing max_operations or concurrent_chains for better throughput"
                })
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.metrics_history.clear()
        self.operation_stats.clear()
        self.active_operations.clear()
        self.concurrent_count = 0
        logger.info("Performance metrics reset")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


# Convenience functions
async def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function for monitoring operations."""
    return get_performance_monitor().monitor_operation(operation_name, metadata)


def get_operation_stats(operation_name: str) -> Dict[str, Any]:
    """Get statistics for a specific operation."""
    return get_performance_monitor().get_operation_statistics(operation_name)


def get_overall_stats() -> Dict[str, Any]:
    """Get overall performance statistics."""
    return get_performance_monitor().get_overall_statistics()


def get_optimization_recommendations() -> List[Dict[str, Any]]:
    """Get optimization recommendations."""
    return get_performance_monitor().get_optimization_recommendations() 