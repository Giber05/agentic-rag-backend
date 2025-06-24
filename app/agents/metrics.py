"""
Agent metrics and performance monitoring.
"""

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for a single agent."""
    agent_id: str
    agent_type: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    last_operation_time: Optional[datetime] = None
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        if self.successful_operations == 0:
            return 0.0
        return self.total_processing_time_ms / self.successful_operations


class AgentMetrics:
    """
    Comprehensive metrics collection and analysis for agents.
    
    Provides:
    - Real-time performance monitoring
    - Historical trend analysis
    - Alerting and anomaly detection
    - Performance optimization insights
    """
    
    def __init__(self, retention_hours: int = 24, max_points_per_metric: int = 1000):
        """
        Initialize the metrics system.
        
        Args:
            retention_hours: How long to keep detailed metrics
            max_points_per_metric: Maximum data points per metric
        """
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        
        # Agent performance tracking
        self._agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        
        # Time-series metrics
        self._time_series_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        
        # Operation tracking
        self._operation_history: deque = deque(maxlen=10000)
        
        # Alert thresholds
        self._alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "avg_processing_time_ms": 5000,  # 5 seconds
            "throughput_drop_percentage": 0.5  # 50% throughput drop
        }
        
        # Performance baselines
        self._performance_baselines: Dict[str, Dict[str, float]] = {}
        
        logger.info("Agent metrics system initialized")
    
    def record_operation(
        self,
        agent_id: str,
        agent_type: str,
        success: bool,
        processing_time_ms: float,
        operation_type: str = "process",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an agent operation for metrics tracking.
        
        Args:
            agent_id: ID of the agent
            agent_type: Type of the agent
            success: Whether the operation was successful
            processing_time_ms: Processing time in milliseconds
            operation_type: Type of operation performed
            metadata: Additional operation metadata
        """
        timestamp = datetime.utcnow()
        
        # Initialize agent metrics if not exists
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_type=agent_type
            )
        
        metrics = self._agent_metrics[agent_id]
        
        # Update basic counters
        metrics.total_operations += 1
        metrics.last_operation_time = timestamp
        
        if success:
            metrics.successful_operations += 1
            metrics.total_processing_time_ms += processing_time_ms
            
            # Update min/max processing times
            metrics.min_processing_time_ms = min(metrics.min_processing_time_ms, processing_time_ms)
            metrics.max_processing_time_ms = max(metrics.max_processing_time_ms, processing_time_ms)
        else:
            metrics.failed_operations += 1
        
        # Update derived metrics
        metrics.error_rate = metrics.failed_operations / metrics.total_operations
        
        # Record time-series data
        self._record_time_series(agent_id, "processing_time_ms", processing_time_ms, timestamp)
        self._record_time_series(agent_id, "success_rate", 1.0 if success else 0.0, timestamp)
        self._record_time_series(agent_id, "operations_count", 1.0, timestamp)
        
        # Record operation history
        self._operation_history.append({
            "timestamp": timestamp,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "success": success,
            "processing_time_ms": processing_time_ms,
            "operation_type": operation_type,
            "metadata": metadata or {}
        })
        
        # Clean up old data
        self._cleanup_old_data()
        
        # Update throughput calculations
        self._update_throughput_metrics()
        
        logger.debug(f"Recorded operation for agent {agent_id}: success={success}, time={processing_time_ms}ms")
    
    def _record_time_series(self, agent_id: str, metric_name: str, value: float, timestamp: datetime) -> None:
        """Record a time-series metric point."""
        key = f"{agent_id}:{metric_name}"
        self._time_series_metrics[key].append(MetricPoint(timestamp, value))
    
    def _cleanup_old_data(self) -> None:
        """Remove old metric data beyond retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        # Clean up time-series data
        for key, points in self._time_series_metrics.items():
            while points and points[0].timestamp < cutoff_time:
                points.popleft()
        
        # Clean up operation history
        while self._operation_history and self._operation_history[0]["timestamp"] < cutoff_time:
            self._operation_history.popleft()
    
    def _update_throughput_metrics(self) -> None:
        """Update throughput calculations for all agents."""
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Count operations per agent in the last minute
        agent_operations = defaultdict(int)
        for operation in self._operation_history:
            if operation["timestamp"] >= one_minute_ago:
                agent_operations[operation["agent_id"]] += 1
        
        # Update throughput metrics
        for agent_id, count in agent_operations.items():
            if agent_id in self._agent_metrics:
                self._agent_metrics[agent_id].throughput_per_minute = count
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """
        Get performance metrics for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            AgentPerformanceMetrics or None if not found
        """
        return self._agent_metrics.get(agent_id)
    
    def get_all_agent_metrics(self) -> Dict[str, AgentPerformanceMetrics]:
        """
        Get performance metrics for all agents.
        
        Returns:
            Dictionary mapping agent IDs to their metrics
        """
        return self._agent_metrics.copy()
    
    def get_agent_type_metrics(self, agent_type: str) -> List[AgentPerformanceMetrics]:
        """
        Get metrics for all agents of a specific type.
        
        Args:
            agent_type: Type of agents to get metrics for
            
        Returns:
            List of metrics for agents of the specified type
        """
        return [
            metrics for metrics in self._agent_metrics.values()
            if metrics.agent_type == agent_type
        ]
    
    def get_time_series_data(
        self,
        agent_id: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """
        Get time-series data for a specific metric.
        
        Args:
            agent_id: ID of the agent
            metric_name: Name of the metric
            start_time: Start time for data (default: 1 hour ago)
            end_time: End time for data (default: now)
            
        Returns:
            List of metric points in the specified time range
        """
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.utcnow()
        
        key = f"{agent_id}:{metric_name}"
        points = self._time_series_metrics.get(key, deque())
        
        return [
            point for point in points
            if start_time <= point.timestamp <= end_time
        ]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get a system-wide metrics overview.
        
        Returns:
            Dictionary containing system-wide metrics
        """
        total_agents = len(self._agent_metrics)
        total_operations = sum(m.total_operations for m in self._agent_metrics.values())
        total_successful = sum(m.successful_operations for m in self._agent_metrics.values())
        total_failed = sum(m.failed_operations for m in self._agent_metrics.values())
        
        # Calculate system-wide averages
        avg_success_rate = 0.0
        avg_processing_time = 0.0
        total_throughput = 0.0
        
        if total_agents > 0:
            avg_success_rate = sum(m.success_rate for m in self._agent_metrics.values()) / total_agents
            
            successful_agents = [m for m in self._agent_metrics.values() if m.successful_operations > 0]
            if successful_agents:
                avg_processing_time = sum(m.average_processing_time_ms for m in successful_agents) / len(successful_agents)
            
            total_throughput = sum(m.throughput_per_minute for m in self._agent_metrics.values())
        
        # Agent type distribution
        type_distribution = defaultdict(int)
        for metrics in self._agent_metrics.values():
            type_distribution[metrics.agent_type] += 1
        
        return {
            "total_agents": total_agents,
            "total_operations": total_operations,
            "successful_operations": total_successful,
            "failed_operations": total_failed,
            "system_success_rate": total_successful / max(1, total_operations),
            "system_error_rate": total_failed / max(1, total_operations),
            "average_success_rate": avg_success_rate,
            "average_processing_time_ms": avg_processing_time,
            "total_throughput_per_minute": total_throughput,
            "agent_type_distribution": dict(type_distribution),
            "active_agents": len([m for m in self._agent_metrics.values() if m.last_operation_time and 
                                (datetime.utcnow() - m.last_operation_time).total_seconds() < 300])
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies across all agents.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for agent_id, metrics in self._agent_metrics.items():
            # Check error rate
            if metrics.error_rate > self._alert_thresholds["error_rate"]:
                anomalies.append({
                    "type": "high_error_rate",
                    "agent_id": agent_id,
                    "agent_type": metrics.agent_type,
                    "current_value": metrics.error_rate,
                    "threshold": self._alert_thresholds["error_rate"],
                    "severity": "high" if metrics.error_rate > 0.2 else "medium"
                })
            
            # Check processing time
            if metrics.average_processing_time_ms > self._alert_thresholds["avg_processing_time_ms"]:
                anomalies.append({
                    "type": "slow_processing",
                    "agent_id": agent_id,
                    "agent_type": metrics.agent_type,
                    "current_value": metrics.average_processing_time_ms,
                    "threshold": self._alert_thresholds["avg_processing_time_ms"],
                    "severity": "medium"
                })
            
            # Check for agents that haven't been active recently
            if metrics.last_operation_time:
                inactive_duration = (datetime.utcnow() - metrics.last_operation_time).total_seconds()
                if inactive_duration > 3600:  # 1 hour
                    anomalies.append({
                        "type": "inactive_agent",
                        "agent_id": agent_id,
                        "agent_type": metrics.agent_type,
                        "current_value": inactive_duration,
                        "threshold": 3600,
                        "severity": "low"
                    })
        
        return anomalies
    
    def get_performance_trends(self, agent_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends for an agent over a specified period.
        
        Args:
            agent_id: ID of the agent
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get time-series data
        processing_times = self.get_time_series_data(agent_id, "processing_time_ms", start_time, end_time)
        success_rates = self.get_time_series_data(agent_id, "success_rate", start_time, end_time)
        operation_counts = self.get_time_series_data(agent_id, "operations_count", start_time, end_time)
        
        if not processing_times:
            return {"error": "No data available for the specified period"}
        
        # Calculate trends
        processing_time_values = [p.value for p in processing_times]
        success_rate_values = [p.value for p in success_rates]
        
        return {
            "period_hours": hours,
            "data_points": len(processing_times),
            "processing_time_trend": {
                "min": min(processing_time_values),
                "max": max(processing_time_values),
                "avg": sum(processing_time_values) / len(processing_time_values),
                "trend": self._calculate_trend(processing_time_values)
            },
            "success_rate_trend": {
                "avg": sum(success_rate_values) / len(success_rate_values),
                "trend": self._calculate_trend(success_rate_values)
            },
            "operation_volume": len(operation_counts),
            "operations_per_hour": len(operation_counts) / hours
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percentage = (second_avg - first_avg) / max(first_avg, 0.001) * 100
        
        if change_percentage > 10:
            return "increasing"
        elif change_percentage < -10:
            return "decreasing"
        else:
            return "stable"
    
    def export_metrics(self, format: str = "json") -> Dict[str, Any]:
        """
        Export all metrics data.
        
        Args:
            format: Export format (currently only 'json' supported)
            
        Returns:
            Exported metrics data
        """
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "agent_metrics": {
                agent_id: {
                    "agent_id": metrics.agent_id,
                    "agent_type": metrics.agent_type,
                    "total_operations": metrics.total_operations,
                    "successful_operations": metrics.successful_operations,
                    "failed_operations": metrics.failed_operations,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "average_processing_time_ms": metrics.average_processing_time_ms,
                    "min_processing_time_ms": metrics.min_processing_time_ms,
                    "max_processing_time_ms": metrics.max_processing_time_ms,
                    "throughput_per_minute": metrics.throughput_per_minute,
                    "last_operation_time": metrics.last_operation_time.isoformat() if metrics.last_operation_time else None
                }
                for agent_id, metrics in self._agent_metrics.items()
            },
            "system_overview": self.get_system_overview(),
            "anomalies": self.detect_anomalies()
        } 