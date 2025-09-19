"""
Intelligent MCP Configuration System

This module provides comprehensive configuration management for intelligent
MCP operations, including operation templates, performance settings, and
environment-based configurations.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from pydantic import BaseModel, Field, validator


class OperationTemplate(BaseModel):
    """Configuration for operation templates."""
    
    name: str = Field(..., description="Template name")
    intent_types: List[str] = Field(..., description="Associated intent types")
    operations: List[Dict[str, Any]] = Field(..., description="Operation sequence")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Execution conditions")
    priority: int = Field(default=1, description="Template priority (1-10)")
    enabled: bool = Field(default=True, description="Whether template is enabled")
    description: str = Field(default="", description="Template description")
    
    @validator('priority')
    def validate_priority(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Priority must be between 1 and 10')
        return v


class PerformanceConfig(BaseModel):
    """Performance and resource configuration."""
    
    max_operations: int = Field(default=5, ge=1, le=20, description="Maximum operations per chain")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Chain timeout in seconds")
    cost_limit: float = Field(default=1.0, ge=0.1, le=10.0, description="Maximum cost per chain")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # Cache settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, ge=100, le=10000, description="Maximum cache entries")
    
    # Concurrency settings
    max_concurrent_chains: int = Field(default=10, ge=1, le=100, description="Max concurrent operation chains")
    operation_timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="Individual operation timeout")
    
    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay in seconds")


class IntentConfig(BaseModel):
    """Configuration for intent detection."""
    
    enabled_intents: List[str] = Field(default_factory=lambda: [
        "issue_analysis", "documentation_search", "project_overview",
        "troubleshooting", "status_check", "relationship_mapping", "general_search"
    ], description="Enabled intent types")
    
    intent_weights: Dict[str, float] = Field(default_factory=dict, description="Intent detection weights")
    entity_patterns: Dict[str, List[str]] = Field(default_factory=dict, description="Custom entity patterns")
    confidence_boost: Dict[str, float] = Field(default_factory=dict, description="Confidence boost per intent")


class MCPServiceConfig(BaseModel):
    """MCP service configuration."""
    
    jira_enabled: bool = Field(default=True, description="Enable Jira operations")
    confluence_enabled: bool = Field(default=True, description="Enable Confluence operations")
    
    # Connection settings
    connection_timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="Connection timeout")
    request_timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum connection retries")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=10, le=1000, description="Rate limit per minute")
    burst_limit: int = Field(default=10, ge=1, le=50, description="Burst request limit")


class IntelligentMCPConfig(BaseModel):
    """Main intelligent MCP configuration."""
    
    # Core settings
    enabled: bool = Field(default=True, description="Enable intelligent MCP features")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    
    # Component configurations
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    intelligence: IntentConfig = Field(default_factory=IntentConfig)
    operations: MCPServiceConfig = Field(default_factory=MCPServiceConfig)
    
    # Operation templates
    templates: List[OperationTemplate] = Field(default_factory=list)
    
    # Feature flags
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_operation_preview: bool = Field(default=True, description="Enable operation preview")
    enable_statistics: bool = Field(default=True, description="Enable statistics collection")
    enable_health_checks: bool = Field(default=True, description="Enable health monitoring")
    
    # Security settings
    require_authentication: bool = Field(default=False, description="Require API authentication")
    allowed_origins: List[str] = Field(default_factory=list, description="Allowed CORS origins")
    rate_limit_per_user: int = Field(default=100, description="Rate limit per user per hour")


class ConfigurationManager:
    """Manages intelligent MCP configuration with file-based and environment overrides."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or "config/intelligent_mcp_config.json"
        self.config_dir = Path(self.config_file).parent
        self._config: Optional[IntelligentMCPConfig] = None
        self._default_templates = self._get_default_templates()
    
    def load_config(self) -> IntelligentMCPConfig:
        """Load configuration from file and environment variables."""
        try:
            # Start with default configuration
            config_data = {}
            
            # Load from file if exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
            
            # Apply environment overrides
            config_data = self._apply_environment_overrides(config_data)
            
            # Create configuration object
            self._config = IntelligentMCPConfig(**config_data)
            
            # Add default templates if none exist
            if not self._config.templates:
                self._config.templates = self._default_templates
            
            return self._config
            
        except Exception as e:
            # Return default configuration on error
            self._config = IntelligentMCPConfig(templates=self._default_templates)
            return self._config
    
    def save_config(self, config: Optional[IntelligentMCPConfig] = None) -> bool:
        """Save configuration to file."""
        try:
            if config:
                self._config = config
            
            if not self._config:
                return False
            
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(self._config.dict(), f, indent=2, default=str)
            
            return True
            
        except Exception:
            return False
    
    def get_config(self) -> IntelligentMCPConfig:
        """Get current configuration."""
        if not self._config:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> IntelligentMCPConfig:
        """Update configuration with new values."""
        if not self._config:
            self.load_config()
        
        # Apply updates
        config_dict = self._config.dict()
        config_dict.update(updates)
        
        # Validate and create new config
        self._config = IntelligentMCPConfig(**config_dict)
        
        return self._config
    
    def get_template_by_name(self, name: str) -> Optional[OperationTemplate]:
        """Get operation template by name."""
        if not self._config:
            self.load_config()
        
        for template in self._config.templates:
            if template.name == name:
                return template
        return None
    
    def add_template(self, template: OperationTemplate) -> bool:
        """Add new operation template."""
        if not self._config:
            self.load_config()
        
        # Check if template already exists
        if self.get_template_by_name(template.name):
            return False
        
        self._config.templates.append(template)
        return True
    
    def remove_template(self, name: str) -> bool:
        """Remove operation template by name."""
        if not self._config:
            self.load_config()
        
        original_count = len(self._config.templates)
        self._config.templates = [t for t in self._config.templates if t.name != name]
        
        return len(self._config.templates) < original_count
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_mapping = {
            'INTELLIGENT_MCP_ENABLED': 'enabled',
            'INTELLIGENT_MCP_DEBUG': 'debug_mode',
            'INTELLIGENT_MCP_ENVIRONMENT': 'environment',
            'INTELLIGENT_MCP_MAX_OPERATIONS': 'performance.max_operations',
            'INTELLIGENT_MCP_TIMEOUT': 'performance.timeout',
            'INTELLIGENT_MCP_COST_LIMIT': 'performance.cost_limit',
            'INTELLIGENT_MCP_CONFIDENCE_THRESHOLD': 'performance.confidence_threshold',
            'INTELLIGENT_MCP_ENABLE_CACHING': 'performance.enable_caching',
            'INTELLIGENT_MCP_CACHE_TTL': 'performance.cache_ttl',
            'INTELLIGENT_MCP_JIRA_ENABLED': 'mcp_service.jira_enabled',
            'INTELLIGENT_MCP_CONFLUENCE_ENABLED': 'mcp_service.confluence_enabled',
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value based on type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                # Set nested configuration value
                self._set_nested_value(config_data, config_path, value)
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_default_templates(self) -> List[OperationTemplate]:
        """Get default operation templates."""
        return [
            OperationTemplate(
                name="issue_deep_analysis",
                intent_types=["issue_analysis"],
                operations=[
                    {
                        "type": "jira_get_issue",
                        "description": "Get detailed issue information",
                        "condition": "has_issue_key"
                    },
                    {
                        "type": "confluence_search",
                        "description": "Search for related documentation",
                        "condition": "has_issues",
                        "depends_on": "jira_get_issue"
                    }
                ],
                priority=9,
                description="Deep analysis of specific Jira issues with related documentation"
            ),
            OperationTemplate(
                name="documentation_search",
                intent_types=["documentation_search"],
                operations=[
                    {
                        "type": "confluence_search",
                        "description": "Search Confluence for documentation",
                        "condition": None
                    },
                    {
                        "type": "jira_search",
                        "description": "Search for related issues",
                        "condition": "has_results",
                        "depends_on": "confluence_search"
                    }
                ],
                priority=8,
                description="Find documentation and related issues"
            ),
            OperationTemplate(
                name="project_overview",
                intent_types=["project_overview", "status_check"],
                operations=[
                    {
                        "type": "jira_search",
                        "description": "Get project issues",
                        "condition": "has_project"
                    },
                    {
                        "type": "confluence_search",
                        "description": "Get project documentation",
                        "condition": "has_project"
                    }
                ],
                priority=7,
                description="Comprehensive project overview with issues and documentation"
            ),
            OperationTemplate(
                name="troubleshooting_search",
                intent_types=["troubleshooting"],
                operations=[
                    {
                        "type": "confluence_search",
                        "description": "Search troubleshooting documentation",
                        "condition": None
                    },
                    {
                        "type": "jira_search",
                        "description": "Search for related issues and solutions",
                        "condition": "has_results"
                    }
                ],
                priority=8,
                description="Find troubleshooting information and related issues"
            ),
            OperationTemplate(
                name="relationship_mapping",
                intent_types=["relationship_mapping"],
                operations=[
                    {
                        "type": "jira_get_issue",
                        "description": "Get primary issue details",
                        "condition": "has_issue_key"
                    },
                    {
                        "type": "jira_search",
                        "description": "Find related issues",
                        "condition": "has_issues"
                    }
                ],
                priority=6,
                description="Map relationships between issues and components"
            ),
            OperationTemplate(
                name="general_search",
                intent_types=["general_search"],
                operations=[
                    {
                        "type": "confluence_search",
                        "description": "General Confluence search",
                        "condition": None
                    },
                    {
                        "type": "jira_search",
                        "description": "General Jira search",
                        "condition": None
                    }
                ],
                priority=5,
                description="General search across both Jira and Confluence"
            )
        ]


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_file)
    
    return _config_manager


def get_intelligent_config() -> IntelligentMCPConfig:
    """Get current intelligent MCP configuration."""
    return get_config_manager().get_config()


def update_intelligent_config(updates: Dict[str, Any]) -> IntelligentMCPConfig:
    """Update intelligent MCP configuration."""
    return get_config_manager().update_config(updates) 