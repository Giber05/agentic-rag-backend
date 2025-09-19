"""
MCP Operation Orchestrator - The Brain for Intelligent Operation Chaining

This service analyzes user queries and intelligently plans and executes
chained MCP operations across Jira and Confluence, similar to Cursor's assistant.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

from .mcp_atlassian_service import MCPAtlassianService
from ..core.config import settings
from ..services.openai_service import get_openai_service
from ..core.openai_config import OpenAIModels
from ..core.intelligent_config import get_intelligent_config, OperationTemplate

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of MCP operations available."""
    JIRA_SEARCH = "jira_search"
    JIRA_GET_ISSUE = "jira_get_issue"
    JIRA_GET_PROJECT = "jira_get_project"
    CONFLUENCE_SEARCH = "confluence_search"
    CONFLUENCE_GET_PAGE = "confluence_get_page"
    CONFLUENCE_GET_CHILDREN = "confluence_get_children"
    MCP_TOOL = "mcp_tool"  # Generic tool execution


class QueryIntent(Enum):
    """Detected user query intents."""
    ISSUE_ANALYSIS = "issue_analysis"
    DOCUMENTATION_SEARCH = "documentation_search"
    PROJECT_OVERVIEW = "project_overview"
    TROUBLESHOOTING = "troubleshooting"
    STATUS_CHECK = "status_check"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    GENERAL_SEARCH = "general_search"


@dataclass
class OperationStep:
    """Represents a single operation in a chain."""
    operation_type: OperationType
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[str]] = None  # Step IDs this depends on
    step_id: str = None
    condition: Optional[str] = None  # Condition to execute this step
    
    def __post_init__(self):
        if self.step_id is None:
            self.step_id = f"{self.operation_type.value}_{id(self)}"


@dataclass
class OperationPlan:
    """Represents a complete operation execution plan."""
    query: str
    intent: QueryIntent
    steps: List[OperationStep]
    estimated_cost: float
    estimated_duration: float
    confidence: float
    reasoning: str
    plan_id: str = None
    
    def __post_init__(self):
        if self.plan_id is None:
            self.plan_id = f"plan_{int(datetime.utcnow().timestamp())}"


@dataclass
class OperationResult:
    """Result of a single operation."""
    step_id: str
    operation_type: OperationType
    success: bool
    data: Any
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChainResult:
    """Result of a complete operation chain."""
    plan_id: str
    query: str
    intent: QueryIntent
    success: bool
    results: List[OperationResult]
    final_data: Any
    total_duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MCPOperationOrchestrator:
    """
    Intelligent orchestrator for chained MCP operations.
    
    This is the "brain" that analyzes queries, determines intent,
    and plans optimal operation chains across Jira and Confluence.
    """
    
    def __init__(self, mcp_service: Optional[MCPAtlassianService] = None):
        self.mcp_service = mcp_service or MCPAtlassianService()
        self.config = get_intelligent_config()
        self.operation_templates = self._load_operation_templates()
        self.query_patterns = self._initialize_query_patterns()
        
        # Statistics and monitoring
        self.stats = {
            "plans_created": 0,
            "chains_executed": 0,
            "successful_chains": 0,
            "failed_chains": 0,
            "avg_chain_duration": 0.0,
            "intent_distribution": {},
            "operation_usage": {}
        }
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and its dependencies."""
        try:
            if not await self.mcp_service.initialize():
                logger.error("Failed to initialize MCP service")
                return False
            
            logger.info("MCP Operation Orchestrator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            return False
    
    async def analyze_and_execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_operations: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> ChainResult:
        """
        Main entry point: analyze query and execute operation chain.
        
        Args:
            query: User query to analyze and process
            context: Additional context for planning
            max_operations: Maximum number of operations in chain (uses config default if None)
            timeout: Maximum execution time in seconds (uses config default if None)
            
        Returns:
            ChainResult with execution results
        """
        start_time = datetime.utcnow()
        
        # Use configuration defaults if not provided
        max_operations = max_operations or self.config.performance.max_operations
        timeout = timeout or self.config.performance.timeout
        
        try:
            # Step 1: Analyze query and create plan
            plan = await self.analyze_query(query, context, max_operations)
            logger.info(f"Created operation plan with {len(plan.steps)} steps for query: '{query}'")
            
            # Step 2: Execute the planned operations
            result = await self.execute_operation_chain(plan, timeout)
            
            # Step 3: Update statistics
            self._update_stats(plan, result, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze and execute query '{query}': {str(e)}")
            
            return ChainResult(
                plan_id="error_plan",
                query=query,
                intent=QueryIntent.GENERAL_SEARCH,
                success=False,
                results=[],
                final_data=None,
                total_duration=(datetime.utcnow() - start_time).total_seconds(),
                error=str(e)
            )

    # ==============================
    # ADAPTIVE AI-DRIVEN ORCHESTRATION
    # ==============================
    async def adaptive_execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> ChainResult:
        """Execute an adaptive, AI-driven operation chain.
        The AI selects the next action after each step based on current state.
        """
        start_time = datetime.utcnow()
        context = context or {}
        hard_limit = max_steps or self.config.performance.max_operations
        time_limit = timeout or self.config.performance.timeout
        results: List[OperationResult] = []
        state: Dict[str, Any] = {
            "goal": query,
            "context": context,
            "steps": [],
            "last_result": None,
            "metadata": {},
            "has_jira_results": False,
            "has_confluence_results": False,
        }
        try:
            step_index = 0
            while step_index < hard_limit:
                # Check timeout
                if (datetime.utcnow() - start_time).total_seconds() > time_limit:
                    logger.warning(f"Adaptive chain timeout after {time_limit}s")
                    break

                # Ask AI to decide next action
                decision = await self._decide_next_action(query, state)
                action = (decision or {}).get("action")
                if not action or action == "stop":
                    # Soft enforcement: if we have Jira results but no Confluence yet, try a Confluence search once
                    if state.get("has_jira_results") and not state.get("has_confluence_results") and step_index + 1 < hard_limit:
                        logger.info("AI proposed stop but no Confluence evidence yet; inserting a confluence_search step")
                        action = "confluence_search"
                        kw = self._extract_keywords(query)
                        parameters = {"query": f"text ~ \"{' '.join(kw[:3])}\"", "limit": 100}
                    else:
                        logger.info("AI decided to stop the workflow")
                        break

                parameters = (decision or {}).get("parameters", {})
                description = (decision or {}).get("reason", f"AI selected action {action}")

                # Map action to OperationType and execute
                operation_type = self._map_action_to_operation(action)
                if not operation_type:
                    logger.warning(f"Unknown action from AI: {action}. Stopping.")
                    break

                step_start = datetime.utcnow()
                exec_params = self._resolve_parameters(parameters, {k: v for k, v in state.items() if k != "context"})
                # Normalize params for generic MCP tool actions when AI returns the tool name directly
                if operation_type == OperationType.MCP_TOOL and "tool_name" not in exec_params:
                    exec_params = {"tool_name": action, "arguments": exec_params}
                try:
                    step_exec = await self._execute_single_operation(operation_type, exec_params)
                    step_duration = (datetime.utcnow() - step_start).total_seconds()
                    op_result = OperationResult(
                        step_id=f"{operation_type.value}_{step_index+1}",
                        operation_type=operation_type,
                        success=step_exec.get("success", False),
                        data=step_exec.get("data"),
                        error=step_exec.get("error"),
                        duration=step_duration,
                        metadata=step_exec.get("metadata", {})
                    )
                except Exception as e:
                    step_duration = (datetime.utcnow() - step_start).total_seconds()
                    op_result = OperationResult(
                        step_id=f"{operation_type.value}_{step_index+1}",
                        operation_type=operation_type,
                        success=False,
                        data=None,
                        error=str(e),
                        duration=step_duration,
                        metadata={}
                    )

                # Update state
                results.append(op_result)
                state[op_result.step_id] = op_result.data
                state["last_result"] = {
                    "step_id": op_result.step_id,
                    "operation": op_result.operation_type.value,
                    "success": op_result.success,
                    "error": op_result.error,
                    "metadata": op_result.metadata,
                    "data_preview": str(op_result.data)[:800] if op_result.data is not None else None
                }
                state["steps"].append({
                    "step_id": op_result.step_id,
                    "operation": op_result.operation_type.value,
                    "success": op_result.success,
                    "reason": description
                })

                # Mark evidence flags when data present
                if op_result.success and op_result.data is not None:
                    if "jira" in op_result.operation_type.value:
                        issues = []
                        if isinstance(op_result.data, dict) and "issues" in op_result.data:
                            issues = op_result.data.get("issues", [])
                        elif isinstance(op_result.data, list):
                            issues = op_result.data
                        if len(issues) > 0:
                            state["has_jira_results"] = True
                    if "confluence" in op_result.operation_type.value:
                        pages = []
                        if isinstance(op_result.data, list):
                            pages = op_result.data
                        elif isinstance(op_result.data, dict) and "results" in op_result.data:
                            pages = op_result.data.get("results", [])
                        if len(pages) > 0:
                            state["has_confluence_results"] = True

                step_index += 1

                # Early stop if last step indicates no further action
                if not op_result.success and action in {"jira_search", "confluence_search"}:
                    # If search failed, consider stopping to avoid loops
                    break

            # Synthesize final results
            final_data = self._synthesize_results(
                OperationPlan(query=query, intent=self._detect_intent(query, context), steps=[], estimated_cost=0.0, estimated_duration=0.0, confidence=0.0, reasoning="adaptive"),
                results,
                {k: v for k, v in state.items() if k not in {"context", "metadata"}}
            )
            total_duration = (datetime.utcnow() - start_time).total_seconds()
            chain_result = ChainResult(
                plan_id=f"adaptive_{int(start_time.timestamp())}",
                query=query,
                intent=self._detect_intent(query, context),
                success=len([r for r in results if r.success]) > 0,
                results=results,
                final_data=final_data,
                total_duration=total_duration,
                metadata={
                    "steps_executed": len(results),
                    "plan_confidence": 0.0,
                    "adaptive": True
                }
            )
            self.stats["chains_executed"] += 1
            if chain_result.success:
                self.stats["successful_chains"] += 1
            else:
                self.stats["failed_chains"] += 1
            return chain_result

        except Exception as e:
            logger.error(f"Adaptive execution failed: {str(e)}")
            total_duration = (datetime.utcnow() - start_time).total_seconds()
            return ChainResult(
                plan_id=f"adaptive_{int(start_time.timestamp())}",
                query=query,
                intent=QueryIntent.GENERAL_SEARCH,
                success=False,
                results=results,
                final_data=None,
                total_duration=total_duration,
                error=str(e)
            )

    async def _decide_next_action(self, goal: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to decide the next action based on current state."""
        openai = get_openai_service()
        # Rich action space: allow direct MCP tool invocation for extended coverage
        available_actions = [
            {"name": "jira_search", "params": ["query", "max_results", "jql_override"]},
            {"name": "jira_get_issue", "params": ["issue_key"]},
            {"name": "jira_get_projects", "params": []},
            {"name": "confluence_search", "params": ["query", "limit", "spaces_filter"]},
            {"name": "confluence_get_page", "params": ["page_id", "title", "space_key", "include_metadata", "convert_to_markdown"]},
            {"name": "confluence_get_children", "params": ["parent_id", "limit", "include_content", "convert_to_markdown"]},
            {"name": "mcp_tool", "params": ["tool_name", "arguments"]},
            {"name": "stop", "params": []}
        ]
        allowed_tool_names = [
            "jira_get_user_profile","jira_get_issue","jira_search","jira_search_fields",
            "jira_get_project_issues","jira_get_transitions","jira_get_worklog","jira_download_attachments",
            "jira_get_agile_boards","jira_get_board_issues","jira_get_sprints_from_board","jira_get_sprint_issues",
            "jira_get_link_types","jira_create_issue","jira_batch_create_issues","jira_batch_get_changelogs",
            "jira_update_issue","jira_delete_issue","jira_add_comment","jira_add_worklog","jira_link_to_epic",
            "jira_create_issue_link","jira_create_remote_issue_link","jira_remove_issue_link","jira_transition_issue",
            "jira_create_sprint","jira_update_sprint","jira_get_project_versions","jira_get_all_projects",
            "jira_create_version","jira_batch_create_versions",
            "confluence_search","confluence_get_page","confluence_get_page_children","confluence_get_comments",
            "confluence_get_labels","confluence_add_label","confluence_create_page","confluence_update_page",
            "confluence_delete_page","confluence_add_comment","confluence_search_user"
        ]
        system_prompt = (
            "You are an orchestration planner for Jira and Confluence operations. "
            "Choose the best next action to achieve the user's goal. "
            "Only respond with a compact JSON object using keys: action, parameters, reason. "
            "Valid actions: " + ", ".join([a["name"] for a in available_actions]) + ". "
            "When using action 'mcp_tool', set parameters.tool_name to one of: " + ", ".join(allowed_tool_names) + ". "
            "Place all tool parameters under parameters.arguments. "
            "Prefer minimal, high-signal steps. Stop when goal seems achieved or no progress can be made."
        )
        state_compact = {
            "goal": goal,
            "last_result": state.get("last_result"),
            "recent_steps": state.get("steps", [])[-5:],
            "context_keys": list((state.get("context") or {}).keys()),
            "has_jira_results": state.get("has_jira_results", False),
            "has_confluence_results": state.get("has_confluence_results", False)
        }
        user_prompt = (
            "STATE: " + json.dumps(state_compact, ensure_ascii=False) + "\n" +
            "Guidelines: Prefer to obtain corroborating evidence from both Jira and Confluence before stopping when relevant. "
            "If Jira results exist but no Confluence evidence yet, perform a targeted confluence_search using salient keywords.\n" +
            "Return JSON only. Example: {\"action\": \"jira_search\", \"parameters\": {\"query\": \"text ~ 'new bug'\", \"max_results\": 5}, \"reason\": \"find related issues\"}"
        )
        try:
            resp = await openai.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=OpenAIModels.GPT_4_1_MINI,
                max_tokens=300,
                temperature=0.2,
                stream=False,
                use_cache=False
            )
            content = resp.choices[0].message.content or "{}"
            # Extract JSON
            try:
                content_str = content.strip()
                if content_str.startswith("```"):
                    # strip fences
                    inner = content_str.strip("`")
                    if "\n" in inner:
                        inner = inner.split("\n", 1)[1]
                    content_str = inner.strip()
                decision = json.loads(content_str)
                # Basic validation
                if isinstance(decision, dict) and "action" in decision:
                    return decision
            except Exception:
                pass
            return {"action": "stop", "parameters": {}, "reason": "invalid decision"}
        except Exception as e:
            logger.error(f"Decision agent failed: {str(e)}")
            return {"action": "stop", "parameters": {}, "reason": "decision error"}

    def _map_action_to_operation(self, action: str) -> Optional[OperationType]:
        mapping = {
            "jira_search": OperationType.JIRA_SEARCH,
            "jira_get_issue": OperationType.JIRA_GET_ISSUE,
            "jira_get_projects": OperationType.JIRA_GET_PROJECT,
            "confluence_search": OperationType.CONFLUENCE_SEARCH,
            "confluence_get_page": OperationType.CONFLUENCE_GET_PAGE,
            "confluence_get_children": OperationType.CONFLUENCE_GET_CHILDREN,
            "mcp_tool": OperationType.MCP_TOOL,
        }
        if action in mapping:
            return mapping[action]
        # Fallback: any unrecognized Jira/Confluence tool name should be routed via generic MCP_TOOL
        if action.startswith("jira_") or action.startswith("confluence_"):
            return OperationType.MCP_TOOL
        return None
    
    async def analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_operations: int = 5
    ) -> OperationPlan:
        """
        Analyze user query and create an intelligent operation plan.
        
        Args:
            query: User query to analyze
            context: Additional context for planning
            max_operations: Maximum operations in the plan
            
        Returns:
            OperationPlan with steps to execute
        """
        try:
            # Step 1: Detect query intent
            intent = self._detect_intent(query, context)
            logger.info(f"Detected intent: {intent.value} for query: '{query}'")
            
            # Step 2: Extract entities and keywords
            entities = self._extract_entities(query)
            keywords = self._extract_keywords(query)
            
            # Step 3: Select operation template
            template = self._select_template(intent, entities, keywords)
            
            # Step 4: Generate operation steps
            steps = self._generate_steps(query, intent, template, entities, keywords, max_operations)
            
            # Step 5: Estimate cost and duration
            estimated_cost = self._estimate_cost(steps)
            estimated_duration = self._estimate_duration(steps)
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(query, intent, steps, entities)
            
            plan = OperationPlan(
                query=query,
                intent=intent,
                steps=steps,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration,
                confidence=confidence,
                reasoning=self._generate_reasoning(intent, steps, entities)
            )
            
            self.stats["plans_created"] += 1
            return plan
            
        except Exception as e:
            logger.error(f"Failed to analyze query '{query}': {str(e)}")
            raise
    
    async def execute_operation_chain(
        self,
        plan: OperationPlan,
        timeout: float = 30.0
    ) -> ChainResult:
        """
        Execute a planned operation chain with dependency management.
        
        Args:
            plan: OperationPlan to execute
            timeout: Maximum execution time
            
        Returns:
            ChainResult with execution results
        """
        start_time = datetime.utcnow()
        results = []
        context_data = {}  # Shared data between operations
        
        try:
            logger.info(f"Executing operation chain for plan: {plan.plan_id}")
            
            # Execute steps with dependency resolution
            for step in plan.steps:
                step_start = datetime.utcnow()
                
                # Check if step should be executed based on condition
                if step.condition and not self._evaluate_condition(step.condition, context_data):
                    logger.info(f"Skipping step {step.step_id} due to condition: {step.condition}")
                    continue
                
                # Wait for dependencies
                if step.depends_on:
                    if not self._check_dependencies(step.depends_on, results):
                        logger.warning(f"Dependencies not met for step {step.step_id}")
                        continue
                
                # Execute the operation
                try:
                    logger.info(f"Executing step: {step.step_id} - {step.description}")
                    
                    # Resolve parameters with context data
                    resolved_params = self._resolve_parameters(step.parameters, context_data)
                    
                    # Execute the specific operation
                    step_result = await self._execute_single_operation(
                        step.operation_type,
                        resolved_params
                    )
                    
                    step_duration = (datetime.utcnow() - step_start).total_seconds()
                    
                    result = OperationResult(
                        step_id=step.step_id,
                        operation_type=step.operation_type,
                        success=step_result.get("success", False),
                        data=step_result.get("data"),
                        error=step_result.get("error"),
                        duration=step_duration,
                        metadata=step_result.get("metadata", {})
                    )
                    
                    results.append(result)
                    
                    # Update context with result data
                    if result.success and result.data:
                        context_data[step.step_id] = result.data
                        context_data[f"{step.operation_type.value}_result"] = result.data
                    
                    logger.info(f"Step {step.step_id} completed in {step_duration:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {str(e)}")
                    
                    step_duration = (datetime.utcnow() - step_start).total_seconds()
                    results.append(OperationResult(
                        step_id=step.step_id,
                        operation_type=step.operation_type,
                        success=False,
                        data=None,
                        error=str(e),
                        duration=step_duration
                    ))
                
                # Check timeout
                if (datetime.utcnow() - start_time).total_seconds() > timeout:
                    logger.warning(f"Operation chain timeout after {timeout}s")
                    break
            
            # Synthesize final results
            final_data = self._synthesize_results(plan, results, context_data)
            total_duration = (datetime.utcnow() - start_time).total_seconds()
            
            chain_result = ChainResult(
                plan_id=plan.plan_id,
                query=plan.query,
                intent=plan.intent,
                success=len([r for r in results if r.success]) > 0,
                results=results,
                final_data=final_data,
                total_duration=total_duration,
                metadata={
                    "steps_executed": len(results),
                    "steps_successful": len([r for r in results if r.success]),
                    "context_data_keys": list(context_data.keys()),
                    "plan_confidence": plan.confidence
                }
            )
            
            self.stats["chains_executed"] += 1
            if chain_result.success:
                self.stats["successful_chains"] += 1
            else:
                self.stats["failed_chains"] += 1
            
            logger.info(f"Chain execution completed in {total_duration:.2f}s")
            return chain_result
            
        except Exception as e:
            logger.error(f"Failed to execute operation chain: {str(e)}")
            
            total_duration = (datetime.utcnow() - start_time).total_seconds()
            return ChainResult(
                plan_id=plan.plan_id,
                query=plan.query,
                intent=plan.intent,
                success=False,
                results=results,
                final_data=None,
                total_duration=total_duration,
                error=str(e)
            )
    
    # ======================
    # PRIVATE HELPER METHODS
    # ======================
    
    def _detect_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryIntent:
        """Detect the intent of the user query."""
        query_lower = query.lower()
        
        # Issue analysis patterns
        if any(pattern in query_lower for pattern in [
            "issue", "bug", "ticket", "jira", "key:", "project-",
            "assigned to", "status", "priority", "resolve"
        ]):
            return QueryIntent.ISSUE_ANALYSIS
        
        # Documentation search patterns
        if any(pattern in query_lower for pattern in [
            "documentation", "docs", "confluence", "wiki", "page",
            "how to", "guide", "tutorial", "manual", "readme"
        ]):
            return QueryIntent.DOCUMENTATION_SEARCH
        
        # Project overview patterns
        if any(pattern in query_lower for pattern in [
            "project overview", "project status", "dashboard",
            "summary", "progress", "milestone", "release"
        ]):
            return QueryIntent.PROJECT_OVERVIEW
        
        # Troubleshooting patterns
        if any(pattern in query_lower for pattern in [
            "error", "problem", "troubleshoot", "debug", "fix",
            "not working", "failed", "broken", "issue with"
        ]):
            return QueryIntent.TROUBLESHOOTING
        
        # Status check patterns
        if any(pattern in query_lower for pattern in [
            "status of", "what's the status", "current state",
            "progress on", "update on", "latest on"
        ]):
            return QueryIntent.STATUS_CHECK
        
        # Relationship mapping patterns
        if any(pattern in query_lower for pattern in [
            "related to", "linked", "depends on", "blocks",
            "relationship", "connection", "associated"
        ]):
            return QueryIntent.RELATIONSHIP_MAPPING
        
        return QueryIntent.GENERAL_SEARCH
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities like issue keys, project names, etc."""
        entities = {
            "issue_keys": [],
            "project_keys": [],
            "usernames": [],
            "dates": [],
            "statuses": []
        }
        
        # Extract Jira issue keys (e.g., PROJ-123, DEV-456)
        issue_pattern = r'\b[A-Z]{2,}-\d+\b'
        entities["issue_keys"] = re.findall(issue_pattern, query)
        
        # Extract project keys (uppercase words before hyphens)
        if entities["issue_keys"]:
            entities["project_keys"] = list(set([key.split('-')[0] for key in entities["issue_keys"]]))
        
        # Extract potential usernames (@username or "assigned to username")
        username_pattern = r'@(\w+)|assigned to (\w+)|by (\w+)'
        matches = re.findall(username_pattern, query, re.IGNORECASE)
        entities["usernames"] = [match for group in matches for match in group if match]
        
        # Extract status mentions
        status_keywords = ["open", "closed", "in progress", "done", "todo", "resolved", "blocked"]
        entities["statuses"] = [status for status in status_keywords if status in query.lower()]
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "will", "would", "could", "should", "may", "might"
        }
        
        words = re.findall(r'\b\w{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:100]  # Limit to top 10 keywords
    
    def _select_template(
        self,
        intent: QueryIntent,
        entities: Dict[str, List[str]],
        keywords: List[str]
    ) -> str:
        """Select the most appropriate operation template."""
        # Return template based on intent and available entities
        if intent == QueryIntent.ISSUE_ANALYSIS and entities["issue_keys"]:
            return "issue_deep_analysis"
        elif intent == QueryIntent.DOCUMENTATION_SEARCH:
            return "documentation_search"
        elif intent == QueryIntent.PROJECT_OVERVIEW and entities["project_keys"]:
            return "project_overview"
        elif intent == QueryIntent.TROUBLESHOOTING:
            return "troubleshooting_search"
        else:
            return "general_search"
    
    def _generate_steps(
        self,
        query: str,
        intent: QueryIntent,
        template: str,
        entities: Dict[str, List[str]],
        keywords: List[str],
        max_operations: int
    ) -> List[OperationStep]:
        """Generate operation steps based on template and context."""
        steps = []
        
        if template == "issue_deep_analysis":
            # Get issue details first
            for issue_key in entities["issue_keys"][:100]:  # Limit to 2 issues
                steps.append(OperationStep(
                    operation_type=OperationType.JIRA_GET_ISSUE,
                    parameters={"issue_key": issue_key},
                    description=f"Get details for issue {issue_key}"
                ))
            
            # Search for related documentation
            steps.append(OperationStep(
                operation_type=OperationType.CONFLUENCE_SEARCH,
                parameters={
                    "query": f"text ~ \"{' '.join(keywords[:3])}\"",
                    "limit": 100
                },
                description="Search for related documentation",
                condition="has_issues"
            ))
        
        elif template == "documentation_search":
            # Search Confluence first
            steps.append(OperationStep(
                operation_type=OperationType.CONFLUENCE_SEARCH,
                parameters={
                    "query": f"text ~ \"{' '.join(keywords[:5])}\"",
                    "limit": 100
                },
                description="Search Confluence documentation"
            ))
            
            # Search Jira for related issues
            if keywords:
                jql = f"text ~ \"{' '.join(keywords[:3])}\""
                steps.append(OperationStep(
                    operation_type=OperationType.JIRA_SEARCH,
                    parameters={"jql_override": jql, "max_results": 100},
                    description="Search for related Jira issues"
                ))
        
        elif template == "project_overview":
            # Get project info
            for project_key in entities["project_keys"][:1]:  # One project at a time
                steps.append(OperationStep(
                    operation_type=OperationType.JIRA_SEARCH,
                    parameters={
                        "jql_override": f"project = {project_key} ORDER BY updated DESC",
                        "max_results": 100
                    },
                    description=f"Get recent issues for project {project_key}"
                ))
                
                # Search for project documentation
                steps.append(OperationStep(
                    operation_type=OperationType.CONFLUENCE_SEARCH,
                    parameters={
                        "query": f"text ~ \"{project_key}\"",
                        "limit": 100
                    },
                    description=f"Search for {project_key} documentation"
                ))
        
        else:  # general_search
            # Balanced search across both platforms
            if keywords:
                confluence_query = f"text ~ \"{' '.join(keywords[:3])}\""
                steps.append(OperationStep(
                    operation_type=OperationType.CONFLUENCE_SEARCH,
                    parameters={"query": confluence_query, "limit": 100},
                    description="Search Confluence for relevant content"
                ))
                
                jira_query = f"text ~ \"{' '.join(keywords[:3])}\""
                steps.append(OperationStep(
                    operation_type=OperationType.JIRA_SEARCH,
                    parameters={"jql_override": jira_query, "max_results": 100},
                    description="Search Jira for relevant issues"
                ))
        
        return steps[:max_operations]
    
    def _estimate_cost(self, steps: List[OperationStep]) -> float:
        """Estimate the cost of executing the operation chain."""
        # Base costs per operation type
        operation_costs = {
            OperationType.JIRA_SEARCH: 0.1,
            OperationType.JIRA_GET_ISSUE: 0.05,
            OperationType.JIRA_GET_PROJECT: 0.05,
            OperationType.CONFLUENCE_SEARCH: 0.1,
            OperationType.CONFLUENCE_GET_PAGE: 0.05,
            OperationType.CONFLUENCE_GET_CHILDREN: 0.08
        }
        
        total_cost = sum(operation_costs.get(step.operation_type, 0.1) for step in steps)
        return round(total_cost, 3)
    
    def _estimate_duration(self, steps: List[OperationStep]) -> float:
        """Estimate the duration of executing the operation chain."""
        # Base durations per operation type (in seconds)
        operation_durations = {
            OperationType.JIRA_SEARCH: 2.0,
            OperationType.JIRA_GET_ISSUE: 1.0,
            OperationType.JIRA_GET_PROJECT: 1.5,
            OperationType.CONFLUENCE_SEARCH: 2.5,
            OperationType.CONFLUENCE_GET_PAGE: 1.0,
            OperationType.CONFLUENCE_GET_CHILDREN: 1.5
        }
        
        # Assume some operations can run in parallel
        sequential_duration = sum(operation_durations.get(step.operation_type, 2.0) for step in steps)
        parallel_factor = 0.7  # 30% time savings from parallelization
        
        return round(sequential_duration * parallel_factor, 2)
    
    def _calculate_confidence(
        self,
        query: str,
        intent: QueryIntent,
        steps: List[OperationStep],
        entities: Dict[str, List[str]]
    ) -> float:
        """Calculate confidence score for the operation plan."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for specific entities
        if entities["issue_keys"]:
            confidence += 0.2
        if entities["project_keys"]:
            confidence += 0.1
        
        # Higher confidence for clear intent patterns
        if intent != QueryIntent.GENERAL_SEARCH:
            confidence += 0.2
        
        # Higher confidence for reasonable number of steps
        if 1 <= len(steps) <= 3:
            confidence += 0.1
        elif len(steps) > 5:
            confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        intent: QueryIntent,
        steps: List[OperationStep],
        entities: Dict[str, List[str]]
    ) -> str:
        """Generate human-readable reasoning for the operation plan."""
        reasoning_parts = [
            f"Detected intent: {intent.value.replace('_', ' ')}"
        ]
        
        if entities["issue_keys"]:
            reasoning_parts.append(f"Found specific issues: {', '.join(entities['issue_keys'])}")
        
        if entities["project_keys"]:
            reasoning_parts.append(f"Targeting projects: {', '.join(entities['project_keys'])}")
        
        operation_summary = f"Planning {len(steps)} operations: " + ", ".join([
            f"{step.operation_type.value}" for step in steps
        ])
        reasoning_parts.append(operation_summary)
        
        return "; ".join(reasoning_parts)
    
    async def _execute_single_operation(
        self,
        operation_type: OperationType,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single MCP operation."""
        try:
            if operation_type == OperationType.JIRA_SEARCH:
                return await self.mcp_service.search_jira_issues(**parameters)
            elif operation_type == OperationType.JIRA_GET_ISSUE:
                return await self.mcp_service.get_jira_issue(**parameters)
            elif operation_type == OperationType.JIRA_GET_PROJECT:
                return await self.mcp_service.get_jira_projects()
            elif operation_type == OperationType.CONFLUENCE_SEARCH:
                return await self.mcp_service.search_confluence_content(**parameters)
            elif operation_type == OperationType.CONFLUENCE_GET_PAGE:
                return await self.mcp_service.get_confluence_page(**parameters)
            elif operation_type == OperationType.CONFLUENCE_GET_CHILDREN:
                return await self.mcp_service.get_confluence_page_children(**parameters)
            elif operation_type == OperationType.MCP_TOOL:
                # Generic passthrough: expects {'tool_name': str, 'arguments': {..}}
                tool_name: str = parameters.get("tool_name")
                arguments: Dict[str, Any] = parameters.get("arguments", {})
                if not tool_name:
                    return {"success": False, "error": "Missing tool_name"}
                return await self.mcp_service.execute_tool(tool_name, arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation type: {operation_type}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter references using context data."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract context reference
                ref = value[2:-1]
                if ref in context_data:
                    resolved[key] = context_data[ref]
                else:
                    logger.warning(f"Context reference not found: {ref}")
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    def _check_dependencies(self, depends_on: List[str], results: List[OperationResult]) -> bool:
        """Check if all dependencies are satisfied."""
        completed_steps = {result.step_id for result in results if result.success}
        return all(dep in completed_steps for dep in depends_on)
    
    def _evaluate_condition(self, condition: str, context_data: Dict[str, Any]) -> bool:
        """Evaluate a condition string against context data."""
        # Simple condition evaluation
        if condition == "has_issues":
            return any("issues" in str(value) for value in context_data.values())
        
        # Default to True for unknown conditions
        return True
    
    def _synthesize_results(
        self,
        plan: OperationPlan,
        results: List[OperationResult],
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final results from all operation results."""
        synthesis = {
            "query": plan.query,
            "intent": plan.intent.value,
            "total_operations": len(results),
            "successful_operations": len([r for r in results if r.success]),
            "data_sources": [],
            "key_findings": []
        }
        
        # Aggregate data by source
        jira_data = []
        confluence_data = []
        
        for result in results:
            if result.success and result.data:
                if "jira" in result.operation_type.value:
                    jira_data.append(result.data)
                    synthesis["data_sources"].append("jira")
                elif "confluence" in result.operation_type.value:
                    confluence_data.append(result.data)
                    synthesis["data_sources"].append("confluence")
        
        # Add aggregated data
        if jira_data:
            synthesis["jira_results"] = jira_data
            synthesis["key_findings"].append(f"Found {len(jira_data)} Jira result(s)")
        
        if confluence_data:
            synthesis["confluence_results"] = confluence_data
            synthesis["key_findings"].append(f"Found {len(confluence_data)} Confluence result(s)")
        
        # Remove duplicates from data sources
        synthesis["data_sources"] = list(set(synthesis["data_sources"]))
        
        return synthesis
    
    def _update_stats(
        self,
        plan: OperationPlan,
        result: ChainResult,
        start_time: datetime
    ):
        """Update orchestrator statistics."""
        # Update intent distribution
        intent_str = plan.intent.value
        self.stats["intent_distribution"][intent_str] = self.stats["intent_distribution"].get(intent_str, 0) + 1
        
        # Update operation usage
        for step in plan.steps:
            op_str = step.operation_type.value
            self.stats["operation_usage"][op_str] = self.stats["operation_usage"].get(op_str, 0) + 1
        
        # Update average duration
        current_avg = self.stats["avg_chain_duration"]
        total_chains = self.stats["chains_executed"]
        
        if total_chains > 0:
            self.stats["avg_chain_duration"] = (
                (current_avg * (total_chains - 1) + result.total_duration) / total_chains
            )
        else:
            self.stats["avg_chain_duration"] = result.total_duration
    
    def _load_operation_templates(self) -> Dict[str, Any]:
        """Load operation templates from configuration."""
        templates = {}
        
        # Load templates from configuration
        for template in self.config.templates:
            if template.enabled and template.name not in templates:
                templates[template.name] = {
                    "intent_types": template.intent_types,
                    "operations": template.operations,
                    "conditions": template.conditions,
                    "priority": template.priority,
                    "description": template.description
                }
        
        return templates
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for query analysis."""
        return {
            "issue_patterns": [
                r"\b[A-Z]{2,}-\d+\b",  # Issue keys
                r"issue\s+#?\d+",       # Issue numbers
                r"ticket\s+#?\d+"       # Ticket numbers
            ],
            "project_patterns": [
                r"project\s+([A-Z]+)",
                r"in\s+([A-Z]{2,})\s+project"
            ],
            "user_patterns": [
                r"@(\w+)",
                r"assigned\s+to\s+(\w+)",
                r"created\s+by\s+(\w+)"
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        stats = self.stats.copy()
        
        # Add calculated metrics
        if stats["chains_executed"] > 0:
            stats["success_rate"] = stats["successful_chains"] / stats["chains_executed"]
            stats["failure_rate"] = stats["failed_chains"] / stats["chains_executed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the orchestrator."""
        try:
            # Check MCP service health
            mcp_health = await self.mcp_service.health_check()
            
            return {
                "healthy": mcp_health.get("success", False),
                "mcp_service": mcp_health,
                "orchestrator_stats": self.get_statistics(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            } 