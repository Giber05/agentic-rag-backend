"""
Enhanced Source Retrieval Agent with multi-source support (DB, Jira, Combined)
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .source_retrieval import SourceRetrievalAgent, RetrievedSource, SourceType, RelevanceScore
from ..services.mcp_atlassian_service import MCPAtlassianService

logger = logging.getLogger(__name__)

class EnhancedSourceRetrievalAgent(SourceRetrievalAgent):
    """Enhanced source retrieval agent with multi-source support"""
    
    def __init__(self, agent_id: Optional[str] = None, agent_type: str = "enhanced_source_retrieval", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, agent_type, config)
        
        # Always try to initialize MCP Atlassian service if mcp-config.json exists
        self.mcp_atlassian_service = None
        try:
            import os
            if os.path.exists("mcp-config.json"):
                from ..services.mcp_atlassian_service import MCPAtlassianService
                self.mcp_atlassian_service = MCPAtlassianService()
                logger.info("MCP Atlassian service initialized successfully")
            else:
                logger.info("No mcp-config.json found, Atlassian integration disabled")
        except Exception as e:
            logger.warning(f"Could not initialize MCP Atlassian service: {str(e)}")
        
        self.name = "EnhancedSourceRetrieval"
        
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process source retrieval based on specified source type
        
        Args:
            input_data: Contains query, source, and other parameters
            
        Returns:
            Dict containing retrieved sources and metadata
        """
        query = input_data.get("query", "").strip()
        source = input_data.get("source", "db")
        conversation_history = input_data.get("conversation_history", [])
        user_id = input_data.get("user_id", "anonymous")
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Enhanced source retrieval processing with source: {source}")
        
        try:
            if source == "db":
                return await self._retrieve_from_db(query, conversation_history, user_id)
            elif source == "jira":
                return await self._retrieve_from_jira(query, conversation_history, user_id)
            elif source == "confluence":
                return await self._retrieve_from_confluence(query, conversation_history, user_id)
            elif source == "jira&db":
                return await self._retrieve_from_combined(query, conversation_history, user_id, ["jira", "db"])
            elif source == "confluence&db":
                return await self._retrieve_from_combined(query, conversation_history, user_id, ["confluence", "db"])
            elif source == "jira&confluence":
                return await self._retrieve_from_combined(query, conversation_history, user_id, ["jira", "confluence"])
            elif source == "all":
                return await self._retrieve_from_combined(query, conversation_history, user_id, ["jira", "confluence", "db"])
            elif source == "intelligent":
                return await self._retrieve_from_intelligent_mcp(query, conversation_history, user_id)
            else:
                logger.warning(f"Unknown source type: {source}, defaulting to db")
                return await self._retrieve_from_db(query, conversation_history, user_id)
                
        except Exception as e:
            logger.error(f"Error in enhanced source retrieval: {str(e)}")
            # Fallback to DB retrieval
            logger.info("Falling back to database retrieval")
            return await self._retrieve_from_db(query, conversation_history, user_id)
    
    async def _retrieve_from_db(self, query: str, conversation_history: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Retrieve sources from vector database only"""
        logger.info("Retrieving from database only")
        
        # Use parent class method for DB retrieval
        input_data = {
            "query": query,
            "conversation_history": conversation_history,
            "user_id": user_id,
            "retrieval_config": {"max_results": 10}
        }
        result = await super()._process(input_data)
        
        # Convert parent result format to our format
        sources = result.get('sources', [])
        db_count = len(sources)
        
        formatted_result = {
            'sources': sources,  # Use 'sources' key to match base class interface
            'source_stats': {
                'db_count': db_count,
                'jira_count': 0,
                'total_count': db_count
            },
            'retrieval_metadata': result.get('retrieval_metadata', {}),
            'total_sources': db_count,
            'strategy_used': 'enhanced_db'
        }
        
        logger.info(f"Retrieved {db_count} sources from database")
        return formatted_result
    
    async def _retrieve_from_jira(self, query: str, conversation_history: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Retrieve sources from Jira only"""
        logger.info("Retrieving from Jira only")
        
        if not self.mcp_atlassian_service:
            logger.warning("Atlassian service not available, falling back to database")
            return await self._retrieve_from_db(query, conversation_history, user_id)
        
        try:
            # Initialize the service if not already done
            if not self.mcp_atlassian_service.connection_verified:
                await self.mcp_atlassian_service.initialize()
            
            # Search Jira issues
            jira_result = await self.mcp_atlassian_service.search_jira_issues(query, max_results=10)
            
            logger.info(f"Jira search result: success={jira_result.get('success')}, error={jira_result.get('error')}")
            
            jira_issues = []
            if jira_result.get("success"):
                data = jira_result.get("data")
                # Preferred: service returns parsed dict/list
                if isinstance(data, dict) and "issues" in data:
                    jira_issues = data.get("issues", [])
                elif isinstance(data, list):
                    jira_issues = data
                else:
                    # Backward compatibility: nested content/text JSON
                    nested = data or {}
                    if isinstance(nested, dict) and nested.get("content"):
                        import json as _json
                        text_content = nested["content"][0].get("text", "{}")
                        try:
                            parsed = _json.loads(text_content)
                            jira_issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
                        except Exception:
                            jira_issues = []
            else:
                logger.warning(f"Jira search failed: {jira_result.get('error', 'Unknown error')}")
                jira_issues = []
            
            jira_sources = self._convert_jira_to_sources(jira_issues)
            
            jira_count = len(jira_sources)
            
            result = {
                'sources': [source.to_dict() for source in jira_sources],  # Convert to dict format
                'source_stats': {
                    'db_count': 0,
                    'jira_count': jira_count,
                    'total_count': jira_count
                },
                'retrieval_metadata': {
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'source_type': 'jira',
                    'agent_id': self.agent_id,
                    'strategy': 'enhanced_jira'
                },
                'total_sources': jira_count,
                'strategy_used': 'enhanced_jira'
            }
            
            logger.info(f"Retrieved {jira_count} sources from Jira")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving from Jira: {str(e)}")
            logger.info("Falling back to database retrieval")
            return await self._retrieve_from_db(query, conversation_history, user_id)
    
    async def _retrieve_from_confluence(self, query: str, conversation_history: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Retrieve sources from Confluence only"""
        logger.info("Retrieving from Confluence only")
        
        if not self.mcp_atlassian_service:
            logger.warning("Atlassian service not available, falling back to database")
            return await self._retrieve_from_db(query, conversation_history, user_id)
        
        try:
            # Initialize the service if not already done
            if not self.mcp_atlassian_service.connection_verified:
                await self.mcp_atlassian_service.initialize()
            
            # Search Confluence content
            confluence_result = await self.mcp_atlassian_service.search_confluence_content(query, limit=10)
            
            logger.info(f"Confluence search result: success={confluence_result.get('success')}, error={confluence_result.get('error')}")
            
            confluence_pages = []
            if confluence_result.get("success"):
                data = confluence_result.get("data")
                if isinstance(data, list):
                    confluence_pages = data
                elif isinstance(data, dict) and "results" in data:
                    confluence_pages = data.get("results", [])
                else:
                    # Backward compatibility: nested content/text JSON
                    nested = data or {}
                    if isinstance(nested, dict) and nested.get("content") and not nested.get("isError", False):
                        import json as _json
                        text_content = nested["content"][0].get("text", "[]")
                        try:
                            parsed = _json.loads(text_content)
                            confluence_pages = parsed if isinstance(parsed, list) else []
                        except Exception:
                            confluence_pages = []
                    elif isinstance(nested, dict) and nested.get("isError", False):
                        logger.warning("Confluence MCP returned error payload")
                        confluence_pages = []
            else:
                logger.warning(f"Confluence search failed: {confluence_result.get('error', 'Unknown error')}")
                confluence_pages = []
            
            confluence_sources = self._convert_confluence_to_sources(confluence_pages)
            
            confluence_count = len(confluence_sources)
            
            result = {
                'sources': [source.to_dict() for source in confluence_sources],  # Convert to dict format
                'source_stats': {
                    'db_count': 0,
                    'jira_count': 0,
                    'confluence_count': confluence_count,
                    'total_count': confluence_count
                },
                'retrieval_metadata': {
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'source_type': 'confluence',
                    'agent_id': self.agent_id,
                    'strategy': 'enhanced_confluence'
                },
                'total_sources': confluence_count,
                'strategy_used': 'enhanced_confluence'
            }
            
            logger.info(f"Retrieved {confluence_count} sources from Confluence")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving from Confluence: {str(e)}")
            logger.info("Falling back to database retrieval")
            return await self._retrieve_from_db(query, conversation_history, user_id)
    
    async def _retrieve_from_combined(self, query: str, conversation_history: List[Dict[str, Any]], user_id: str, sources: List[str] = None) -> Dict[str, Any]:
        """Retrieve sources from multiple sources (DB, Jira, Confluence)"""
        if sources is None:
            sources = ["db", "jira"]  # Default for backward compatibility
        
        logger.info(f"Retrieving from combined sources: {sources}")
        
        # Prepare tasks for parallel execution
        tasks = []
        
        # Include DB retrieval if requested
        if "db" in sources:
            db_task = asyncio.create_task(self._retrieve_from_db(query, conversation_history, user_id))
            tasks.append(('db', db_task))
        
        # Include Jira if requested and service is available
        if "jira" in sources and self.mcp_atlassian_service:
            jira_task = asyncio.create_task(self._retrieve_from_jira(query, conversation_history, user_id))
            tasks.append(('jira', jira_task))
        elif "jira" in sources:
            logger.warning("Jira requested but Atlassian service not available")
        
        # Include Confluence if requested and service is available
        if "confluence" in sources and self.mcp_atlassian_service:
            confluence_task = asyncio.create_task(self._retrieve_from_confluence(query, conversation_history, user_id))
            tasks.append(('confluence', confluence_task))
        elif "confluence" in sources:
            logger.warning("Confluence requested but Atlassian service not available")
        
        # Execute tasks in parallel
        try:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            combined_sources = []
            db_count = 0
            jira_count = 0
            confluence_count = 0
            
            for i, (source_type, _) in enumerate(tasks):
                result = results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Error retrieving from {source_type}: {str(result)}")
                    continue
                
                if source_type == 'db':
                    db_sources = result.get('sources', [])
                    combined_sources.extend(db_sources)
                    db_count = len(db_sources)
                elif source_type == 'jira':
                    jira_sources = result.get('sources', [])
                    combined_sources.extend(jira_sources)
                    jira_count = len(jira_sources)
                elif source_type == 'confluence':
                    confluence_sources = result.get('sources', [])
                    combined_sources.extend(confluence_sources)
                    confluence_count = len(confluence_sources)
            
            # Sort combined sources by relevance score if available
            def get_relevance_score(source):
                if isinstance(source, dict):
                    # Handle dict format with nested relevance_score
                    rel_score = source.get('relevance_score', {})
                    if isinstance(rel_score, dict):
                        return rel_score.get('semantic_score', 0.0)
                    return rel_score if isinstance(rel_score, (int, float)) else 0.0
                return 0.0
            
            combined_sources.sort(key=get_relevance_score, reverse=True)
            
            # Limit total results
            max_results = 20
            if len(combined_sources) > max_results:
                combined_sources = combined_sources[:max_results]
                logger.info(f"Limited combined results to {max_results} items")
            
            total_count = len(combined_sources)
            
            result = {
                'sources': combined_sources,  # Already in dict format from sub-methods
                'source_stats': {
                    'db_count': db_count,
                    'jira_count': jira_count,
                    'confluence_count': confluence_count,
                    'total_count': total_count
                },
                'retrieval_metadata': {
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'source_type': 'combined',
                    'agent_id': self.agent_id,
                    'strategy': 'enhanced_combined'
                },
                'total_sources': total_count,
                'strategy_used': 'enhanced_combined'
            }
            
            logger.info(f"Retrieved {total_count} combined sources (DB: {db_count}, Jira: {jira_count}, Confluence: {confluence_count})")
            return result
            
        except Exception as e:
            logger.error(f"Error in combined retrieval: {str(e)}")
            logger.info("Falling back to database retrieval")
            return await self._retrieve_from_db(query, conversation_history, user_id)
    
    def _convert_jira_to_sources(self, jira_issues: List[Dict[str, Any]]) -> List[RetrievedSource]:
        """Convert Jira issues to RetrievedSource objects"""
        sources = []
        
        for issue in jira_issues:
            try:
                # Extract issue information - adapting to MCP response structure
                key = issue.get('key', 'Unknown')
                summary = issue.get('summary', 'No summary')
                description = issue.get('description', 'No description')
                status = issue.get('status', {}).get('name', 'Unknown')
                issue_type = issue.get('issue_type', {}).get('name', 'Unknown')
                priority = issue.get('priority', {}).get('name', 'Unknown')
                assignee = issue.get('assignee', {})
                assignee_name = assignee.get('display_name', 'Unassigned') if assignee else 'Unassigned'
                
                # Create content combining summary and description
                content = f"Summary: {summary}\n\nDescription: {description}"
                
                # Create metadata
                metadata = {
                    'source_type': 'jira',
                    'issue_key': key,
                    'status': status,
                    'issue_type': issue_type,
                    'priority': priority,
                    'assignee': assignee_name,
                    'url': f"https://sprout-id.atlassian.net/browse/{key}"
                }
                
                # Create RetrievedSource object
                relevance_score = RelevanceScore(
                    semantic_score=0.8,  # Default score for Jira issues
                    keyword_score=0.7,
                    context_score=0.6
                )
                
                source = RetrievedSource(
                    source_id=key,
                    content=content,
                    source_type=SourceType.API,  # Jira is an API source
                    relevance_score=relevance_score,
                    metadata=metadata,
                    document_title=f"[{key}] {summary}",
                    url=f"https://sprout-id.atlassian.net/browse/{key}"
                )
                
                sources.append(source)
                
            except Exception as e:
                logger.error(f"Error converting Jira issue to source: {str(e)}")
                continue
        
        return sources 
    
    def _convert_confluence_to_sources(self, confluence_pages: List[Dict[str, Any]]) -> List[RetrievedSource]:
        """Convert Confluence pages to RetrievedSource objects"""
        sources = []
        
        for page in confluence_pages:
            try:
                # Extract page information - adapting to actual MCP response structure
                page_id = page.get('id', 'Unknown')
                title = page.get('title', 'No title')
                
                # Handle content field - can be a dict with 'value' or direct string
                content_raw = page.get('content', {})
                if isinstance(content_raw, dict) and 'value' in content_raw:
                    content = content_raw['value']
                elif isinstance(content_raw, str):
                    content = content_raw
                else:
                    content = 'No content available'
                
                # Extract space information
                space_info = page.get('space', {})
                space_name = space_info.get('name', 'Unknown Space')
                space_key = space_info.get('key', 'Unknown')
                
                # Use the URL provided in the response or construct one
                url = page.get('url', f"https://sprout-id.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}")
                
                # Create metadata
                metadata = {
                    'source_type': 'confluence',
                    'page_id': page_id,
                    'space_name': space_name,
                    'space_key': space_key,
                    'page_type': page.get('type', 'page'),
                    'url': url,
                    'created': page.get('created', ''),
                    'updated': page.get('updated', '')
                }
                
                # Create RetrievedSource object
                relevance_score = RelevanceScore(
                    semantic_score=0.8,  # Default score for Confluence pages
                    keyword_score=0.7,
                    context_score=0.6
                )
                
                source = RetrievedSource(
                    source_id=page_id,
                    content=content,
                    source_type=SourceType.API,  # Confluence is an API source
                    relevance_score=relevance_score,
                    metadata=metadata,
                    document_title=title,
                    url=url
                )
                
                sources.append(source)
                
            except Exception as e:
                logger.error(f"Error converting Confluence page to source: {str(e)}")
                continue
        
        return sources

    async def _retrieve_from_intelligent_mcp(self, query: str, conversation_history: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Retrieve sources using intelligent MCP agent with orchestrated chaining"""
        logger.info("Retrieving from intelligent MCP orchestration")
        
        try:
            # Import and use the intelligent MCP agent
            from .intelligent_mcp_agent import IntelligentMCPAgent
            
            # Create agent instance
            agent = IntelligentMCPAgent()
            await agent.initialize()
            
            # Process with intelligent chaining
            result = await agent.process({
                "query": query,
                "context": {
                    "conversation_history": conversation_history,
                    "user_id": user_id
                },
                "config": {
                    "max_operations": 3,
                    "timeout": 25.0,
                    "confidence_threshold": 0.4,
                    "enable_streaming": False
                }
            })
            
            if result.success:
                chain_result = result.data
                
                # Extract and convert sources from adaptive chain result
                sources = []
                operation_stats = {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "jira_operations": 0,
                    "confluence_operations": 0
                }
                
                if chain_result.get("success"):
                    operation_results = chain_result.get("operations", [])
                    operation_stats["total_operations"] = len(operation_results)
                    
                    for op in operation_results:
                        if not op.get("success"):
                            continue
                        operation_stats["successful_operations"] += 1
                        op_type = (op.get("operation_type") or "").lower()
                        data = op.get("data")
                        # Jira
                        if "jira" in op_type:
                            operation_stats["jira_operations"] += 1
                            issues = []
                            if isinstance(data, dict) and "issues" in data:
                                issues = data.get("issues", [])
                            elif isinstance(data, list):
                                issues = data
                            jira_sources = self._convert_jira_to_sources(issues)
                            sources.extend([s.to_dict() for s in jira_sources])
                        # Confluence
                        elif "confluence" in op_type:
                            operation_stats["confluence_operations"] += 1
                            pages = []
                            if isinstance(data, list):
                                pages = data
                            elif isinstance(data, dict) and "results" in data:
                                pages = data.get("results", [])
                            confluence_sources = self._convert_confluence_to_sources(pages)
                            sources.extend([s.to_dict() for s in confluence_sources])
                
                total_count = len(sources)
                
                result = {
                    'sources': sources,
                    'source_stats': {
                        'db_count': 0,
                        'jira_count': sum(1 for s in sources if s.get('metadata', {}).get('source_type') == 'jira'),
                        'confluence_count': sum(1 for s in sources if s.get('metadata', {}).get('source_type') == 'confluence'),
                        'total_count': total_count
                    },
                    'retrieval_metadata': {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'source_type': 'intelligent_mcp',
                        'agent_id': self.agent_id,
                        'strategy': 'intelligent_orchestration',
                        'operation_stats': operation_stats
                    },
                    'total_sources': total_count,
                    'strategy_used': 'intelligent_mcp'
                }
                
                logger.info(f"Intelligent MCP retrieved {total_count} sources from {operation_stats['total_operations']} operations")
                return result
                
            else:
                logger.warning(f"Intelligent MCP processing failed: {chain_result.get('error', 'Unknown error')}")
                logger.info("Falling back to database retrieval")
                return await self._retrieve_from_db(query, conversation_history, user_id)
                
        except Exception as e:
            logger.error(f"Error in intelligent MCP retrieval: {str(e)}")
            logger.info("Falling back to database retrieval")
            return await self._retrieve_from_db(query, conversation_history, user_id)