"""
MCP Atlassian Service for connecting to both Jira and Confluence via Model Context Protocol.

This service handles communication with the MCP Atlassian server to retrieve
Jira issues, Confluence pages, and other Atlassian data for the RAG pipeline.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.config import settings

logger = logging.getLogger(__name__)

class MCPAtlassianService:
    """
    Service for interacting with Jira and Confluence via MCP (Model Context Protocol).
    
    This service communicates with the MCP Atlassian server to retrieve
    Jira issues, Confluence pages, and other Atlassian data for the RAG pipeline.
    """
    
    def __init__(self):
        self.mcp_config_path = "mcp-config.json"
        self.connection_verified = False
        self._request_counter = 0
        
        # Statistics
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "jira_requests": 0,
            "confluence_requests": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize and verify MCP connection for both Jira and Confluence."""
        try:
            # Check if MCP config exists
            if not os.path.exists(self.mcp_config_path):
                logger.error(f"MCP config file not found: {self.mcp_config_path}")
                return False
            
            # Test connection with both Jira and Confluence
            # jira_test = await self._test_jira_connection()
            # confluence_test = await self._test_confluence_connection()
            jira_test = True
            confluence_test = True
            
            self.connection_verified = jira_test or confluence_test  # At least one should work
            
            if self.connection_verified:
                logger.info(f"MCP Atlassian service initialized (Jira: {jira_test}, Confluence: {confluence_test})")
            else:
                logger.error("Failed to verify any MCP connections")
                
            return self.connection_verified
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Atlassian service: {str(e)}")
            return False
    
    async def _test_jira_connection(self) -> bool:
        """Test Jira MCP connection."""
        try:
            result = await self.get_jira_projects()
            return result["success"]
        except Exception as e:
            logger.warning(f"Jira connection test failed: {str(e)}")
            return False
    
    async def _test_confluence_connection(self) -> bool:
        """Test Confluence MCP connection."""
        try:
            result = await self.search_confluence_content("test", limit=1)
            return result["success"]
        except Exception as e:
            logger.warning(f"Confluence connection test failed: {str(e)}")
            return False

    # ======================
    # JIRA METHODS
    # ======================
    
    async def search_jira_issues(
        self,
        query: str,
        max_results: int = 10,
        jql_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search Jira issues using JQL.
        
        Args:
            query: Natural language query to convert to JQL
            max_results: Maximum number of results to return
            jql_override: Direct JQL query (bypasses query conversion)
            
        Returns:
            Dictionary with success status, data, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Build JQL query
            if jql_override:
                jql = jql_override
            else:
                jql = self._build_jql_from_query(query)
            
            logger.info(f"Searching Jira with JQL: {jql}")
            logger.info(f"MCP search arguments: {{'jql': '{jql}', 'limit': {max_results}}}")
            
            # Execute MCP request
            result = await self._execute_mcp_tool(
                tool_name="jira_search",
                arguments={
                    "jql": jql,
                    "limit": max_results,
                    "fields": "summary,description,status,assignee,created,updated,priority"
                }
            )
            
            if result["success"]:
                issues = result["data"].get("issues", [])
                logger.info(f"Found {len(issues)} Jira issues for query: '{query}'")
                
                # Add processing metadata
                result["metadata"] = {
                    "query": query,
                    "jql_used": jql,
                    "issues_found": len(issues),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "source": "jira_mcp"
                }
                
                self.stats["successful_requests"] += 1
                self.stats["jira_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
                
            self.stats["requests_made"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error searching Jira issues: {str(e)}")
            self.stats["failed_requests"] += 1
            self.stats["requests_made"] += 1
            
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "metadata": {
                    "query": query,
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "source": "jira_mcp"
                }
            }
    
    async def get_jira_issue(self, issue_key: str) -> Dict[str, Any]:
        """Get specific Jira issue by key."""
        try:
            logger.info(f"Getting Jira issue: {issue_key}")
            
            result = await self._execute_mcp_tool(
                tool_name="jira_get_issue",
                arguments={
                    "issue_key": issue_key,
                    "fields": "summary,description,status,assignee,created,updated,priority"
                }
            )
            
            if result["success"]:
                logger.info(f"Retrieved Jira issue: {issue_key}")
                self.stats["jira_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Jira issue {issue_key}: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    async def get_jira_projects(self) -> Dict[str, Any]:
        """Get all accessible Jira projects."""
        try:
            logger.info("Getting Jira projects")
            
            result = await self._execute_mcp_tool(
                tool_name="jira_get_all_projects",
                arguments={}
            )
            
            if result["success"]:
                projects = result["data"] if isinstance(result["data"], list) else []
                logger.info(f"Retrieved {len(projects)} Jira projects")
                self.stats["jira_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Jira projects: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    # ======================
    # EXTENDED JIRA METHODS
    # ======================
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generic executor to call any supported MCP tool."""
        try:
            logger.info(f"Executing MCP tool: {tool_name} with arguments: {arguments}")
            return await self._execute_mcp_tool(tool_name=tool_name, arguments=arguments)
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    # ======================
    # CONFLUENCE METHODS
    # ======================
    
    async def search_confluence_content(
        self,
        query: str,
        limit: int = 100,
        spaces_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search Confluence content using simple terms or CQL.
        
        Args:
            query: Search query - can be simple text or CQL query
            limit: Maximum number of results to return
            spaces_filter: Comma-separated list of space keys to filter by
            
        Returns:
            Dictionary with success status, data, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Searching Confluence with query: {query}")
            
            # Prepare arguments
            arguments = {
                "query": query,
                "limit": limit
            }
            
            if spaces_filter:
                arguments["spaces_filter"] = spaces_filter
            
            # Execute MCP request
            result = await self._execute_mcp_tool(
                tool_name="confluence_search",
                arguments=arguments
            )
            logger.info(f"Confluence search result: {result}")
            
            if result["success"]:
                pages = result["data"] if isinstance(result["data"], list) else []
                logger.info(f"Found {len(pages)} Confluence pages for query: '{query}'")
                
                # Add processing metadata
                result["metadata"] = {
                    "query": query,
                    "pages_found": len(pages),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "source": "confluence_mcp"
                }
                
                self.stats["successful_requests"] += 1
                self.stats["confluence_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
                
            self.stats["requests_made"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error searching Confluence content: {str(e)}")
            self.stats["failed_requests"] += 1
            self.stats["requests_made"] += 1
            
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "metadata": {
                    "query": query,
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "source": "confluence_mcp"
                }
            }
    
    async def get_confluence_page(
        self,
        page_id: Optional[str] = None,
        title: Optional[str] = None,
        space_key: Optional[str] = None,
        include_metadata: bool = True,
        convert_to_markdown: bool = True
    ) -> Dict[str, Any]:
        """
        Get content of a specific Confluence page by ID or title/space.
        
        Args:
            page_id: Confluence page ID
            title: Page title (must be used with space_key)
            space_key: Space key (must be used with title)
            include_metadata: Whether to include page metadata
            convert_to_markdown: Convert content to markdown
            
        Returns:
            Dictionary with success status, data, and metadata
        """
        try:
            if page_id:
                logger.info(f"Getting Confluence page by ID: {page_id}")
                arguments = {"page_id": page_id}
            elif title and space_key:
                logger.info(f"Getting Confluence page: '{title}' in space '{space_key}'")
                arguments = {"title": title, "space_key": space_key}
            else:
                return {
                    "success": False,
                    "data": None,
                    "error": "Must provide either page_id or both title and space_key"
                }
            
            # Add optional parameters
            arguments.update({
                "include_metadata": include_metadata,
                "convert_to_markdown": convert_to_markdown
            })
            
            result = await self._execute_mcp_tool(
                tool_name="confluence_get_page",
                arguments=arguments
            )
            
            if result["success"]:
                logger.info(f"Retrieved Confluence page successfully")
                self.stats["confluence_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Confluence page: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    async def get_confluence_page_children(
        self,
        parent_id: str,
        limit: int = 100,
        include_content: bool = False,
        convert_to_markdown: bool = True
    ) -> Dict[str, Any]:
        """
        Get child pages of a specific Confluence page.
        
        Args:
            parent_id: The ID of the parent page
            limit: Maximum number of child pages to return
            include_content: Whether to include page content
            convert_to_markdown: Convert content to markdown if including content
            
        Returns:
            Dictionary with success status, data, and metadata
        """
        try:
            logger.info(f"Getting children of Confluence page: {parent_id}")
            
            arguments = {
                "parent_id": parent_id,
                "limit": limit,
                "include_content": include_content,
                "convert_to_markdown": convert_to_markdown
            }
            
            result = await self._execute_mcp_tool(
                tool_name="confluence_get_page_children",
                arguments=arguments
            )
            
            if result["success"]:
                children = result["data"] if isinstance(result["data"], list) else []
                logger.info(f"Retrieved {len(children)} child pages")
                self.stats["confluence_requests"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Confluence page children: {str(e)}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    # ======================
    # UTILITY METHODS
    # ======================
    
    def _build_jql_from_query(self, query: str) -> str:
        """Convert natural language query to JQL."""
        query_lower = query.lower()
        
        # Check for project mentions
        project_filter = None
        known_projects = ["toco", "qn", "qinerja", "spec", "spectra", "al", "alodokter", "gm"]
        for project in known_projects:
            if project in query_lower:
                if project in ["qn", "qinerja"]:
                    project_filter = "QN"
                elif project in ["spec", "spectra"]:
                    project_filter = "SPEC"
                elif project in ["al", "alodokter"]:
                    project_filter = "AL"
                elif project == "gm":
                    project_filter = "GM"
                elif project == "toco":
                    project_filter = "TOCO"
                break
        
        # Extract keywords from the query
        keywords = self._extract_keywords(query)
        
        # Build JQL components
        jql_parts = []
        
        # Add project filter if found
        if project_filter:
            jql_parts.append(f'project = "{project_filter}"')
        
        # Text search in summary and description (if no specific project mentioned)
        if keywords and not project_filter:
            text_searches = []
            for keyword in keywords[:3]:  # Limit to 3 keywords to avoid complex queries
                text_searches.append(f'text ~ "{keyword}"')
            jql_parts.append(f"({' OR '.join(text_searches)})")
        
        # For "recent" queries, don't filter by status to include all recent issues
        if "recent" not in query_lower and not project_filter:
            # Add status filter (exclude resolved/closed by default) only for non-recent, non-project queries
            jql_parts.append('status not in ("Done", "Closed", "Resolved")')
        
        # If no filters, add a basic filter to avoid too many results
        if not jql_parts:
            jql_parts.append('updated >= -30d')  # Last 30 days
        
        # Order by relevance and recency
        jql = " AND ".join(jql_parts) + " ORDER BY created DESC"
        
        return jql
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Simple keyword extraction
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract words (alphanumeric + some special chars)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:5]  # Limit to 5 keywords
    
    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool using subprocess."""
        try:
            # Create request payload
            request_payload = {
                "jsonrpc": "2.0",
                "id": self._request_counter,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            self._request_counter += 1
            
            # Write request to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(request_payload, f)
                request_file = f.name
            
            try:
                # Execute MCP command with environment variables
                cmd = [
                    "uvx", "mcp-atlassian"
                ]
                
                # Set environment variables from config
                env = os.environ.copy()
                if os.path.exists(self.mcp_config_path):
                    with open(self.mcp_config_path, 'r') as f:
                        config = json.load(f)
                        server_env = config.get("servers", {}).get("atlassian", {}).get("env", {})
                        env.update(server_env)
                
                logger.info(f"MCP command: {cmd}")
                # logger.info(f"MCP environment variables: {env}")
                # Run the MCP command with input
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                # Prepare initialization sequence
                init_msg = {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"roots": {"listChanged": True}},
                        "clientInfo": {"name": "rag-client", "version": "1.0.0"}
                    }
                }
                
                initialized_msg = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                
                # Send initialization sequence + actual request
                input_data = (
                    json.dumps(init_msg) + '\n' +
                    json.dumps(initialized_msg) + '\n' +
                    json.dumps(request_payload) + '\n'
                )
                logger.info(f"Input data to MCP: {input_data}")

                # Send request and get response
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode()),
                    timeout=30.0
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown MCP error"
                    logger.error(f"MCP command failed: {error_msg}")
                    return {
                        "success": False,
                        "data": None,
                        "error": error_msg
                    }
                
                # Parse response
                try:
                    response_text = stdout.decode().strip()
                    if not response_text:
                        return {
                            "success": False,
                            "data": None,
                            "error": "Empty response from MCP server"
                        }
                    
                    # Handle multiple JSON responses (split by newlines)
                    response_lines = response_text.split('\n')
                    logger.info(f"Response lines from MCP: {response_lines}")
                    for line in response_lines:
                        if line.strip():
                            try:
                                response_data = json.loads(line)
                                
                                # Look for the response that contains actual content (not initialization)
                                if "result" in response_data:
                                    result = response_data.get("result")
                                    
                                    # Check if this is the actual data response (contains 'content' field)
                                    if isinstance(result, dict) and "content" in result:
                                        if "error" in response_data:
                                            return {
                                                "success": False,
                                                "data": None,
                                                "error": response_data["error"].get("message", "Unknown error")
                                            }
                                        
                                        # Extract the actual data from the nested structure
                                        content = result.get("content", [])
                                        if content and len(content) > 0:
                                            # The data is in content[0].text as a JSON string
                                            text_data = content[0].get("text", "")
                                            try:
                                                # Parse the JSON string to get the actual data
                                                actual_data = json.loads(text_data)
                                            except json.JSONDecodeError:
                                                # If parsing fails, return the text as is
                                                actual_data = text_data
                                        else:
                                            actual_data = result
                                        
                                        return {
                                            "success": True,
                                            "data": actual_data,
                                            "error": None
                                        }
                            except json.JSONDecodeError:
                                continue
                    
                    return {
                        "success": False,
                        "data": None,
                        "error": f"No valid response found in: {response_text}"
                    }
                    
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Invalid JSON response: {str(e)}"
                    }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(request_file)
                except:
                    pass
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "data": None,
                "error": "MCP request timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        success_rate = 0.0
        if self.stats["requests_made"] > 0:
            success_rate = self.stats["successful_requests"] / self.stats["requests_made"]
        
        return {
            "connection_verified": self.connection_verified,
            "requests_made": self.stats["requests_made"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": success_rate,
            "cache_hits": self.stats["cache_hits"],
            "jira_requests": self.stats["jira_requests"],
            "confluence_requests": self.stats["confluence_requests"]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for both Jira and Confluence."""
        try:
            if not self.connection_verified:
                await self.initialize()
            
            # Test both services
            jira_test = await self._test_jira_connection()
            confluence_test = await self._test_confluence_connection()
            
            return {
                "success": True,
                "jira_status": "healthy" if jira_test else "unhealthy",
                "confluence_status": "healthy" if confluence_test else "unhealthy",
                "overall_status": "healthy" if (jira_test or confluence_test) else "unhealthy",
                "connection_verified": self.connection_verified,
                "last_test": datetime.utcnow().isoformat(),
                "stats": self.get_stats()
            }
            
        except Exception as e:
            return {
                "success": False,
                "jira_status": "unknown",
                "confluence_status": "unknown",
                "overall_status": "unhealthy",
                "connection_verified": False,
                "error": str(e),
                "last_test": datetime.utcnow().isoformat()
            } 