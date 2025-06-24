"""
Agent registry for managing multiple agents in the system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any
from datetime import datetime

from .base import BaseAgent, AgentState, AgentStatus, AgentMessage

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing multiple agents in the system.
    
    Provides functionality for:
    - Agent registration and discovery
    - Agent lifecycle management
    - Agent communication routing
    - Health monitoring and recovery
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[str, Type[BaseAgent]] = {}
        self._message_handlers: Dict[str, List[callable]] = {}
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Agent registry initialized")
    
    async def start(self) -> None:
        """Start the agent registry."""
        if self._running:
            logger.warning("Agent registry is already running")
            return
        
        logger.info("Starting agent registry")
        self._running = True
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Agent registry started")
    
    async def stop(self) -> None:
        """Stop the agent registry and all registered agents."""
        if not self._running:
            logger.warning("Agent registry is not running")
            return
        
        logger.info("Stopping agent registry")
        self._running = False
        
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop all agents
        await self.stop_all_agents()
        
        logger.info("Agent registry stopped")
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent type for later instantiation.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        self._agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_start: bool = True
    ) -> BaseAgent:
        """
        Create and register a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional specific ID for the agent
            config: Agent configuration
            auto_start: Whether to start the agent automatically
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent type is not registered
            RuntimeError: If agent creation fails
        """
        if agent_type not in self._agent_types:
            raise ValueError(f"Agent type '{agent_type}' is not registered")
        
        try:
            # Create agent instance
            agent_class = self._agent_types[agent_type]
            agent = agent_class(
                agent_id=agent_id,
                agent_type=agent_type,
                config=config
            )
            
            # Register the agent
            self._agents[agent.agent_id] = agent
            
            # Subscribe to agent messages for routing
            agent.subscribe(self._route_message)
            
            # Start the agent if requested
            if auto_start:
                success = await agent.start()
                if not success:
                    # Remove from registry if start failed
                    del self._agents[agent.agent_id]
                    raise RuntimeError(f"Failed to start agent {agent.agent_id}")
            
            logger.info(f"Created and registered agent {agent.agent_id} of type {agent_type}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent of type {agent_type}: {str(e)}")
            raise
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an existing agent instance.
        
        Args:
            agent: Agent instance to register
        """
        self._agents[agent.agent_id] = agent
        agent.subscribe(self._route_message)
        logger.info(f"Registered existing agent {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False
        
        agent = self._agents[agent_id]
        agent.unsubscribe(self._route_message)
        del self._agents[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents of the specified type
        """
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]
    
    def list_agents(self) -> List[BaseAgent]:
        """
        Get all registered agents.
        
        Returns:
            List of all registered agents
        """
        return list(self._agents.values())
    
    def list_agent_states(self) -> List[AgentState]:
        """
        Get the states of all registered agents.
        
        Returns:
            List of agent states
        """
        return [agent.state for agent in self._agents.values()]
    
    async def start_agent(self, agent_id: str) -> bool:
        """
        Start a specific agent.
        
        Args:
            agent_id: ID of the agent to start
            
        Returns:
            True if started successfully, False otherwise
        """
        agent = self.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await agent.start()
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a specific agent.
        
        Args:
            agent_id: ID of the agent to stop
            
        Returns:
            True if stopped successfully, False otherwise
        """
        agent = self.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        return await agent.stop()
    
    async def start_all_agents(self) -> Dict[str, bool]:
        """
        Start all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to start success status
        """
        results = {}
        for agent_id, agent in self._agents.items():
            try:
                results[agent_id] = await agent.start()
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {str(e)}")
                results[agent_id] = False
        
        return results
    
    async def stop_all_agents(self) -> Dict[str, bool]:
        """
        Stop all registered agents.
        
        Returns:
            Dictionary mapping agent IDs to stop success status
        """
        results = {}
        for agent_id, agent in self._agents.items():
            try:
                results[agent_id] = await agent.stop()
            except Exception as e:
                logger.error(f"Failed to stop agent {agent_id}: {str(e)}")
                results[agent_id] = False
        
        return results
    
    async def send_message(
        self,
        message: AgentMessage,
        target_agent_id: Optional[str] = None
    ) -> bool:
        """
        Send a message to a specific agent or broadcast to all agents.
        
        Args:
            message: Message to send
            target_agent_id: Specific agent to send to, or None for broadcast
            
        Returns:
            True if message was sent successfully
        """
        try:
            if target_agent_id:
                # Send to specific agent
                agent = self.get_agent(target_agent_id)
                if not agent:
                    logger.error(f"Target agent {target_agent_id} not found")
                    return False
                
                await agent._message_queue.put(message)
                logger.debug(f"Sent message to agent {target_agent_id}")
                
            else:
                # Broadcast to all agents
                for agent in self._agents.values():
                    await agent._message_queue.put(message)
                
                logger.debug(f"Broadcast message to {len(self._agents)} agents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        status_counts = {}
        for agent in self._agents.values():
            status = agent.state.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        type_counts = {}
        for agent in self._agents.values():
            agent_type = agent.agent_type
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        return {
            "total_agents": len(self._agents),
            "registered_types": len(self._agent_types),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "healthy_agents": len([
                a for a in self._agents.values() if a.is_healthy
            ]),
            "running_agents": len([
                a for a in self._agents.values() if a.is_running
            ])
        }
    
    async def _route_message(self, message: AgentMessage) -> None:
        """
        Route messages between agents.
        
        Args:
            message: Message to route
        """
        try:
            if message.recipient_id:
                # Route to specific recipient
                await self.send_message(message, message.recipient_id)
            else:
                # Handle broadcast or call message handlers
                handlers = self._message_handlers.get(message.message_type, [])
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error routing message: {str(e)}")
    
    def add_message_handler(self, message_type: str, handler: callable) -> None:
        """
        Add a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        
        self._message_handlers[message_type].append(handler)
        logger.info(f"Added message handler for type: {message_type}")
    
    def remove_message_handler(self, message_type: str, handler: callable) -> None:
        """
        Remove a message handler.
        
        Args:
            message_type: Type of message
            handler: Handler function to remove
        """
        if message_type in self._message_handlers:
            handlers = self._message_handlers[message_type]
            if handler in handlers:
                handlers.remove(handler)
                logger.info(f"Removed message handler for type: {message_type}")
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for all agents."""
        try:
            while self._running:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self._health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {str(e)}")
                    await asyncio.sleep(5.0)  # Shorter delay on error
                    
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all agents."""
        unhealthy_agents = []
        
        for agent_id, agent in self._agents.items():
            try:
                if not agent.is_healthy:
                    unhealthy_agents.append(agent_id)
                    logger.warning(f"Agent {agent_id} is unhealthy: {agent.state.status}")
                    
                    # Attempt recovery for error state agents
                    if agent.state.status == AgentStatus.ERROR:
                        logger.info(f"Attempting to restart unhealthy agent {agent_id}")
                        await agent.stop()
                        await asyncio.sleep(1.0)
                        await agent.start()
                        
            except Exception as e:
                logger.error(f"Error checking health of agent {agent_id}: {str(e)}")
        
        if unhealthy_agents:
            logger.warning(f"Found {len(unhealthy_agents)} unhealthy agents: {unhealthy_agents}")
        else:
            logger.debug(f"All {len(self._agents)} agents are healthy") 