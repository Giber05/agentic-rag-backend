"""
Base agent class and core interfaces for the agent framework.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentState(BaseModel):
    """Agent state model."""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current agent status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Agent creation time")
    started_at: Optional[datetime] = Field(None, description="Agent start time")
    stopped_at: Optional[datetime] = Field(None, description="Agent stop time")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    
    class Config:
        use_enum_values = True


class AgentResult(BaseModel):
    """Agent processing result model."""
    agent_id: str = Field(..., description="Agent that produced the result")
    agent_type: str = Field(..., description="Type of agent")
    success: bool = Field(..., description="Whether the operation was successful")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Result timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")


class AgentMessage(BaseModel):
    """Message for agent communication."""
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique message ID")
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_id: Optional[str] = Field(None, description="Recipient agent ID (None for broadcast)")
    message_type: str = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request-response")


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the RAG system.
    
    Provides common functionality for:
    - Agent lifecycle management
    - State tracking and persistence
    - Communication protocols
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/name of the agent
            config: Agent configuration dictionary
        """
        self.agent_id = agent_id or str(uuid4())
        self.agent_type = agent_type or self.__class__.__name__
        self.config = config or {}
        
        # Initialize state
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            metadata={"config": self.config}
        )
        
        # Communication
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: List[callable] = []
        
        # Lifecycle management
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self._start_time: Optional[float] = None
        self._total_processing_time = 0.0
        self._operation_count = 0
        
        logger.info(f"Initialized agent {self.agent_id} of type {self.agent_type}")
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._running and self.state.status == AgentStatus.RUNNING
    
    @property
    def is_healthy(self) -> bool:
        """Check if the agent is in a healthy state."""
        return self.state.status not in [AgentStatus.ERROR, AgentStatus.STOPPED]
    
    async def start(self) -> bool:
        """
        Start the agent.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._running:
                logger.warning(f"Agent {self.agent_id} is already running")
                return True
            
            logger.info(f"Starting agent {self.agent_id}")
            self.state.status = AgentStatus.STARTING
            self.state.started_at = datetime.utcnow()
            
            # Perform agent-specific initialization
            await self._on_start()
            
            # Start the main agent loop
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            
            self.state.status = AgentStatus.RUNNING
            self.state.last_activity = datetime.utcnow()
            
            logger.info(f"Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent {self.agent_id}: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return False
    
    async def stop(self) -> bool:
        """
        Stop the agent gracefully.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self._running:
                logger.warning(f"Agent {self.agent_id} is not running")
                return True
            
            logger.info(f"Stopping agent {self.agent_id}")
            self.state.status = AgentStatus.STOPPING
            
            # Signal shutdown
            self._running = False
            self._shutdown_event.set()
            
            # Wait for the main loop to finish
            if self._task:
                try:
                    await asyncio.wait_for(self._task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {self.agent_id} did not stop gracefully, cancelling")
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        pass
            
            # Perform agent-specific cleanup
            await self._on_stop()
            
            self.state.status = AgentStatus.STOPPED
            self.state.stopped_at = datetime.utcnow()
            
            logger.info(f"Agent {self.agent_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent {self.agent_id}: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return False
    
    async def pause(self) -> bool:
        """
        Pause the agent temporarily.
        
        Returns:
            True if paused successfully, False otherwise
        """
        try:
            if self.state.status != AgentStatus.RUNNING:
                logger.warning(f"Agent {self.agent_id} is not running, cannot pause")
                return False
            
            logger.info(f"Pausing agent {self.agent_id}")
            self.state.status = AgentStatus.PAUSED
            await self._on_pause()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause agent {self.agent_id}: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return False
    
    async def resume(self) -> bool:
        """
        Resume the agent from paused state.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        try:
            if self.state.status != AgentStatus.PAUSED:
                logger.warning(f"Agent {self.agent_id} is not paused, cannot resume")
                return False
            
            logger.info(f"Resuming agent {self.agent_id}")
            await self._on_resume()
            self.state.status = AgentStatus.RUNNING
            self.state.last_activity = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume agent {self.agent_id}: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return False
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Process input data and return result.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            AgentResult with processing outcome
        """
        start_time = time.time()
        
        try:
            if not self.is_healthy:
                raise RuntimeError(f"Agent {self.agent_id} is not in a healthy state")
            
            # Update activity timestamp
            self.state.last_activity = datetime.utcnow()
            
            # Perform agent-specific processing
            result_data = await self._process(input_data, **kwargs)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update performance metrics
            self._total_processing_time += processing_time_ms
            self._operation_count += 1
            
            # Create successful result
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                data=result_data,
                processing_time_ms=processing_time_ms,
                metadata={
                    "input_size": len(str(input_data)),
                    "operation_count": self._operation_count
                }
            )
            
            logger.debug(f"Agent {self.agent_id} processed request in {processing_time_ms}ms")
            return result
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error(f"Agent {self.agent_id} processing failed: {str(e)}")
            
            # Create error result
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
            
            return result
    
    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to other agents.
        
        Args:
            message: Message to send
        """
        # Notify all subscribers
        for subscriber in self._subscribers:
            try:
                await subscriber(message)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {str(e)}")
    
    def subscribe(self, callback: callable) -> None:
        """
        Subscribe to messages from this agent.
        
        Args:
            callback: Callback function to handle messages
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: callable) -> None:
        """
        Unsubscribe from messages from this agent.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        uptime = 0
        if self.state.started_at:
            uptime = (datetime.utcnow() - self.state.started_at).total_seconds()
        
        avg_processing_time = 0
        if self._operation_count > 0:
            avg_processing_time = self._total_processing_time / self._operation_count
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.state.status,
            "uptime_seconds": uptime,
            "total_operations": self._operation_count,
            "total_processing_time_ms": self._total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "last_activity": self.state.last_activity,
            "is_healthy": self.is_healthy
        }
    
    async def _run_loop(self) -> None:
        """Main agent execution loop."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    # Check for messages
                    try:
                        message = await asyncio.wait_for(
                            self._message_queue.get(), 
                            timeout=1.0
                        )
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        pass  # No message, continue loop
                    
                    # Perform periodic tasks
                    await self._on_tick()
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in agent {self.agent_id} main loop: {str(e)}")
                    await asyncio.sleep(1.0)  # Longer delay on error
                    
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id} main loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in agent {self.agent_id} main loop: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Agent-specific processing logic.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processing result data
        """
        pass
    
    # Optional lifecycle hooks that subclasses can override
    
    async def _on_start(self) -> None:
        """Called when the agent is starting."""
        pass
    
    async def _on_stop(self) -> None:
        """Called when the agent is stopping."""
        pass
    
    async def _on_pause(self) -> None:
        """Called when the agent is paused."""
        pass
    
    async def _on_resume(self) -> None:
        """Called when the agent is resumed."""
        pass
    
    async def _on_tick(self) -> None:
        """Called periodically during the main loop."""
        pass
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: Incoming message to handle
        """
        logger.debug(f"Agent {self.agent_id} received message: {message.message_type}")
        # Default implementation does nothing
        # Subclasses can override to handle specific message types 