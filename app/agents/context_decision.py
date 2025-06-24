"""
Context Decision Agent for determining retrieval necessity.

This agent handles:
- Semantic similarity assessment between query and conversation history
- Multi-step decision logic for context necessity evaluation
- Confidence scoring for decisions
- Decision explanation and logging
- Adaptive decision thresholds
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..core.openai_config import OpenAIModels
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ContextNecessity(Enum):
    """Enumeration for context necessity decisions."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_NEEDED = "not_needed"


class DecisionReason(Enum):
    """Enumeration for decision reasoning."""
    NEW_TOPIC = "new_topic"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    FACTUAL_QUERY = "factual_query"
    CONVERSATIONAL = "conversational"
    COMPLEX_QUERY = "complex_query"
    SIMPLE_QUERY = "simple_query"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    SUFFICIENT_CONTEXT = "sufficient_context"


class ContextDecisionAgent(BaseAgent):
    """
    Agent responsible for determining whether additional context retrieval is needed.
    
    Capabilities:
    - Semantic similarity assessment
    - Multi-step decision logic
    - Context necessity evaluation
    - Confidence scoring
    - Decision explanation
    - Adaptive thresholds
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "context_decision",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Context Decision Agent."""
        super().__init__(agent_id, agent_type, config)
        
        # Configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.7) if config else 0.7
        self.context_window_size = config.get("context_window_size", 5) if config else 5
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.6) if config else 0.6
        self.enable_ai_assessment = config.get("enable_ai_assessment", True) if config else True
        self.adaptive_thresholds = config.get("adaptive_thresholds", True) if config else True
        
        # OpenAI service for advanced assessment (will be initialized on start)
        self.openai_service = None
        
        # Query patterns that typically require context
        self.context_requiring_patterns = [
            r'\b(what|how|why|when|where|who)\b.*\b(this|that|it|they|them)\b',
            r'\b(explain|elaborate|tell me more|continue|expand)\b',
            r'\b(compare|difference|similar|contrast)\b',
            r'\b(example|instance|case|scenario)\b',
            r'\b(follow.?up|related|connection|link)\b',
            r'\b(previous|earlier|before|mentioned|discussed)\b'
        ]
        
        # Query patterns that typically don't require context
        self.standalone_patterns = [
            r'^\s*(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'^\s*(thank you|thanks|bye|goodbye|see you)\b',
            r'^\s*(yes|no|ok|okay|sure|fine)\s*$',
            r'\b(what is|define|definition of)\s+\w+\s*$',
            r'\b(how to)\s+\w+',
            r'^\s*\w+\s*\?\s*$'  # Single word questions
        ]
        
        # Factual query patterns
        self.factual_patterns = [
            r'\b(what is|what are|define|definition)\b',
            r'\b(when did|when was|when will)\b',
            r'\b(where is|where are|where can)\b',
            r'\b(who is|who was|who are)\b',
            r'\b(how many|how much|how long)\b',
            r'\b(list|name|identify)\b.*\b(all|some|few)\b'
        ]
        
        logger.info(f"Context Decision Agent {self.agent_id} initialized")
    
    async def _on_start(self) -> None:
        """Initialize OpenAI service when agent starts."""
        try:
            # Import here to avoid circular imports
            from ..core.dependencies import get_openai_service
            self.openai_service = get_openai_service()
            logger.info(f"Context Decision Agent {self.agent_id} connected to OpenAI service")
        except Exception as e:
            logger.warning(f"Could not connect to OpenAI service: {str(e)}")
            self.openai_service = None
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate whether additional context retrieval is needed.
        
        Args:
            input_data: Contains the query, conversation history, and optional parameters
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        query = input_data.get("query", "").strip()
        conversation_history = input_data.get("conversation_history", [])
        current_context = input_data.get("current_context", {})
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Step 1: Pattern-based initial assessment
        pattern_assessment = self._assess_query_patterns(query)
        
        # Step 2: Conversation context analysis
        context_analysis = self._analyze_conversation_context(query, conversation_history)
        
        # Step 3: Semantic similarity assessment
        similarity_assessment = await self._assess_semantic_similarity(
            query, conversation_history, current_context
        )
        
        # Step 4: AI-powered assessment (if enabled)
        ai_assessment = await self._ai_powered_assessment(
            query, conversation_history, current_context
        ) if self.enable_ai_assessment else None
        
        # Step 5: Multi-factor decision making
        decision_result = self._make_decision(
            query,
            pattern_assessment,
            context_analysis,
            similarity_assessment,
            ai_assessment
        )
        
        # Step 6: Adaptive threshold adjustment
        if self.adaptive_thresholds:
            self._adjust_thresholds(decision_result)
        
        # Prepare result
        result = {
            "query": query,
            "decision": decision_result["necessity"].value,
            "confidence": decision_result["confidence"],
            "reasoning": decision_result["reasoning"],
            "decision_factors": {
                "pattern_assessment": pattern_assessment,
                "context_analysis": context_analysis,
                "similarity_assessment": similarity_assessment,
                "ai_assessment": ai_assessment
            },
            "recommendations": decision_result["recommendations"],
            "metadata": {
                "processing_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "thresholds_used": {
                    "similarity_threshold": self.similarity_threshold,
                    "confidence_threshold": self.min_confidence_threshold
                }
            }
        }
        
        logger.debug(
            f"Context decision: {decision_result['necessity'].value} "
            f"(confidence: {decision_result['confidence']:.2f}) "
            f"for query: '{query[:50]}...'"

        )
        logger.info(f"Result of context decision agent: {result}")
        
        return result
    
    def _assess_query_patterns(self, query: str) -> Dict[str, Any]:
        """Assess query using pattern matching."""
        query_lower = query.lower()
        
        # Check for context-requiring patterns
        context_required_score = 0
        matched_requiring_patterns = []
        for pattern in self.context_requiring_patterns:
            if re.search(pattern, query_lower):
                context_required_score += 1
                matched_requiring_patterns.append(pattern)
        
        # Check for standalone patterns
        standalone_score = 0
        matched_standalone_patterns = []
        for pattern in self.standalone_patterns:
            if re.search(pattern, query_lower):
                standalone_score += 1
                matched_standalone_patterns.append(pattern)
        
        # Check for factual patterns
        factual_score = 0
        matched_factual_patterns = []
        for pattern in self.factual_patterns:
            if re.search(pattern, query_lower):
                factual_score += 1
                matched_factual_patterns.append(pattern)
        
        # Calculate pattern-based necessity
        if standalone_score > 0:
            pattern_necessity = ContextNecessity.NOT_NEEDED
            pattern_confidence = 0.8
        elif context_required_score > 0:
            pattern_necessity = ContextNecessity.REQUIRED
            pattern_confidence = 0.7
        elif factual_score > 0:
            pattern_necessity = ContextNecessity.OPTIONAL
            pattern_confidence = 0.6
        else:
            pattern_necessity = ContextNecessity.OPTIONAL
            pattern_confidence = 0.5
        
        return {
            "necessity": pattern_necessity,
            "confidence": pattern_confidence,
            "context_required_score": context_required_score,
            "standalone_score": standalone_score,
            "factual_score": factual_score,
            "matched_patterns": {
                "requiring": matched_requiring_patterns,
                "standalone": matched_standalone_patterns,
                "factual": matched_factual_patterns
            }
        }
    
    def _analyze_conversation_context(
        self, 
        query: str, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze conversation context to determine necessity."""
        if not conversation_history:
            return {
                "necessity": ContextNecessity.REQUIRED,
                "confidence": 0.8,
                "reason": DecisionReason.NEW_TOPIC,
                "context_available": False,
                "recent_messages": 0,
                "topic_continuity": 0.0
            }
        
        # Get recent messages within context window
        recent_messages = conversation_history[-self.context_window_size:]
        
        # Analyze query for pronouns and references
        pronouns_and_refs = re.findall(
            r'\b(this|that|it|they|them|he|she|his|her|its|their)\b',
            query.lower()
        )
        
        # Check for follow-up indicators
        follow_up_indicators = re.findall(
            r'\b(also|additionally|furthermore|moreover|besides|too|as well)\b',
            query.lower()
        )
        
        # Calculate topic continuity score
        topic_continuity = self._calculate_topic_continuity(query, recent_messages)
        
        # Determine necessity based on context analysis
        if len(pronouns_and_refs) > 0 or len(follow_up_indicators) > 0:
            necessity = ContextNecessity.REQUIRED
            confidence = 0.9
            reason = DecisionReason.FOLLOW_UP
        elif topic_continuity > 0.7:
            necessity = ContextNecessity.OPTIONAL
            confidence = 0.7
            reason = DecisionReason.FOLLOW_UP
        elif len(recent_messages) > 0:
            necessity = ContextNecessity.OPTIONAL
            confidence = 0.6
            reason = DecisionReason.CONVERSATIONAL
        else:
            necessity = ContextNecessity.REQUIRED
            confidence = 0.8
            reason = DecisionReason.NEW_TOPIC
        
        return {
            "necessity": necessity,
            "confidence": confidence,
            "reason": reason,
            "context_available": len(conversation_history) > 0,
            "recent_messages": len(recent_messages),
            "topic_continuity": topic_continuity,
            "pronouns_found": len(pronouns_and_refs),
            "follow_up_indicators": len(follow_up_indicators)
        }
    
    def _calculate_topic_continuity(
        self, 
        query: str, 
        recent_messages: List[Dict[str, Any]]
    ) -> float:
        """Calculate topic continuity score between query and recent messages."""
        if not recent_messages:
            return 0.0
        
        query_words = set(query.lower().split())
        
        # Remove common stop words for better comparison
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        query_words = query_words - stop_words
        
        if not query_words:
            return 0.0
        
        max_similarity = 0.0
        
        for message in recent_messages:
            content = message.get('content', '')
            if isinstance(content, str):
                message_words = set(content.lower().split()) - stop_words
                
                if message_words:
                    # Calculate Jaccard similarity
                    intersection = query_words.intersection(message_words)
                    union = query_words.union(message_words)
                    similarity = len(intersection) / len(union) if union else 0.0
                    max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    async def _assess_semantic_similarity(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess semantic similarity using embeddings if available."""
        try:
            if not self.openai_service or not conversation_history:
                return {
                    "similarity_score": 0.0,
                    "necessity": ContextNecessity.REQUIRED,
                    "confidence": 0.5,
                    "method": "fallback"
                }
            
            # Get query embedding
            query_embedding = await self.openai_service.create_embedding(query)
            if not query_embedding:
                return {
                    "similarity_score": 0.0,
                    "necessity": ContextNecessity.REQUIRED,
                    "confidence": 0.5,
                    "method": "embedding_failed"
                }
            
            query_vector = query_embedding
            
            # Get embeddings for recent conversation messages
            recent_messages = conversation_history[-self.context_window_size:]
            message_texts = [msg.get('content', '') for msg in recent_messages if msg.get('content')]
            
            if not message_texts:
                return {
                    "similarity_score": 0.0,
                    "necessity": ContextNecessity.REQUIRED,
                    "confidence": 0.8,
                    "method": "no_context"
                }
            
            # Get embeddings for context messages
            context_embeddings = []
            for text in message_texts:
                embedding = await self.openai_service.create_embedding(text)
                if embedding:
                    context_embeddings.append(embedding)
            
            if not context_embeddings:
                return {
                    "similarity_score": 0.0,
                    "necessity": ContextNecessity.REQUIRED,
                    "confidence": 0.5,
                    "method": "context_embedding_failed"
                }
            
            # Calculate cosine similarities
            max_similarity = 0.0
            for context_vector in context_embeddings:
                similarity = self._cosine_similarity(query_vector, context_vector)
                max_similarity = max(max_similarity, similarity)
            
            # Determine necessity based on similarity
            if max_similarity > self.similarity_threshold:
                necessity = ContextNecessity.OPTIONAL
                confidence = 0.8
            elif max_similarity > self.similarity_threshold * 0.7:
                necessity = ContextNecessity.OPTIONAL
                confidence = 0.6
            else:
                necessity = ContextNecessity.REQUIRED
                confidence = 0.7
            
            return {
                "similarity_score": max_similarity,
                "necessity": necessity,
                "confidence": confidence,
                "method": "semantic_embedding",
                "threshold_used": self.similarity_threshold
            }
            
        except Exception as e:
            logger.warning(f"Semantic similarity assessment failed: {str(e)}")
            return {
                "similarity_score": 0.0,
                "necessity": ContextNecessity.REQUIRED,
                "confidence": 0.5,
                "method": "error",
                "error": str(e)
            }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _ai_powered_assessment(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        current_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Use AI to assess context necessity."""
        if not self.openai_service:
            return None
        
        try:
            # Prepare conversation context
            context_text = ""
            if conversation_history:
                recent_messages = conversation_history[-3:]  # Last 3 messages
                context_text = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in recent_messages
                ])
            
            prompt = f"""Analyze whether the following query requires additional context retrieval from a knowledge base.

Query: "{query}"

Recent conversation context:
{context_text if context_text else "No previous conversation"}

Consider:
1. Does the query reference previous topics or use pronouns that need context?
2. Is this a standalone factual question that can be answered independently?
3. Does the query build upon previous conversation?
4. Is additional information needed to provide a complete answer?

Respond with one of: REQUIRED, OPTIONAL, NOT_NEEDED
Also provide a confidence score (0.0-1.0) and brief reasoning.

Format: DECISION|CONFIDENCE|REASONING"""

            response = await self.openai_service.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=OpenAIModels.GPT_4_1_NANO,
                max_tokens=150,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            parts = result_text.split('|')
            
            if len(parts) >= 3:
                decision_str = parts[0].strip().upper()
                confidence_str = parts[1].strip()
                reasoning = parts[2].strip()
                
                # Parse decision
                if decision_str == "REQUIRED":
                    necessity = ContextNecessity.REQUIRED
                elif decision_str == "NOT_NEEDED":
                    necessity = ContextNecessity.NOT_NEEDED
                else:
                    necessity = ContextNecessity.OPTIONAL
                
                # Parse confidence
                try:
                    confidence = float(confidence_str)
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
                
                return {
                    "necessity": necessity,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "method": "ai_assessment"
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"AI-powered assessment failed: {str(e)}")
            return None
    
    def _make_decision(
        self,
        query: str,
        pattern_assessment: Dict[str, Any],
        context_analysis: Dict[str, Any],
        similarity_assessment: Dict[str, Any],
        ai_assessment: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make final decision based on all assessment factors."""
        
        # Collect all assessments
        assessments = [
            ("pattern", pattern_assessment),
            ("context", context_analysis),
            ("similarity", similarity_assessment)
        ]
        
        if ai_assessment:
            assessments.append(("ai", ai_assessment))
        
        # Weight the assessments
        weights = {
            "pattern": 0.2,
            "context": 0.3,
            "similarity": 0.3,
            "ai": 0.2
        }
        
        # Calculate weighted scores for each necessity level
        necessity_scores = {
            ContextNecessity.REQUIRED: 0.0,
            ContextNecessity.OPTIONAL: 0.0,
            ContextNecessity.NOT_NEEDED: 0.0
        }
        
        total_weight = 0.0
        confidence_sum = 0.0
        
        for assessment_type, assessment in assessments:
            weight = weights.get(assessment_type, 0.0)
            necessity = assessment.get("necessity")
            confidence = assessment.get("confidence", 0.5)
            
            if necessity in necessity_scores:
                necessity_scores[necessity] += weight * confidence
                total_weight += weight
                confidence_sum += confidence
        
        # Determine final decision
        if total_weight > 0:
            # Normalize scores
            for necessity in necessity_scores:
                necessity_scores[necessity] /= total_weight
            
            # Find the necessity with highest score
            final_necessity = max(necessity_scores, key=necessity_scores.get)
            final_confidence = confidence_sum / len(assessments)
        else:
            # Fallback decision
            final_necessity = ContextNecessity.OPTIONAL
            final_confidence = 0.5
        
        # Generate reasoning
        reasoning_parts = []
        for assessment_type, assessment in assessments:
            if assessment.get("necessity"):
                reasoning_parts.append(
                    f"{assessment_type}: {assessment['necessity'].value} "
                    f"(conf: {assessment.get('confidence', 0.5):.2f})"
                )
        
        reasoning = "; ".join(reasoning_parts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_necessity, final_confidence, assessments
        )
        
        return {
            "necessity": final_necessity,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "necessity_scores": {k.value: v for k, v in necessity_scores.items()},
            "recommendations": recommendations
        }
    
    def _generate_recommendations(
        self,
        necessity: ContextNecessity,
        confidence: float,
        assessments: List[Tuple[str, Dict[str, Any]]]
    ) -> List[str]:
        """Generate recommendations based on the decision."""
        recommendations = []
        
        if necessity == ContextNecessity.REQUIRED:
            recommendations.append("Retrieve relevant context from knowledge base")
            if confidence < 0.7:
                recommendations.append("Consider multiple retrieval strategies")
        elif necessity == ContextNecessity.OPTIONAL:
            recommendations.append("Context retrieval may improve answer quality")
            if confidence > 0.8:
                recommendations.append("Proceed with lightweight context retrieval")
            else:
                recommendations.append("Evaluate query complexity before retrieval")
        else:  # NOT_NEEDED
            recommendations.append("Query can be answered without additional context")
            if confidence < 0.7:
                recommendations.append("Monitor for follow-up questions")
        
        # Add specific recommendations based on assessments
        for assessment_type, assessment in assessments:
            if assessment_type == "context" and assessment.get("pronouns_found", 0) > 0:
                recommendations.append("Query contains pronouns - context likely needed")
            elif assessment_type == "similarity" and assessment.get("similarity_score", 0) > 0.8:
                recommendations.append("High similarity to recent conversation")
        
        return recommendations
    
    def _adjust_thresholds(self, decision_result: Dict[str, Any]) -> None:
        """Adjust thresholds based on decision confidence."""
        confidence = decision_result.get("confidence", 0.5)
        
        # Adjust similarity threshold based on confidence
        if confidence < 0.6:
            # Lower threshold if we're not confident
            self.similarity_threshold = max(0.5, self.similarity_threshold - 0.05)
        elif confidence > 0.9:
            # Raise threshold if we're very confident
            self.similarity_threshold = min(0.9, self.similarity_threshold + 0.02)
        
        # Keep thresholds within reasonable bounds
        self.similarity_threshold = max(0.5, min(0.9, self.similarity_threshold)) 