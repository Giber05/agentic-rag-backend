"""
Query Rewriting Agent for optimizing user queries.

This agent handles:
- Spell and grammar correction
- Query simplification and clarity enhancement
- Query normalization for consistent embeddings
- Query preprocessing and validation
- Query expansion techniques
"""

import asyncio
import logging
import re
import string
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..core.openai_config import OpenAIModels

from .base import BaseAgent

logger = logging.getLogger(__name__)


class QueryRewritingAgent(BaseAgent):
    """
    Agent responsible for rewriting and optimizing user queries.
    
    Capabilities:
    - Spell checking and correction
    - Grammar improvement
    - Query simplification
    - Normalization for embeddings
    - Query expansion
    - Validation and sanitization
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "query_rewriter",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Query Rewriting Agent."""
        super().__init__(agent_id, agent_type, config)
        
        # Configuration
        self.max_query_length = config.get("max_query_length", 500) if config else 500
        self.min_query_length = config.get("min_query_length", 3) if config else 3
        self.enable_expansion = config.get("enable_expansion", True) if config else True
        self.enable_spell_check = config.get("enable_spell_check", True) if config else True
        self.enable_grammar_check = config.get("enable_grammar_check", True) if config else True
        
        # OpenAI service for advanced rewriting (will be initialized on start)
        self.openai_service = None
        
        # Common query patterns and replacements
        self.common_replacements = {
            "whats": "what is",
            "hows": "how is",
            "wheres": "where is",
            "whos": "who is",
            "whens": "when is",
            "cant": "cannot",
            "wont": "will not",
            "dont": "do not",
            "isnt": "is not",
            "arent": "are not",
            "wasnt": "was not",
            "werent": "were not",
            "hasnt": "has not",
            "havent": "have not",
            "hadnt": "had not",
            "wouldnt": "would not",
            "shouldnt": "should not",
            "couldnt": "could not",
            "mustnt": "must not"
        }
        
        # Stop words that can be removed for better search
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "the", "this", "but", "they", "have",
            "had", "what", "said", "each", "which", "she", "do", "how", "their",
            "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more",
            "go", "no", "way", "could", "my", "than", "first", "been", "call",
            "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
            "come", "made", "may", "part"
        }
        
        logger.info(f"Query Rewriting Agent {self.agent_id} initialized")
    
    async def _on_start(self) -> None:
        """Initialize OpenAI service when agent starts."""
        try:
            # Import here to avoid circular imports
            from ..core.dependencies import get_openai_service
            self.openai_service = get_openai_service()
            logger.info(f"Query Rewriting Agent {self.agent_id} connected to OpenAI service")
        except Exception as e:
            logger.warning(f"Could not connect to OpenAI service: {str(e)}")
            self.openai_service = None
    
    async def _process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process and rewrite the user query.
        
        Args:
            input_data: Contains the query and optional parameters
            
        Returns:
            Dictionary with rewritten query and metadata
        """
        query = input_data.get("query", "").strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Validation
        validation_result = self._validate_query(query)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid query: {validation_result['reason']}")
        
        # Step 1: Basic preprocessing
        preprocessed = self._preprocess_query(query)
        
        # Step 2: Spell and grammar correction
        corrected = await self._correct_spelling_and_grammar(preprocessed)
        
        # Step 3: Query normalization
        normalized = self._normalize_query(corrected)
        
        # Step 4: Query expansion (if enabled)
        expanded = await self._expand_query(normalized) if self.enable_expansion else normalized
        
        # Step 5: Final optimization
        optimized = self._optimize_query(expanded)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(query, optimized)
        
        # Prepare result
        result = {
            "original_query": query,
            "rewritten_query": optimized,
            "preprocessing_steps": {
                "preprocessed": preprocessed,
                "corrected": corrected,
                "normalized": normalized,
                "expanded": expanded,
                "optimized": optimized
            },
            "improvements": self._identify_improvements(query, optimized),
            "confidence": confidence,
            "metadata": {
                "original_length": len(query),
                "rewritten_length": len(optimized),
                "processing_timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
        }
        
        logger.debug(f"Query rewritten: '{query}' -> '{optimized}' (confidence: {confidence:.2f})")
        
        return result
    
    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate the input query."""
        if len(query) < self.min_query_length:
            return {
                "valid": False,
                "reason": f"Query too short (minimum {self.min_query_length} characters)"
            }
        
        if len(query) > self.max_query_length:
            return {
                "valid": False,
                "reason": f"Query too long (maximum {self.max_query_length} characters)"
            }
        
        # Check for malicious content
        if self._contains_malicious_content(query):
            return {
                "valid": False,
                "reason": "Query contains potentially malicious content"
            }
        
        return {"valid": True, "reason": None}
    
    def _contains_malicious_content(self, query: str) -> bool:
        """Check for potentially malicious content in the query."""
        malicious_patterns = [
            r"<script",
            r"javascript:",
            r"eval\(",
            r"exec\(",
            r"system\(",
            r"__import__",
            r"DROP\s+TABLE",
            r"DELETE\s+FROM",
            r"INSERT\s+INTO",
            r"UPDATE\s+SET"
        ]
        
        query_lower = query.lower()
        for pattern in malicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _preprocess_query(self, query: str) -> str:
        """Basic preprocessing of the query."""
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that don't add meaning
        processed = re.sub(r'[^\w\s\-\?\!\.]', ' ', processed)
        
        # Handle common contractions
        for contraction, expansion in self.common_replacements.items():
            processed = re.sub(r'\b' + contraction + r'\b', expansion, processed, flags=re.IGNORECASE)
        
        # Normalize case (keep first letter capitalized)
        if processed:
            processed = processed[0].upper() + processed[1:].lower()
        
        return processed.strip()
    
    async def _correct_spelling_and_grammar(self, query: str) -> str:
        """Correct spelling and grammar using OpenAI if available."""
        if not self.enable_spell_check or not self.openai_service:
            return query
        
        try:
            # Use OpenAI for spell and grammar correction
            prompt = f"""Please correct any spelling and grammar errors in the following query while preserving its original meaning and intent. Only make necessary corrections, don't rephrase unnecessarily.

Query: "{query}"

Corrected query:"""
            
            response = await self.openai_service.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=OpenAIModels.GPT_4_1_NANO,
                max_tokens=100,
                temperature=0.1
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # Remove quotes if the model added them
            corrected = corrected.strip('"\'')
            
            # Validate the correction isn't too different
            if self._similarity_score(query, corrected) < 0.5:
                logger.warning(f"Correction too different, keeping original: {query}")
                return query
            
            return corrected
            
        except Exception as e:
            logger.warning(f"Spell check failed: {str(e)}")
            return query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize the query for consistent processing."""
        # Convert to lowercase for processing
        normalized = query.lower()
        
        # Remove punctuation except question marks and periods
        normalized = re.sub(r'[^\w\s\?\.]', ' ', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Ensure it ends with appropriate punctuation
        if normalized and not normalized.endswith(('.', '?', '!')):
            if any(word in normalized for word in ['what', 'how', 'when', 'where', 'who', 'why', 'which']):
                normalized += '?'
            else:
                normalized += '.'
        
        return normalized
    
    async def _expand_query(self, query: str) -> str:
        """Expand the query with related terms if beneficial."""
        if not self.enable_expansion or not self.openai_service:
            return query
        
        try:
            # Use OpenAI to suggest query expansion
            prompt = f"""Given the following search query, suggest a slightly expanded version that would help find more relevant information. Keep it concise and focused. Only expand if it would genuinely improve search results.

Original query: "{query}"

Expanded query (or return original if expansion isn't beneficial):"""
            
            response = await self.openai_service.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=OpenAIModels.GPT_4_1_NANO,
                max_tokens=100,
                temperature=0.2
            )
            
            expanded = response.choices[0].message.content.strip().strip('"\'')
            
            # Only use expansion if it's not too much longer
            if len(expanded) <= len(query) * 1.5:
                return expanded
            else:
                return query
                
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return query
    
    def _optimize_query(self, query: str) -> str:
        """Final optimization of the query."""
        # Remove redundant words
        words = query.split()
        
        # Remove stop words if query is long enough
        if len(words) > 5:
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            if len(filtered_words) >= 3:  # Keep at least 3 meaningful words
                query = ' '.join(filtered_words)
        
        # Final cleanup
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _identify_improvements(self, original: str, rewritten: str) -> List[str]:
        """Identify what improvements were made."""
        improvements = []
        
        if len(rewritten) != len(original):
            if len(rewritten) > len(original):
                improvements.append("expanded")
            else:
                improvements.append("simplified")
        
        if original.lower() != rewritten.lower():
            improvements.append("corrected")
        
        if re.search(r'[^\w\s\?\.\!]', original) and not re.search(r'[^\w\s\?\.\!]', rewritten):
            improvements.append("normalized")
        
        if not improvements:
            improvements.append("validated")
        
        return improvements
    
    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        """Calculate confidence score for the rewriting."""
        # Base confidence
        confidence = 0.8
        
        # Adjust based on similarity
        similarity = self._similarity_score(original, rewritten)
        if similarity > 0.9:
            confidence += 0.1
        elif similarity < 0.5:
            confidence -= 0.3
        
        # Adjust based on length change
        length_ratio = len(rewritten) / len(original) if original else 1.0
        if 0.8 <= length_ratio <= 1.2:
            confidence += 0.05
        
        # Adjust based on grammar improvements
        if self._has_better_grammar(original, rewritten):
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _has_better_grammar(self, original: str, rewritten: str) -> bool:
        """Simple heuristic to check if grammar improved."""
        # Check for common grammar improvements
        improvements = [
            (r'\bi\b', r'\bI\b'),  # Capitalized "I"
            (r'\s+', r' '),        # Reduced multiple spaces
            (r'[^\w\s\?\.\!]', ''), # Removed special characters
        ]
        
        for pattern, replacement in improvements:
            if re.search(pattern, original) and not re.search(pattern, rewritten):
                return True
        
        return False 