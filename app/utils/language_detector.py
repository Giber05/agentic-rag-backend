"""
Language detection utility for determining user query language
and providing appropriate response language instructions.
"""

import re
from typing import Dict, Tuple
from enum import Enum


class Language(Enum):
    """Supported languages for the RAG system."""
    ENGLISH = "en"
    INDONESIAN = "id"
    AUTO = "auto"


class LanguageDetector:
    """Utility class for detecting language in user queries."""
    
    def __init__(self):
        # Indonesian language patterns and common words
        self.indonesian_patterns = {
            'words': [
                'apa', 'yang', 'adalah', 'bagaimana', 'mengapa', 'dimana', 'kapan', 
                'siapa', 'ini', 'itu', 'dan', 'atau', 'dengan', 'untuk', 'dari',
                'ke', 'di', 'pada', 'dalam', 'akan', 'sudah', 'sedang', 'bisa',
                'dapat', 'harus', 'perlu', 'ingin', 'mau', 'tidak', 'ya', 'ada',
                'fitur', 'sistem', 'aplikasi', 'website', 'pengguna', 'data',
                'informasi', 'tentang', 'mengenai', 'jelaskan', 'tolong', 'mohon',
                'saya', 'kamu', 'kami', 'mereka', 'dia', 'kita', 'project',
                'projek', 'aplikasi', 'sistem', 'website'
            ],
            'suffixes': ['kan', 'an', 'nya', 'ku', 'mu'],
            'prefixes': ['me', 'di', 'ter', 'ber', 'ke', 'se', 'per', 'peng', 'mem']
        }
        
        # English language patterns and common words
        self.english_patterns = {
            'words': [
                'what', 'how', 'why', 'where', 'when', 'who', 'which', 'can',
                'could', 'would', 'should', 'will', 'the', 'and', 'or', 'with',
                'for', 'from', 'to', 'in', 'on', 'at', 'by', 'is', 'are', 'was',
                'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'about', 'explain', 'please', 'help', 'feature', 'system', 'app',
                'application', 'website', 'user', 'data', 'information', 'project'
            ],
            'articles': ['a', 'an', 'the'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
        }
    
    def detect_language(self, text: str) -> Tuple[Language, float]:
        """
        Detect the language of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        if not text or len(text.strip()) < 3:
            return Language.ENGLISH, 0.5  # Default to English for very short text
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return Language.ENGLISH, 0.5
        
        indonesian_score = self._calculate_indonesian_score(text_lower, words)
        english_score = self._calculate_english_score(text_lower, words)
        
        # Normalize scores
        total_score = indonesian_score + english_score
        if total_score > 0:
            indonesian_confidence = indonesian_score / total_score
            english_confidence = english_score / total_score
        else:
            indonesian_confidence = 0.5
            english_confidence = 0.5
        
        # Determine language based on higher confidence
        if indonesian_confidence > english_confidence:
            return Language.INDONESIAN, indonesian_confidence
        else:
            return Language.ENGLISH, english_confidence
    
    def _calculate_indonesian_score(self, text: str, words: list) -> float:
        """Calculate Indonesian language score."""
        score = 0.0
        
        # Check for Indonesian words
        for word in words:
            if word in self.indonesian_patterns['words']:
                score += 2.0
        
        # Check for Indonesian suffixes
        for word in words:
            for suffix in self.indonesian_patterns['suffixes']:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    score += 1.0
                    break
        
        # Check for Indonesian prefixes
        for word in words:
            for prefix in self.indonesian_patterns['prefixes']:
                if word.startswith(prefix) and len(word) > len(prefix) + 1:
                    score += 1.0
                    break
        
        # Check for specific Indonesian patterns
        if any(pattern in text for pattern in ['apa itu', 'bagaimana cara', 'jelaskan tentang']):
            score += 3.0
        
        return score
    
    def _calculate_english_score(self, text: str, words: list) -> float:
        """Calculate English language score."""
        score = 0.0
        
        # Check for English words
        for word in words:
            if word in self.english_patterns['words']:
                score += 2.0
        
        # Check for English articles
        for word in words:
            if word in self.english_patterns['articles']:
                score += 1.5
        
        # Check for English pronouns
        for word in words:
            if word in self.english_patterns['pronouns']:
                score += 1.0
        
        # Check for specific English patterns
        if any(pattern in text for pattern in ['what is', 'how to', 'explain about', 'tell me']):
            score += 3.0
        
        return score
    
    def get_language_instruction(self, language: Language) -> str:
        """
        Get the language instruction to add to AI prompts.
        
        Args:
            language: The detected language
            
        Returns:
            String instruction for the AI model
        """
        instructions = {
            Language.ENGLISH: "Please respond in English.",
            Language.INDONESIAN: "Mohon merespons dalam bahasa Indonesia.",
            Language.AUTO: "Please respond in the same language as the user's question."
        }
        
        return instructions.get(language, instructions[Language.ENGLISH])
    
    def get_language_name(self, language: Language) -> str:
        """Get the human-readable name of the language."""
        names = {
            Language.ENGLISH: "English",
            Language.INDONESIAN: "Indonesian",
            Language.AUTO: "Auto-detect"
        }
        
        return names.get(language, "English")


# Global instance
language_detector = LanguageDetector()


def detect_query_language(query: str) -> Tuple[Language, float]:
    """
    Convenience function to detect language of a query.
    
    Args:
        query: User query text
        
    Returns:
        Tuple of (detected_language, confidence_score)
    """
    return language_detector.detect_language(query)


def get_language_instruction_for_query(query: str) -> str:
    """
    Get language instruction based on query language detection.
    
    Args:
        query: User query text
        
    Returns:
        Language instruction string for AI prompt
    """
    detected_language, confidence = detect_query_language(query)
    
    # If confidence is low, use auto-detect
    if confidence < 0.6:
        detected_language = Language.AUTO
    
    return language_detector.get_language_instruction(detected_language) 