"""
Query Classification Service
Classifies queries into External, Context-Dependent, or Document-Based types
"""

import re
from enum import Enum
from typing import List, Optional
import logging
from models.rag_models import ChatMessage

logger = logging.getLogger(__name__)

class QueryType(Enum):
    EXTERNAL = "external"           # Conversation context only
    CONTEXT_DEPENDENT = "context"   # Enhanced RAG + context
    DOCUMENT_BASED = "document"     # Standard RAG + Wikipedia

class QueryClassifier:
    """Service for classifying queries into appropriate processing types"""
    
    def __init__(self):
        # External query patterns (conversation context only)
        self.external_patterns = [
            r"my name is", r"i am", r"i'm", r"what is my", r"who am i",
            r"how are you", r"hello", r"hi", r"goodbye", r"thank you",
            r"i'm feeling", r"can you help", r"i need help", r"i don't understand",
            r"nice to meet you", r"pleased to meet you", r"good morning",
            r"good afternoon", r"good evening", r"see you later", r"take care"
        ]
        
        # Context-dependent patterns (need conversation context)
        self.context_patterns = [
            r"this", r"that", r"it", r"the above", r"the previous",
            r"how to calculate this", r"explain this", r"tell me more about this",
            r"what are the.*of this", r"can you explain that", r"tell me more about that",
            r"how does this work", r"what is this", r"what does this mean",
            r"the.*mentioned above", r"the.*we discussed", r"the.*from before"
        ]
        
        logger.info("QueryClassifier initialized with pattern matching")
    
    async def classify_query(
        self, 
        query: str, 
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> QueryType:
        """
        Classify query into External, Context-Dependent, or Document-Based
        
        Args:
            query: User query to classify
            conversation_history: Previous conversation messages for context
            
        Returns:
            QueryType indicating how to process the query
        """
        try:
            query_lower = query.lower().strip()
            
            # Check for external queries first (highest priority)
            if self._is_external_query(query_lower):
                logger.info(f"Query classified as EXTERNAL: {query[:50]}...")
                return QueryType.EXTERNAL
            
            # Check for context-dependent queries
            if self._is_context_dependent_query(query_lower, conversation_history):
                logger.info(f"Query classified as CONTEXT_DEPENDENT: {query[:50]}...")
                return QueryType.CONTEXT_DEPENDENT
            
            # Default to document-based
            logger.info(f"Query classified as DOCUMENT_BASED: {query[:50]}...")
            return QueryType.DOCUMENT_BASED
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Default to document-based on error
            return QueryType.DOCUMENT_BASED
    
    def _is_external_query(self, query: str) -> bool:
        """Check if query is external (conversation context only)"""
        return any(re.search(pattern, query) for pattern in self.external_patterns)
    
    def _is_context_dependent_query(self, query: str, conversation_history: Optional[List[ChatMessage]] = None) -> bool:
        """Check if query is context-dependent (needs conversation context)"""
        
        # Check for context-dependent patterns
        has_context_pattern = any(re.search(pattern, query) for pattern in self.context_patterns)
        
        # Additional check: if query is very short and we have conversation history
        is_short_with_context = (
            len(query.split()) <= 3 and 
            conversation_history and 
            len(conversation_history) > 1
        )
        
        return has_context_pattern or is_short_with_context
    
    def get_classification_reason(self, query: str, query_type: QueryType) -> str:
        """Get human-readable reason for classification"""
        query_lower = query.lower()
        
        if query_type == QueryType.EXTERNAL:
            matched_patterns = [p for p in self.external_patterns if re.search(p, query_lower)]
            return f"External query (matched patterns: {matched_patterns[:2]})"
        
        elif query_type == QueryType.CONTEXT_DEPENDENT:
            matched_patterns = [p for p in self.context_patterns if re.search(p, query_lower)]
            return f"Context-dependent query (matched patterns: {matched_patterns[:2]})"
        
        else:
            return "Document-based query (default classification)"

# Global instance
query_classifier = QueryClassifier()