"""
Query Processing Service
Handles PII filtering, intent classification, query rewriting, and routing decisions
"""

import time
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from models.rag_models import (
    QueryProcessingResult, PIIDetectionResult, IntentType, 
    IntentClassificationResult
)
from services.pii_service import pii_service
from services.generation import generation_service

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Service for processing queries through PII filtering, intent classification, and rewriting"""
    
    def __init__(self):
        self.intent_keywords = self._initialize_intent_keywords()
        self.rag_keywords = self._initialize_rag_keywords()
        self.greeting_patterns = self._initialize_greeting_patterns()
        
    def _initialize_intent_keywords(self) -> Dict[IntentType, List[str]]:
        """Initialize keywords for intent classification"""
        return {
            IntentType.FACTUAL_QUERY: [
                "what", "how", "when", "where", "why", "who", "which", "define", "explain",
                "describe", "calculate", "analyze", "compare", "difference", "between",
                "meaning", "definition", "process", "procedure", "steps", "method",
                "example", "examples", "show me", "tell me about", "information about"
            ],
            IntentType.CONVERSATIONAL: [
                "think", "opinion", "feel", "believe", "prefer", "like", "dislike",
                "recommend", "suggest", "advice", "help me", "assist", "support",
                "chat", "talk", "discuss", "conversation", "personal", "experience"
            ],
            IntentType.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "greetings", "howdy", "what's up", "how are you", "nice to meet",
                "goodbye", "bye", "see you", "farewell", "thank you", "thanks"
            ],
            IntentType.CLARIFICATION: [
                "clarify", "explain more", "elaborate", "what do you mean", "can you",
                "could you", "more details", "specifically", "in particular",
                "follow up", "regarding", "about that", "previous", "earlier",
                "you mentioned", "you said"
            ],
            IntentType.TASK_ORIENTED: [
                "create", "make", "build", "generate", "write", "compose", "draft",
                "design", "plan", "schedule", "organize", "list", "summarize",
                "translate", "convert", "format", "edit", "review", "check"
            ]
        }
    
    def _initialize_rag_keywords(self) -> List[str]:
        """Initialize keywords that suggest RAG should be used"""
        return [
            # Document-specific terms
            "document", "report", "paper", "study", "research", "analysis",
            "statistics", "findings", "results", "conclusion", "methodology",
            "section", "chapter", "page", "paragraph", "table", "figure", "chart",
            
            # Financial/business terms (based on your domain)
            "financial", "finance", "investment", "revenue", "profit", "loss",
            "market", "trading", "portfolio", "risk", "compliance", "regulation",
            "audit", "accounting", "budget", "forecast", "valuation", "metrics",
            
            # Factual inquiry terms
            "according to", "based on", "mentioned in", "states that", "shows that",
            "indicates", "demonstrates", "evidence", "source", "reference", "cite",
            "quote", "extract", "specific", "exact", "precise", "detailed"
        ]
    
    def _initialize_greeting_patterns(self) -> List[str]:
        """Initialize regex patterns for greetings"""
        return [
            r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b",
            r"^(hi|hello|hey).*my\s+name\s+is\b",  # Introductions
            r"\b(how\s+are\s+you|what's\s+up|how's\s+it\s+going)\b",
            r"^(thanks?|thank\s+you|goodbye|bye|see\s+you)\b",
            r"\bmy\s+name\s+is\b.*\band\s+i\b",  # "My name is X and I..."
            r"^(hello|hi).*\bi\s+(am|work|do)\b"  # "Hello, I am/work/do..."
        ]
    
    async def process_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryProcessingResult:
        """
        Process a query through the complete pipeline
        
        Args:
            query: User query to process
            conversation_history: Previous conversation messages for context
            
        Returns:
            QueryProcessingResult with all processing details
        """
        start_time = time.time()
        
        try:
            # Step 1: PII Detection and Filtering
            pii_result = await pii_service.detect_pii(query)
            
            # Check if query should be rejected due to PII
            should_reject, rejection_reason = await pii_service.should_reject_query(pii_result)
            if should_reject:
                raise ValueError(f"Query rejected: {rejection_reason}")
            
            # Use masked query for further processing
            processed_query = pii_result.masked_query
            
            # Step 2: Intent Classification
            intent_result = await self._classify_intent(processed_query, conversation_history)
            
            # Step 3: Routing Decision
            routing_decision = "RAG" if intent_result.should_use_rag else "ChatGPT"
            
            # Step 4: Query Expansion (only for RAG queries)
            expanded_queries = []
            if intent_result.should_use_rag:
                expanded_queries = await self._expand_query(processed_query)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = QueryProcessingResult(
                original_query=query,
                processed_query=processed_query,
                pii_detection=pii_result,
                intent_classification={
                    "predicted_intent": intent_result.predicted_intent.value,
                    "confidence_score": intent_result.confidence_score,
                    "should_use_rag": intent_result.should_use_rag,
                    "reasoning": intent_result.reasoning
                },
                routing_decision=routing_decision,
                expanded_queries=expanded_queries,
                processing_time_ms=processing_time
            )
            
            logger.info(f"Query processed: {routing_decision} routing, intent: {intent_result.predicted_intent.value}")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return a safe fallback result
            processing_time = (time.time() - start_time) * 1000
            return QueryProcessingResult(
                original_query=query,
                processed_query=query,  # Use original if processing fails
                pii_detection=PIIDetectionResult(
                    has_pii=False,
                    pii_types=[],
                    masked_query=query,
                    confidence_score=0.0,
                    detected_patterns=[]
                ),
                intent_classification={
                    "predicted_intent": IntentType.CONVERSATIONAL.value,
                    "confidence_score": 0.5,
                    "should_use_rag": False,
                    "reasoning": "Fallback due to processing error"
                },
                routing_decision="ChatGPT",
                expanded_queries=[],
                processing_time_ms=processing_time
            )
    
    async def _classify_intent(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> IntentClassificationResult:
        """
        Classify user intent based on query content and conversation context
        
        Args:
            query: Processed query text
            conversation_history: Previous conversation messages
            
        Returns:
            IntentClassificationResult with classification details
        """
        query_lower = query.lower()
        
        # Check for greetings first
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower):
                return IntentClassificationResult(
                    predicted_intent=IntentType.GREETING,
                    confidence_score=0.95,
                    should_use_rag=False,
                    reasoning="Detected greeting pattern"
                )
        
        # Check for clarification requests (needs conversation context)
        if conversation_history and len(conversation_history) > 0:
            clarification_score = self._calculate_keyword_score(query_lower, IntentType.CLARIFICATION)
            if clarification_score > 0.3:
                return IntentClassificationResult(
                    predicted_intent=IntentType.CLARIFICATION,
                    confidence_score=clarification_score,
                    should_use_rag=True,  # Usually needs document context
                    reasoning="Follow-up question detected with conversation context"
                )
        
        # Calculate scores for each intent type
        intent_scores = {}
        for intent_type in IntentType:
            if intent_type not in [IntentType.GREETING, IntentType.CLARIFICATION]:
                intent_scores[intent_type] = self._calculate_keyword_score(query_lower, intent_type)
        
        # Find the highest scoring intent
        predicted_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        confidence_score = intent_scores[predicted_intent]
        
        # Determine RAG vs ChatGPT routing
        should_use_rag = self._should_use_rag(query_lower, predicted_intent, confidence_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(predicted_intent, confidence_score, should_use_rag, query_lower)
        
        return IntentClassificationResult(
            predicted_intent=predicted_intent,
            confidence_score=confidence_score,
            should_use_rag=should_use_rag,
            reasoning=reasoning
        )
    
    def _calculate_keyword_score(self, query: str, intent_type: IntentType) -> float:
        """Calculate keyword-based score for an intent type"""
        keywords = self.intent_keywords.get(intent_type, [])
        if not keywords:
            return 0.0
        
        matches = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in query:
                matches += 1
        
        # Base score from keyword matches
        base_score = matches / total_keywords
        
        # Boost score for exact phrase matches
        for keyword in keywords:
            if len(keyword.split()) > 1 and keyword in query:
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _should_use_rag(self, query: str, intent_type: IntentType, confidence_score: float) -> bool:
        """Determine if RAG should be used based on intent and content analysis"""
        
        # Always use RAG for factual queries and clarifications
        if intent_type in [IntentType.FACTUAL_QUERY, IntentType.CLARIFICATION]:
            return True
        
        # Never use RAG for greetings
        if intent_type == IntentType.GREETING:
            return False
        
        # Check for RAG-indicating keywords
        rag_keyword_matches = sum(1 for keyword in self.rag_keywords if keyword in query)
        rag_score = rag_keyword_matches / len(self.rag_keywords) if self.rag_keywords else 0
        
        # Use RAG if there are strong document-related keywords
        if rag_score > 0.15:  # Increased threshold to 15% to reduce false positives
            return True
        
        # For conversational and task-oriented queries, use confidence threshold
        if intent_type == IntentType.CONVERSATIONAL:
            return rag_score > 0.05  # Lower threshold for conversational
        
        if intent_type == IntentType.TASK_ORIENTED:
            return rag_score > 0.08 or confidence_score < 0.7  # Use RAG for uncertain task queries
        
        # Default to ChatGPT for unclear cases
        return False
    
    def _generate_reasoning(self, intent: IntentType, confidence: float, use_rag: bool, query: str) -> str:
        """Generate human-readable reasoning for the routing decision"""
        
        base_reasoning = f"Classified as {intent.value} with {confidence:.2f} confidence"
        
        if intent == IntentType.GREETING:
            return f"{base_reasoning}. Greetings routed to ChatGPT for natural conversation."
        
        if intent == IntentType.FACTUAL_QUERY:
            return f"{base_reasoning}. Factual queries routed to RAG for document-based answers."
        
        if intent == IntentType.CLARIFICATION:
            return f"{base_reasoning}. Follow-up questions routed to RAG for contextual information."
        
        rag_keywords_found = [kw for kw in self.rag_keywords if kw in query]
        
        if use_rag:
            if rag_keywords_found:
                return f"{base_reasoning}. RAG selected due to document-related keywords: {', '.join(rag_keywords_found[:3])}."
            else:
                return f"{base_reasoning}. RAG selected based on intent type and query structure."
        else:
            return f"{base_reasoning}. ChatGPT selected for general conversation and non-document queries."
    
    async def _expand_query(self, query: str) -> List[str]:
        """
        Expand query using the existing generation service
        
        Args:
            query: Query to expand
            
        Returns:
            List of expanded query variations
        """
        try:
            expanded_queries = await generation_service.expand_query(query)
            logger.debug(f"Generated {len(expanded_queries)} query expansions")
            return expanded_queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """
        Check if query processor is healthy
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a sample query
            test_result = await self.process_query("What is the financial report about?")
            return (
                test_result.routing_decision in ["RAG", "ChatGPT"] and
                test_result.intent_classification["predicted_intent"] is not None
            )
        except Exception as e:
            logger.error(f"Query processor health check failed: {e}")
            return False


# Global query processor instance
query_processor = QueryProcessor()
