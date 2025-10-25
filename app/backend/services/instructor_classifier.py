"""
Instructor-based Query Classification Service
Minimal implementation for query classification only
"""

import logging
import asyncio
import time
from typing import List, Optional
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
import instructor

from config.rag_config import rag_config
from models.rag_models import ChatMessage
from models.instructor_models import QueryClassification, QueryType, FinancialResponse

logger = logging.getLogger(__name__)


class InstructorQueryClassifier:
    """Lightweight service using Instructor for query classification only"""
    
    def __init__(self, config=None):
        self.config = config or rag_config
        self.client = None
        self._initialize_client()
        
        # Rate limiting and retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 30.0  # Maximum delay in seconds
        self.rate_limit_delay = 2.0  # Additional delay for rate limits
        
        # Timeout configuration
        self.classification_timeout = self.config.instructor_classification_timeout
        self.enhancement_timeout = self.config.instructor_enhancement_timeout
        
        # Circuit breaker pattern
        self.failure_count = 0
        self.max_failures = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.last_failure_time = 0
        self.circuit_open = False
    
    def _initialize_client(self):
        """Initialize OpenAI client with Instructor patch"""
        try:
            self.client = OpenAI(api_key=self.config.openai_api_key)
            self.client = instructor.from_openai(self.client)
            logger.info("Instructor query classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Instructor client: {e}")
            self.client = None
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_open:
            return False
        
        # Check if enough time has passed to try again
        if time.time() - self.last_failure_time > self.circuit_breaker_timeout:
            logger.info("Circuit breaker timeout expired, attempting to close circuit")
            self.circuit_open = False
            self.failure_count = 0
            return False
        
        return True
    
    def _record_success(self):
        """Record successful API call"""
        self.failure_count = 0
        self.circuit_open = False
    
    def _record_failure(self):
        """Record failed API call and update circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def _exponential_backoff_delay(self, attempt: int, is_rate_limit: bool = False) -> float:
        """Calculate exponential backoff delay"""
        if is_rate_limit:
            # Longer delay for rate limits
            delay = self.rate_limit_delay + (self.base_delay * (2 ** attempt))
        else:
            # Standard exponential backoff
            delay = self.base_delay * (2 ** attempt)
        
        # Cap the delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        final_delay = delay + jitter
        
        logger.info(f"Waiting {final_delay:.2f}s before retry (attempt {attempt + 1})")
        await asyncio.sleep(final_delay)
        return final_delay
    
    async def _make_instructor_call_with_retry(self, call_func, *args, **kwargs):
        """Make Instructor API call with retry logic and circuit breaker"""
        if self._is_circuit_breaker_open():
            raise Exception("Circuit breaker is open - too many recent failures")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if not self.client:
                    raise Exception("Instructor client not initialized")
                
                # Run the synchronous call in a thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, call_func, *args, **kwargs
                )
                self._record_success()
                return result
                
            except RateLimitError as e:
                last_exception = e
                logger.warning(f"Rate limit hit (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    await self._exponential_backoff_delay(attempt, is_rate_limit=True)
                else:
                    self._record_failure()
                    
            except (APIConnectionError, APIStatusError) as e:
                last_exception = e
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    await self._exponential_backoff_delay(attempt)
                else:
                    self._record_failure()
                    
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                self._record_failure()
                break  # Don't retry unexpected errors
        
        # If we get here, all retries failed
        raise last_exception
    
    async def classify_query(
        self, 
        query: str, 
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> QueryClassification:
        """
        Classify query using Instructor
        
        Args:
            query: User query to classify
            conversation_history: Previous conversation messages for context
            
        Returns:
            QueryClassification with structured analysis
        """
        try:
            if not self.client:
                # Fallback to simple classification
                return self._simple_classification(query, conversation_history)
            
            # Prepare context information
            context_info = ""
            if conversation_history:
                recent_messages = conversation_history[-2:]  # Last 2 messages for context
                context_info = "\n".join([
                    f"{msg.role}: {msg.content}" for msg in recent_messages
                ])
            
            # Simple classification prompt
            system_prompt = """Classify this query into one of three types:

1. EXTERNAL: Personal/social queries (greetings, personal info, general conversation)
2. CONTEXT_DEPENDENT: Queries that need conversation context (references to "this", "that", "the above")
3. DOCUMENT_BASED: Queries that need financial document search (direct questions about financial concepts)

Be concise and confident in your classification."""

            user_prompt = f"""Query: "{query}"

Context: {context_info if context_info else "No previous conversation"}

Classify this query."""

            # Generate structured classification with retry logic
            def make_classification_call():
                return self.client.chat.completions.create(
                    model="gpt-4",
                    response_model=QueryClassification,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent classification
                    timeout=self.classification_timeout  # Use configurable timeout
                )
            
            classification = await self._make_instructor_call_with_retry(make_classification_call)
            
            logger.info(f"Query classified as {classification.query_type.value} (confidence: {classification.confidence:.2f})")
            return classification
            
        except Exception as e:
            logger.warning(f"Instructor classification failed, using fallback: {e}")
            return self._simple_classification(query, conversation_history)
    
    def _simple_classification(self, query: str, conversation_history: Optional[List[ChatMessage]] = None) -> QueryClassification:
        """Simple fallback classification"""
        query_lower = query.lower()
        
        # Check for external queries
        if any(phrase in query_lower for phrase in ["my name is", "i am", "hello", "hi", "thank you", "how are you"]):
            return QueryClassification(
                query_type=QueryType.EXTERNAL,
                confidence=0.8,
                reasoning="Contains personal/social language"
            )
        
        # Check for context-dependent queries
        if any(phrase in query_lower for phrase in ["this", "that", "the above", "explain this", "tell me more about this"]):
            return QueryClassification(
                query_type=QueryType.CONTEXT_DEPENDENT,
                confidence=0.7,
                reasoning="References previous context"
            )
        
        # Default to document-based
        return QueryClassification(
            query_type=QueryType.DOCUMENT_BASED,
            confidence=0.6,
            reasoning="Default classification for financial queries"
        )
    
    async def enhance_financial_response(
        self, 
        original_answer: str, 
        query: str,
        sources: Optional[List] = None
    ) -> FinancialResponse:
        """
        Enhance financial response with structured output using Instructor
        
        Args:
            original_answer: The original response from RAG
            query: The user's query
            sources: List of sources used for the response
            
        Returns:
            FinancialResponse with enhanced structured information
        """
        try:
            if not self.client:
                # Fallback - return original answer with minimal structure
                return FinancialResponse(
                    answer=original_answer,
                    confidence=0.5,
                    formulas=[],
                    key_metrics=[],
                    follow_up_questions=[]
                )
            
            # Prepare source information
            source_info = ""
            if sources:
                source_types = [getattr(s, 'source_type', 'unknown') for s in sources]
                source_info = f"Sources used: {', '.join(set(source_types))}"
            
            system_prompt = """You are a financial AI assistant. Analyze the given response and extract structured financial information.

Extract:
1. Mathematical formulas (LaTeX format if possible)
2. Key financial metrics mentioned
3. Assess response confidence (0.0-1.0)
4. Generate relevant follow-up questions
5. Identify risk level if applicable
6. Identify time horizon if applicable
7. Generate MATLAB code if calculations are involved

Be precise and focus on financial/quantitative aspects."""

            user_prompt = f"""Query: "{query}"

Response: "{original_answer}"

{source_info}

Analyze this response and provide structured financial information."""

            # Generate structured financial response with retry logic
            def make_enhancement_call():
                return self.client.chat.completions.create(
                    model="gpt-4",
                    response_model=FinancialResponse,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    timeout=self.enhancement_timeout  # Use configurable timeout
                )
            
            financial_response = await self._make_instructor_call_with_retry(make_enhancement_call)
            
            logger.info(f"Financial response enhanced with {len(financial_response.formulas)} formulas, {len(financial_response.key_metrics)} metrics")
            return financial_response
            
        except Exception as e:
            logger.warning(f"Financial response enhancement failed: {e}")
            # Fallback - return original answer with minimal structure
            return FinancialResponse(
                answer=original_answer,
                confidence=0.5,
                formulas=[],
                key_metrics=[],
                follow_up_questions=[]
            )


    def get_status(self) -> dict:
        """Get current status of the instructor classifier"""
        return {
            "circuit_breaker_open": self.circuit_open,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "client_initialized": self.client is not None,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay
        }
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker"""
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        logger.info("Circuit breaker manually reset")


# Global instance
instructor_classifier = InstructorQueryClassifier()
