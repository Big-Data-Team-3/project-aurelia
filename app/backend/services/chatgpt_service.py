"""
Direct ChatGPT Service
Handles direct OpenAI API calls for non-RAG queries with conversation awareness
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

from config.rag_config import rag_config
from models.rag_models import ChatMessage

logger = logging.getLogger(__name__)


class ChatGPTService:
    """Service for direct ChatGPT interactions without RAG"""
    
    def __init__(self, config=None):
        self.config = config or rag_config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("ChatGPT service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT service: {e}")
            raise
    
    async def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> Tuple[str, float]:
        """
        Generate a response using direct ChatGPT API
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
            
        Returns:
            Tuple of (response text, generation time in ms)
        """
        start_time = time.time()
        
        try:
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            
            # Build conversation messages
            messages = self._build_messages(query, conversation_history, system_prompt)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                timeout=self.config.request_timeout
            )
            
            generated_text = response.choices[0].message.content
            generation_time = (time.time() - start_time) * 1000
            
            logger.debug(f"ChatGPT response generated in {generation_time:.2f}ms")
            return generated_text, generation_time
            
        except Exception as e:
            logger.error(f"ChatGPT generation failed: {e}")
            generation_time = (time.time() - start_time) * 1000
            return f"I apologize, but I'm having trouble generating a response right now. Please try again.", generation_time
    
    def _build_messages(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Build message list for OpenAI API
        
        Args:
            query: Current user query
            conversation_history: Previous conversation messages
            system_prompt: Custom system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            system_content = system_prompt
        else:
            system_content = self._get_default_system_prompt()
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (limit to avoid token limits)
        if conversation_history:
            # Keep last 15 messages to stay within token limits
            recent_history = conversation_history[-15:] if len(conversation_history) > 15 else conversation_history
            
            for msg in recent_history:
                # Only add user and assistant messages, skip system messages
                if msg.role in ["user", "assistant"]:
                    messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for ChatGPT interactions"""
        return """You are a helpful AI assistant. You provide thoughtful, accurate, and engaging responses to user queries. 

Key guidelines:
- Be conversational and friendly while maintaining professionalism
- Provide clear and concise answers
- Ask clarifying questions when needed
- Admit when you don't know something rather than guessing
- Be helpful and try to understand the user's intent
- Maintain context from the conversation history
- If asked about specific documents or data you don't have access to, politely explain your limitations

You are having a natural conversation with the user. Respond appropriately to their message."""
    
    async def generate_streaming_response(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ):
        """
        Generate a streaming response using ChatGPT API
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
            
        Yields:
            Response chunks as they are generated
        """
        try:
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            
            # Build conversation messages
            messages = self._build_messages(query, conversation_history, system_prompt)
            
            # Generate streaming response
            stream = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                timeout=self.config.request_timeout,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"ChatGPT streaming generation failed: {e}")
            yield f"I apologize, but I'm having trouble generating a response right now. Please try again."
    
    async def classify_intent_with_llm(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """
        Use ChatGPT to classify user intent (as a backup to rule-based classification)
        
        Args:
            query: User query to classify
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with intent classification results
        """
        try:
            classification_prompt = f"""Analyze the following user query and classify its intent. Consider the conversation context if provided.

Query: "{query}"

Classify the intent as one of:
1. FACTUAL_QUERY - Questions seeking specific information or facts
2. CONVERSATIONAL - General chat, opinions, or casual conversation
3. GREETING - Greetings, pleasantries, or social interactions
4. CLARIFICATION - Follow-up questions or requests for more details
5. TASK_ORIENTED - Requests to perform specific tasks or actions

Also determine if this query would benefit from document-based information (RAG) or general knowledge (ChatGPT).

Respond in this exact format:
INTENT: [intent_type]
CONFIDENCE: [0.0-1.0]
USE_RAG: [true/false]
REASONING: [brief explanation]"""

            messages = [
                {"role": "system", "content": "You are an expert at classifying user intents and routing queries."},
                {"role": "user", "content": classification_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
                timeout=self.config.request_timeout
            )
            
            result_text = response.choices[0].message.content
            
            # Parse the response
            intent = "CONVERSATIONAL"  # Default
            confidence = 0.5
            use_rag = False
            reasoning = "LLM classification"
            
            for line in result_text.split('\n'):
                line = line.strip()
                if line.startswith('INTENT:'):
                    intent = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith('USE_RAG:'):
                    use_rag = line.split(':', 1)[1].strip().lower() == 'true'
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return {
                "predicted_intent": intent,
                "confidence_score": confidence,
                "should_use_rag": use_rag,
                "reasoning": f"LLM classification: {reasoning}",
                "method": "llm"
            }
            
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            return {
                "predicted_intent": "CONVERSATIONAL",
                "confidence_score": 0.3,
                "should_use_rag": False,
                "reasoning": "Fallback due to LLM classification error",
                "method": "fallback"
            }
    
    async def summarize_conversation(
        self,
        conversation_history: List[ChatMessage],
        max_summary_length: int = 500
    ) -> str:
        """
        Generate a summary of conversation history for context management
        
        Args:
            conversation_history: List of conversation messages
            max_summary_length: Maximum length of summary
            
        Returns:
            Conversation summary
        """
        try:
            if not conversation_history:
                return ""
            
            # Format conversation for summarization
            conversation_text = ""
            for msg in conversation_history:
                role = "User" if msg.role == "user" else "Assistant"
                conversation_text += f"{role}: {msg.content}\n"
            
            summary_prompt = f"""Summarize the following conversation, focusing on key topics, decisions, and context that would be important for continuing the conversation:

{conversation_text}

Provide a concise summary in {max_summary_length} characters or less."""

            messages = [
                {"role": "system", "content": "You are an expert at summarizing conversations while preserving important context."},
                {"role": "user", "content": summary_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_summary_length // 4,  # Rough token estimate
                timeout=self.config.request_timeout
            )
            
            summary = response.choices[0].message.content
            logger.debug(f"Generated conversation summary: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return "Previous conversation context available but summarization failed."
    
    async def health_check(self) -> bool:
        """
        Check if ChatGPT service is healthy
        
        Returns:
            True if service is healthy
        """
        try:
            test_response, _ = await self.generate_response("Hello, this is a test.")
            return len(test_response) > 0 and "test" not in test_response.lower()
        except Exception as e:
            logger.error(f"ChatGPT service health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ChatGPT model being used
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.config.generation_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.request_timeout
        }


# Global ChatGPT service instance
chatgpt_service = ChatGPTService()
