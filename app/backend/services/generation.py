"""
Generation Service using OpenAI
Handles text generation with context and prompt engineering
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import OpenAI
import logging

from config.rag_config import rag_config, PromptTemplates
from models.rag_models import SearchResult, ChatMessage


logger = logging.getLogger(__name__)


class GenerationService:
    """Service for text generation using OpenAI models"""
    
    def __init__(self, config=None):
        self.config = config or rag_config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI generation client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context string"""
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Add source information
            source_info = f"Source {i}"
            if result.section:
                source_info += f" (Section: {result.section})"
            if result.pages:
                source_info += f" (Pages: {', '.join(map(str, result.pages))})"
            
            context_part = f"{source_info}:\n{result.text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def generate_response(
        self,
        query: str,
        context_results: List[SearchResult],
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> tuple[str, float]:
        """
        Generate response using context and query
        
        Args:
            query: User query
            context_results: Search results to use as context
            temperature: Generation temperature (overrides config)
            max_tokens: Maximum tokens (overrides config)
            system_prompt: Custom system prompt (overrides default)
            
        Returns:
            Tuple of (generated response, generation time in ms)
        """
        start_time = time.time()
        
        try:
            # Use provided values or fall back to config
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            sys_prompt = system_prompt if system_prompt else PromptTemplates.RAG_SYSTEM_PROMPT
            
            # Format context
            context = self._format_context(context_results)
            
            # Create user prompt
            user_prompt = PromptTemplates.RAG_USER_PROMPT.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                timeout=self.config.request_timeout
            )
            
            generated_text = response.choices[0].message.content
            generation_time = (time.time() - start_time) * 1000
            
            logger.info(f"Response generated in {generation_time:.2f}ms")
            
            return generated_text, generation_time
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_with_conversation_history(
        self,
        query: str,
        context_results: List[SearchResult],
        conversation_history: List[ChatMessage],
        temperature: float = None,
        max_tokens: int = None
    ) -> tuple[str, float]:
        """
        Generate response with conversation history context
        
        Args:
            query: Current user query
            context_results: Retrieved context from RAG
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            Tuple of (generated_text, generation_time_ms)
        """
        start_time = time.time()
        
        try:
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            
            # Format context from search results
            context = self._format_context(context_results)
            
            # Build conversation-aware prompt
            conversation_context = ""
            if conversation_history:
                recent_messages = conversation_history[-5:]
                context_parts = []
                for msg in recent_messages:
                    if hasattr(msg, 'role'):
                        role, content = msg.role, msg.content
                    else:
                        role, content = msg.get('role', 'user'), msg.get('content', '')
                    context_parts.append(f"{role.title()}: {content}")
                conversation_context = "\n".join(context_parts)
                logger.debug(f"Built conversation context with {len(recent_messages)} messages")
            # Enhanced prompt with conversation awareness
            else:
                logger.debug("No conversation history available.")
            system_prompt = f"""You are a financial analysis assistant with access to a comprehensive knowledge base.

    Context from documents:
    {context}

    Recent conversation:
    {conversation_context}

    Instructions:
    - Use the provided context to answer the user's question accurately
    - Reference the conversation history to maintain continuity
    - If the context doesn't contain relevant information, say so clearly
    - Provide specific examples and page references when available
    - Maintain the conversational flow from previous messages"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                timeout=self.config.request_timeout
            )
            
            generated_text = response.choices[0].message.content
            generation_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Generated response with conversation history in {generation_time:.2f}ms")
            return generated_text, generation_time
            
        except Exception as e:
            logger.error(f"Generation with conversation history failed: {e}")
            generation_time = (time.time() - start_time) * 1000
            return "I apologize, but I'm having trouble generating a response right now. Please try again.", generation_time

    async def generate_streaming_response(
        self,
        query: str,
        context_results: List[SearchResult],
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time chat
        
        Args:
            query: User query
            context_results: Search results to use as context
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: Custom system prompt
            
        Yields:
            Text chunks as they are generated
        """
        try:
            # Use provided values or fall back to config
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            sys_prompt = system_prompt if system_prompt else PromptTemplates.RAG_SYSTEM_PROMPT
            
            # Format context
            context = self._format_context(context_results)
            
            # Create user prompt
            user_prompt = PromptTemplates.RAG_USER_PROMPT.format(
                context=context,
                question=query
            )
            
            # Generate streaming response
            stream = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok,
                stream=True,
                timeout=self.config.request_timeout
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error generating response: {str(e)}"
    
    async def generate_response_with_conversation_history(
        self,
        query: str,
        context_results: List[SearchResult],
        conversation_history: List[ChatMessage],
        temperature: float = None,
        max_tokens: int = None
    ) -> tuple[str, float]:
        """
        Generate response considering conversation history
        
        Args:
            query: Current user query
            context_results: Search results for current query
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            Tuple of (generated response, generation time in ms)
        """
        start_time = time.time()
        
        try:
            temp = temperature if temperature is not None else self.config.temperature
            max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": PromptTemplates.RAG_SYSTEM_PROMPT}]
            
            # Add conversation history (limit to last 10 messages to avoid token limit)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Format context for current query
            context = self._format_context(context_results)
            
            # Add current query with context
            current_prompt = PromptTemplates.RAG_USER_PROMPT.format(
                context=context,
                question=query
            )
            messages.append({"role": "user", "content": current_prompt})
            
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
            
            logger.info(f"Conversational response generated in {generation_time:.2f}ms")
            
            return generated_text, generation_time
            
        except Exception as e:
            logger.error(f"Conversational generation failed: {e}")
            raise
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative query formulations for better retrieval
        
        Args:
            query: Original query
            
        Returns:
            List of alternative query formulations
        """
        try:
            prompt = PromptTemplates.QUERY_EXPANSION_PROMPT.format(query=query)
            
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates alternative query formulations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                timeout=self.config.request_timeout
            )
            
            expanded_text = response.choices[0].message.content
            
            # Parse the response to extract alternative queries
            alternatives = []
            for line in expanded_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Alternative queries:'):
                    # Remove numbering and bullet points
                    clean_line = line.lstrip('0123456789.- ')
                    if clean_line and clean_line != query:
                        alternatives.append(clean_line)
            
            logger.info(f"Generated {len(alternatives)} alternative queries")
            return alternatives
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    async def assess_context_relevance(
        self,
        query: str,
        context: str,
        threshold: float = 5.0
    ) -> tuple[bool, float]:
        """
        Assess if context is relevant to the query
        
        Args:
            query: User query
            context: Context text to assess
            threshold: Relevance threshold (1-10 scale)
            
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        try:
            prompt = PromptTemplates.CONTEXT_RELEVANCE_PROMPT.format(
                question=query,
                context=context[:1000]  # Limit context length
            )
            
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {"role": "system", "content": "You are an expert at assessing text relevance. Respond only with a number from 1-10."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10,
                timeout=self.config.request_timeout
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract numeric score
            try:
                score = float(score_text)
                score = max(1.0, min(10.0, score))  # Clamp to 1-10 range
                is_relevant = score >= threshold
                
                return is_relevant, score
                
            except ValueError:
                logger.warning(f"Could not parse relevance score: {score_text}")
                return False, 1.0
                
        except Exception as e:
            logger.error(f"Context relevance assessment failed: {e}")
            return False, 1.0
    
    async def health_check(self) -> bool:
        """Check if the generation service is healthy"""
        try:
            # Test with a simple generation
            response = self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {"role": "user", "content": "Say 'OK' if you can respond."}
                ],
                temperature=0.1,
                max_tokens=10,
                timeout=5
            )
            
            return response.choices[0].message.content.strip().upper() == "OK"
            
        except Exception as e:
            logger.error(f"Generation service health check failed: {e}")
            return False


# Global instance
generation_service = GenerationService()
