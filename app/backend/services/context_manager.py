"""
Context Management Service
Handles conversation summarization, context compression, and smart context windows
"""

import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from models.rag_models import ChatMessage, ChatSession
from services.generation import generation_service
from config.rag_config import rag_config

logger = logging.getLogger(__name__)


@dataclass
class ContextSummary:
    """Represents a summarized context"""
    summary: str
    key_topics: List[str]
    important_entities: List[str]
    conversation_tone: str
    created_at: datetime
    original_message_count: int
    compressed_message_count: int


@dataclass
class ContextWindow:
    """Represents a context window with metadata"""
    messages: List[ChatMessage]
    summary: Optional[ContextSummary]
    window_type: str  # 'recent', 'summarized', 'compressed'
    token_count: int
    created_at: datetime


class ContextManager:
    """Advanced context management with summarization and compression"""
    
    def __init__(self):
        self.max_context_messages = rag_config.session_max_context_messages or 20
        self.summarization_threshold = 15  # Summarize when more than 15 messages
        self.compression_threshold = 30    # Compress when more than 30 messages
        self.max_tokens_per_message = 200  # Approximate tokens per message
        self.target_context_tokens = 4000  # Target context size in tokens
        
    async def get_optimized_context(
        self, 
        session: ChatSession, 
        current_query: str = None
    ) -> ContextWindow:
        """
        Get optimized context for the current conversation
        
        Args:
            session: Chat session with message history
            current_query: Current user query for context relevance
            
        Returns:
            Optimized context window
        """
        start_time = time.time()
        
        try:
            messages = session.messages.copy()
            message_count = len(messages)
            
            # Determine context strategy based on message count
            if message_count <= self.max_context_messages:
                # Use recent messages strategy
                context_window = await self._create_recent_context(messages)
            elif message_count <= self.summarization_threshold:
                # Use sliding window with some summarization
                context_window = await self._create_sliding_context(messages, current_query)
            elif message_count <= self.compression_threshold:
                # Use summarization strategy
                context_window = await self._create_summarized_context(messages, current_query)
            else:
                # Use compression strategy
                context_window = await self._create_compressed_context(messages, current_query)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Context optimization completed in {processing_time:.2f}ms - "
                       f"Strategy: {context_window.window_type}, "
                       f"Messages: {len(context_window.messages)}, "
                       f"Tokens: ~{context_window.token_count}")
            
            return context_window
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            # Fallback to recent messages
            return await self._create_recent_context(session.messages[-self.max_context_messages:])
    
    async def _create_recent_context(self, messages: List[ChatMessage]) -> ContextWindow:
        """Create context from recent messages"""
        recent_messages = messages[-self.max_context_messages:]
        token_count = self._estimate_token_count(recent_messages)
        
        return ContextWindow(
            messages=recent_messages,
            summary=None,
            window_type="recent",
            token_count=token_count,
            created_at=datetime.now()
        )
    
    async def _create_sliding_context(
        self, 
        messages: List[ChatMessage], 
        current_query: str = None
    ) -> ContextWindow:
        """Create context with sliding window and light summarization"""
        
        # Keep recent messages (last 10)
        recent_messages = messages[-10:]
        
        # Summarize older messages if needed
        older_messages = messages[:-10]
        if len(older_messages) > 5:
            summary = await self._summarize_messages(older_messages, "brief")
            
            # Create summary message
            summary_message = ChatMessage(
                role="system",
                content=f"[Previous conversation summary: {summary.summary}]",
                timestamp=datetime.now()
            )
            
            # Combine summary with recent messages
            context_messages = [summary_message] + recent_messages
        else:
            context_messages = recent_messages
        
        token_count = self._estimate_token_count(context_messages)
        
        return ContextWindow(
            messages=context_messages,
            summary=None,
            window_type="sliding",
            token_count=token_count,
            created_at=datetime.now()
        )
    
    async def _create_summarized_context(
        self, 
        messages: List[ChatMessage], 
        current_query: str = None
    ) -> ContextWindow:
        """Create context with conversation summarization"""
        
        # Keep very recent messages (last 5)
        recent_messages = messages[-5:]
        
        # Summarize the rest
        older_messages = messages[:-5]
        summary = await self._summarize_messages(older_messages, "detailed", current_query)
        
        # Create comprehensive summary message
        summary_content = self._format_summary_for_context(summary)
        summary_message = ChatMessage(
            role="system",
            content=summary_content,
            timestamp=datetime.now()
        )
        
        # Combine summary with recent messages
        context_messages = [summary_message] + recent_messages
        token_count = self._estimate_token_count(context_messages)
        
        return ContextWindow(
            messages=context_messages,
            summary=summary,
            window_type="summarized",
            token_count=token_count,
            created_at=datetime.now()
        )
    
    async def _create_compressed_context(
        self, 
        messages: List[ChatMessage], 
        current_query: str = None
    ) -> ContextWindow:
        """Create highly compressed context for very long conversations"""
        
        # Keep only the most recent messages (last 3)
        recent_messages = messages[-3:]
        
        # Create multiple levels of summarization
        older_messages = messages[:-3]
        
        # First level: Summarize in chunks
        chunk_size = 10
        chunk_summaries = []
        
        for i in range(0, len(older_messages), chunk_size):
            chunk = older_messages[i:i + chunk_size]
            if len(chunk) >= 3:  # Only summarize meaningful chunks
                chunk_summary = await self._summarize_messages(chunk, "brief")
                chunk_summaries.append(chunk_summary)
        
        # Second level: Summarize the chunk summaries
        if chunk_summaries:
            # Create a meta-summary from chunk summaries
            meta_summary = await self._create_meta_summary(chunk_summaries, current_query)
            
            # Create compressed context message
            compressed_content = self._format_compressed_context(meta_summary, chunk_summaries)
            compressed_message = ChatMessage(
                role="system",
                content=compressed_content,
                timestamp=datetime.now()
            )
            
            context_messages = [compressed_message] + recent_messages
        else:
            context_messages = recent_messages
        
        token_count = self._estimate_token_count(context_messages)
        
        return ContextWindow(
            messages=context_messages,
            summary=None,  # Meta-summary is embedded in the message
            window_type="compressed",
            token_count=token_count,
            created_at=datetime.now()
        )
    
    async def _summarize_messages(
        self, 
        messages: List[ChatMessage], 
        detail_level: str = "detailed",
        current_query: str = None
    ) -> ContextSummary:
        """Summarize a list of messages using LLM"""
        
        try:
            # Prepare conversation text
            conversation_text = self._format_messages_for_summarization(messages)
            
            # Create summarization prompt
            if detail_level == "brief":
                prompt = self._create_brief_summary_prompt(conversation_text, current_query)
            else:
                prompt = self._create_detailed_summary_prompt(conversation_text, current_query)
            
            # Generate summary using LLM
            summary_response, _ = await generation_service.generate_response(
                query=prompt,
                temperature=0.1,  # Low temperature for consistent summarization
                max_tokens=500
            )
            
            # Parse summary response
            summary_data = self._parse_summary_response(summary_response)
            
            return ContextSummary(
                summary=summary_data.get("summary", ""),
                key_topics=summary_data.get("topics", []),
                important_entities=summary_data.get("entities", []),
                conversation_tone=summary_data.get("tone", "neutral"),
                created_at=datetime.now(),
                original_message_count=len(messages),
                compressed_message_count=1
            )
            
        except Exception as e:
            logger.error(f"Message summarization failed: {e}")
            # Fallback to simple concatenation
            return ContextSummary(
                summary=f"Previous conversation with {len(messages)} messages",
                key_topics=[],
                important_entities=[],
                conversation_tone="neutral",
                created_at=datetime.now(),
                original_message_count=len(messages),
                compressed_message_count=1
            )
    
    async def _create_meta_summary(
        self, 
        chunk_summaries: List[ContextSummary], 
        current_query: str = None
    ) -> ContextSummary:
        """Create a meta-summary from multiple chunk summaries"""
        
        try:
            # Combine chunk summaries
            combined_summaries = "\n\n".join([
                f"Chunk {i+1}: {summary.summary}" 
                for i, summary in enumerate(chunk_summaries)
            ])
            
            # Create meta-summarization prompt
            prompt = f"""
            Create a comprehensive meta-summary of these conversation chunks:
            
            {combined_summaries}
            
            Current query context: {current_query or "No specific context"}
            
            Provide a structured summary with:
            1. Overall conversation theme
            2. Key topics discussed
            3. Important entities mentioned
            4. Conversation tone
            """
            
            # Generate meta-summary
            meta_response, _ = await generation_service.generate_response(
                query=prompt,
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse meta-summary
            meta_data = self._parse_summary_response(meta_response)
            
            total_messages = sum(cs.original_message_count for cs in chunk_summaries)
            
            return ContextSummary(
                summary=meta_data.get("summary", ""),
                key_topics=meta_data.get("topics", []),
                important_entities=meta_data.get("entities", []),
                conversation_tone=meta_data.get("tone", "neutral"),
                created_at=datetime.now(),
                original_message_count=total_messages,
                compressed_message_count=1
            )
            
        except Exception as e:
            logger.error(f"Meta-summarization failed: {e}")
            # Fallback
            return ContextSummary(
                summary=f"Long conversation with {sum(cs.original_message_count for cs in chunk_summaries)} messages",
                key_topics=[],
                important_entities=[],
                conversation_tone="neutral",
                created_at=datetime.now(),
                original_message_count=sum(cs.original_message_count for cs in chunk_summaries),
                compressed_message_count=1
            )
    
    def _format_messages_for_summarization(self, messages: List[ChatMessage]) -> str:
        """Format messages for summarization"""
        formatted = []
        for msg in messages:
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted)
    
    def _create_brief_summary_prompt(self, conversation_text: str, current_query: str = None) -> str:
        """Create prompt for brief summarization"""
        return f"""
        Summarize this conversation briefly (2-3 sentences):
        
        {conversation_text}
        
        Current context: {current_query or "General conversation"}
        
        Focus on the main topics and key information.
        """
    
    def _create_detailed_summary_prompt(self, conversation_text: str, current_query: str = None) -> str:
        """Create prompt for detailed summarization"""
        return f"""
        Create a detailed summary of this conversation:
        
        {conversation_text}
        
        Current context: {current_query or "General conversation"}
        
        Provide:
        1. Summary: Main topics and key information discussed
        2. Topics: List of key topics (comma-separated)
        3. Entities: Important entities mentioned (comma-separated)
        4. Tone: Overall conversation tone (formal, casual, technical, etc.)
        
        Format as JSON-like structure.
        """
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM summary response into structured data"""
        try:
            # Simple parsing - in production, use more robust parsing
            lines = response.strip().split('\n')
            result = {
                "summary": "",
                "topics": [],
                "entities": [],
                "tone": "neutral"
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.lower().startswith('summary:'):
                    result["summary"] = line[8:].strip()
                elif line.lower().startswith('topics:'):
                    topics_text = line[7:].strip()
                    result["topics"] = [t.strip() for t in topics_text.split(',') if t.strip()]
                elif line.lower().startswith('entities:'):
                    entities_text = line[9:].strip()
                    result["entities"] = [e.strip() for e in entities_text.split(',') if e.strip()]
                elif line.lower().startswith('tone:'):
                    result["tone"] = line[5:].strip()
                elif not result["summary"] and not any(result.values()):
                    # If no structured format, use the whole response as summary
                    result["summary"] = response.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Summary parsing failed: {e}")
            return {
                "summary": response.strip(),
                "topics": [],
                "entities": [],
                "tone": "neutral"
            }
    
    def _format_summary_for_context(self, summary: ContextSummary) -> str:
        """Format summary for inclusion in context"""
        return f"""[Conversation Summary]
Previous discussion: {summary.summary}
Key topics: {', '.join(summary.key_topics) if summary.key_topics else 'None'}
Important entities: {', '.join(summary.important_entities) if summary.important_entities else 'None'}
Tone: {summary.conversation_tone}
(Summarized from {summary.original_message_count} messages)"""
    
    def _format_compressed_context(
        self, 
        meta_summary: ContextSummary, 
        chunk_summaries: List[ContextSummary]
    ) -> str:
        """Format compressed context for very long conversations"""
        return f"""[Compressed Conversation Context]
Overall theme: {meta_summary.summary}
Key topics: {', '.join(meta_summary.key_topics) if meta_summary.key_topics else 'None'}
Important entities: {', '.join(meta_summary.important_entities) if meta_summary.important_entities else 'None'}
Tone: {meta_summary.conversation_tone}
(Compressed from {meta_summary.original_message_count} messages across {len(chunk_summaries)} conversation segments)"""
    
    def _estimate_token_count(self, messages: List[ChatMessage]) -> int:
        """Estimate token count for messages"""
        total_chars = sum(len(msg.content) for msg in messages)
        # Rough estimation: 1 token ≈ 4 characters
        return int(total_chars / 4)
    
    async def get_context_analytics(self, session: ChatSession) -> Dict[str, Any]:
        """Get analytics about conversation context"""
        messages = session.messages
        message_count = len(messages)
        
        # Calculate basic metrics
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        # Calculate average message length
        avg_user_length = sum(len(m.content) for m in user_messages) / len(user_messages) if user_messages else 0
        avg_assistant_length = sum(len(m.content) for m in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        
        # Estimate total tokens
        total_tokens = self._estimate_token_count(messages)
        
        # Determine context strategy that would be used
        if message_count <= self.max_context_messages:
            strategy = "recent"
        elif message_count <= self.summarization_threshold:
            strategy = "sliding"
        elif message_count <= self.compression_threshold:
            strategy = "summarized"
        else:
            strategy = "compressed"
        
        return {
            "total_messages": message_count,
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "average_user_message_length": avg_user_length,
            "average_assistant_message_length": avg_assistant_length,
            "estimated_total_tokens": total_tokens,
            "context_strategy": strategy,
            "session_age_hours": (datetime.now() - session.created_at).total_seconds() / 3600,
            "last_activity_hours": (datetime.now() - session.updated_at).total_seconds() / 3600
        }


# Global context manager instance
context_manager = ContextManager()
