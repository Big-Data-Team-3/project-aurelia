"""
Session Management Service
Handles persistent conversation storage with TTL and context window management
Now uses Redis for persistence instead of in-memory storage
"""

import time
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from models.rag_models import ChatSession, ChatMessage
from services.cache_service import cache_service

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing chat sessions with Redis-based storage and TTL"""
    
    def __init__(self, default_ttl_hours: int = 24, max_context_messages: int = 20):
        self.default_ttl_hours = default_ttl_hours
        self.max_context_messages = max_context_messages
        # Keep user_sessions in memory for quick lookup (lightweight)
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> session_ids
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for session cleanup"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def create_session(self, user_id: str, ttl_hours: Optional[int] = None) -> ChatSession:
        """
        Create a new chat session and store it in Redis
        
        Args:
            user_id: User identifier
            ttl_hours: Time to live in hours (uses default if None)
            
        Returns:
            New ChatSession object
        """
        session_id = str(uuid.uuid4())
        ttl = ttl_hours or self.default_ttl_hours
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=ttl),
            max_context_messages=self.max_context_messages
        )
        
        # Store session in Redis
        ttl_seconds = ttl * 3600  # Convert hours to seconds
        success = await cache_service.cache_session(session_id, session, ttl_seconds)
        
        if not success:
            logger.error(f"Failed to cache session {session_id} in Redis")
            raise Exception("Failed to create session in Redis")
        
        # Update user sessions list (keep in memory for quick lookup)
        self.user_sessions[user_id].append(session_id)
        
        logger.info(f"Created new session {session_id} for user {user_id} in Redis")
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get session by ID from Redis
        
        Args:
            session_id: Session identifier
            
        Returns:
            ChatSession if found and not expired, None otherwise
        """
        try:
            session = await cache_service.get_cached_session(session_id)
            
            if not session:
                return None
            
            # Check if session is expired
            if datetime.now() > session.expires_at:
                await self.delete_session(session_id)
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Error retrieving session {session_id} from Redis: {e}")
            return None
    
    async def update_session(self, session_id: str, message: ChatMessage) -> Optional[ChatSession]:
        """
        Add a message to the session and update timestamp in Redis
        
        Args:
            session_id: Session identifier
            message: Message to add
            
        Returns:
            Updated ChatSession or None if session not found
        """
        session = await self.get_session(session_id)
        
        if not session:
            return None
        
        # Add message to session
        session.messages.append(message)
        session.updated_at = datetime.now()
        
        # Trim messages if exceeding context window
        if len(session.messages) > session.max_context_messages:
            # Keep the most recent messages
            session.messages = session.messages[-session.max_context_messages:]
            logger.info(f"Trimmed session {session_id} to {session.max_context_messages} messages")
        
        # Update session in Redis
        try:
            success = await cache_service.cache_session(session_id, session)
            if not success:
                logger.error(f"Failed to update session {session_id} in Redis")
                return None
        except Exception as e:
            logger.error(f"Error updating session {session_id} in Redis: {e}")
            return None
        
        logger.debug(f"Added message to session {session_id}, total messages: {len(session.messages)}")
        return session
    
    async def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[ChatSession]:
        """
        Get all sessions for a user from Redis
        
        Args:
            user_id: User identifier
            active_only: If True, only return non-expired sessions
            
        Returns:
            List of ChatSession objects
        """
        session_ids = self.user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids[:]:  # Copy list to avoid modification during iteration
            try:
                session = await cache_service.get_cached_session(session_id)
                
                if not session:
                    # Remove invalid session ID
                    session_ids.remove(session_id)
                    continue
                
                if active_only and datetime.now() > session.expires_at:
                    # Remove expired session
                    await self.delete_session(session_id)
                    continue
                
                sessions.append(session)
                
            except Exception as e:
                logger.error(f"Error retrieving session {session_id} for user {user_id}: {e}")
                # Remove problematic session ID
                session_ids.remove(session_id)
                continue
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from Redis
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        try:
            # Get session first to get user_id for cleanup
            session = await cache_service.get_cached_session(session_id)
            
            if not session:
                return False
            
            # Delete from Redis
            success = await cache_service.delete_session(session_id)
            
            if not success:
                logger.error(f"Failed to delete session {session_id} from Redis")
                return False
            
            # Remove from user sessions (in-memory tracking)
            user_id = session.user_id
            if user_id in self.user_sessions:
                try:
                    self.user_sessions[user_id].remove(session_id)
                    # Clean up empty user session lists
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                except ValueError:
                    pass  # Session ID not in list
            
            logger.info(f"Deleted session {session_id} from Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def extend_session_ttl(self, session_id: str, additional_hours: int = None) -> Optional[ChatSession]:
        """
        Extend session TTL in Redis
        
        Args:
            session_id: Session identifier
            additional_hours: Hours to add (uses default TTL if None)
            
        Returns:
            Updated ChatSession or None if not found
        """
        session = await self.get_session(session_id)
        
        if not session:
            return None
        
        hours_to_add = additional_hours or self.default_ttl_hours
        session.expires_at = datetime.now() + timedelta(hours=hours_to_add)
        session.updated_at = datetime.now()
        
        # Update session in Redis with new TTL
        try:
            success = await cache_service.cache_session(session_id, session)
            if not success:
                logger.error(f"Failed to extend TTL for session {session_id} in Redis")
                return None
        except Exception as e:
            logger.error(f"Error extending TTL for session {session_id}: {e}")
            return None
        
        logger.info(f"Extended session {session_id} TTL by {hours_to_add} hours in Redis")
        return session
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions from Redis
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        expired_sessions = []
        
        # Check all user sessions for expired ones
        for user_id, session_ids in self.user_sessions.items():
            for session_id in session_ids[:]:  # Copy to avoid modification during iteration
                try:
                    session = await cache_service.get_cached_session(session_id)
                    if not session or now > session.expires_at:
                        expired_sessions.append(session_id)
                except Exception as e:
                    logger.error(f"Error checking session {session_id} for expiration: {e}")
                    # Add to cleanup list if we can't check it
                    expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions from Redis")
        
        return len(expired_sessions)
    
    async def get_session_stats(self) -> Dict[str, int]:
        """
        Get session statistics from Redis
        
        Returns:
            Dictionary with session statistics
        """
        total_sessions = 0
        active_sessions = 0
        total_messages = 0
        
        now = datetime.now()
        
        # Count sessions from user_sessions tracking
        for user_id, session_ids in self.user_sessions.items():
            for session_id in session_ids:
                try:
                    session = await cache_service.get_cached_session(session_id)
                    if session:
                        total_sessions += 1
                        if now <= session.expires_at:
                            active_sessions += 1
                        total_messages += len(session.messages)
                except Exception as e:
                    logger.error(f"Error getting stats for session {session_id}: {e}")
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "expired_sessions": total_sessions - active_sessions,
            "total_messages": total_messages,
            "unique_users": len(self.user_sessions)
        }
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


# Global session service instance
session_service = SessionService()
