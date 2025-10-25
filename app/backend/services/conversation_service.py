"""
Conversation Management Service
Handles persistent conversation storage in CloudSQL
Replaces Redis-based session management
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from models import Conversation, Message, User
from models.rag_models import ChatMessage, SearchResult

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations with CloudSQL persistence"""
    
    def __init__(self):
        self.max_context_messages = 20  # Maximum messages to keep in context
    
    def create_conversation(
        self, 
        db: Session, 
        user_id: int, 
        title: str = None, 
        conversation_type: str = 'general'
    ) -> Conversation:
        """
        Create a new conversation
        
        Args:
            db: Database session
            user_id: User identifier
            title: Conversation title (auto-generated if None)
            conversation_type: Type of conversation ('general' or 'rag_query')
            
        Returns:
            New Conversation object
        """
        try:
            # Auto-generate title if not provided
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            conversation = Conversation(
                user_id=user_id,
                title=title,
                conversation_type=conversation_type,
                status='active'
            )
            
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            
            logger.info(f"Created new conversation {conversation.id} for user {user_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            db.rollback()
            raise
    
    def get_conversation(self, db: Session, conversation_id: int, user_id: int) -> Optional[Conversation]:
        """
        Get conversation by ID for a specific user
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            
        Returns:
            Conversation object or None if not found
        """
        try:
            conversation = db.query(Conversation).filter(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                    Conversation.status == 'active'
                )
            ).first()
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    def get_user_conversations(
        self, 
        db: Session, 
        user_id: int, 
        limit: int = 50,
        conversation_type: str = None
    ) -> List[Conversation]:
        """
        Get all conversations for a user
        
        Args:
            db: Database session
            user_id: User identifier
            limit: Maximum number of conversations to return
            conversation_type: Filter by conversation type
            
        Returns:
            List of Conversation objects ordered by created_at desc
        """
        try:
            query = db.query(Conversation).filter(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.status == 'active'
                )
            )
            
            if conversation_type:
                query = query.filter(Conversation.conversation_type == conversation_type)
            
            conversations = query.order_by(desc(Conversation.created_at)).limit(limit).all()
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving conversations for user {user_id}: {e}")
            return []
    
    def add_message(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int, 
        role: str, 
        content: str,
        message_metadata: dict = None
    ) -> Optional[Message]:
        """
        Add a message to a conversation
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Additional metadata (e.g., search results)
            
        Returns:
            Message object or None if conversation not found
        """
        try:
            # Verify conversation exists and belongs to user
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
                return None
            
            message = Message(
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                message_metadata=message_metadata
            )
            
            db.add(message)
            db.commit()
            db.refresh(message)
            
            logger.debug(f"Added {role} message to conversation {conversation_id}")
            return message
            
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            db.rollback()
            return None
    
    def get_conversation_messages(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int,
        limit: int = None
    ) -> List[Message]:
        """
        Get all messages for a conversation
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of Message objects ordered by created_at asc
        """
        try:
            # Verify conversation exists and belongs to user
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return []
            
            query = db.query(Message).filter(
                and_(
                    Message.conversation_id == conversation_id,
                    Message.user_id == user_id
                )
            ).order_by(Message.created_at.asc())
            
            if limit:
                query = query.limit(limit)
            
            messages = query.all()
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages for conversation {conversation_id}: {e}")
            return []
    
    def get_conversation_context(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int
    ) -> List[ChatMessage]:
        """
        Get conversation messages formatted for RAG context
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            
        Returns:
            List of ChatMessage objects for RAG context
        """
        try:
            messages = self.get_conversation_messages(
                db, conversation_id, user_id, limit=self.max_context_messages
            )
            
            # Convert to ChatMessage format
            chat_messages = []
            for msg in messages:
                # Extract search results from message_metadata if available
                sources = None
                if msg.message_metadata and 'search_results' in msg.message_metadata:
                    sources = [
                        SearchResult(**result) for result in msg.message_metadata['search_results']
                    ]
                
                chat_message = ChatMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at,
                    sources=sources
                )
                chat_messages.append(chat_message)
            
            return chat_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation context for {conversation_id}: {e}")
            return []
    
    def update_conversation_title(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int, 
        title: str
    ) -> bool:
        """
        Update conversation title
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            title: New title
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.title = title
            db.commit()
            
            logger.info(f"Updated title for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating conversation title: {e}")
            db.rollback()
            return False
    
    def delete_conversation(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int
    ) -> bool:
        """
        Soft delete a conversation (set status to 'deleted')
        
        Args:
            db: Database session
            conversation_id: Conversation identifier
            user_id: User identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conversation = self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.status = 'deleted'
            db.commit()
            
            logger.info(f"Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            db.rollback()
            return False
    
    def get_conversation_stats(self, db: Session, user_id: int) -> dict:
        """
        Get conversation statistics for a user
        
        Args:
            db: Database session
            user_id: User identifier
            
        Returns:
            Dictionary with conversation statistics
        """
        try:
            total_conversations = db.query(Conversation).filter(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.status == 'active'
                )
            ).count()
            
            total_messages = db.query(Message).join(Conversation).filter(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.status == 'active'
                )
            ).count()
            
            # Get conversation types breakdown
            type_counts = db.query(
                Conversation.conversation_type,
                db.func.count(Conversation.id)
            ).filter(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.status == 'active'
                )
            ).group_by(Conversation.conversation_type).all()
            
            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'conversation_types': dict(type_counts),
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation stats for user {user_id}: {e}")
            return {
                'total_conversations': 0,
                'total_messages': 0,
                'conversation_types': {},
                'user_id': user_id
            }


# Global instance
conversation_service = ConversationService()
