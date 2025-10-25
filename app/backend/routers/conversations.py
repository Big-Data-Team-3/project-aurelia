"""
Conversation Management API Endpoints
Handles conversation CRUD operations and message management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from config.database import get_db
from models import Conversation, Message, User, AuthToken
from services.conversation_service import conversation_service
from routers.auth import get_current_user_from_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["Conversations"])


# Pydantic models for request/response
from pydantic import BaseModel

class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    conversation_type: str = 'general'

class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str
    conversation_type: str
    status: str
    created_at: datetime
    message_count: int = 0

class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    user_id: int
    role: str
    content: str
    metadata: Optional[dict] = None
    created_at: datetime

class AddMessageRequest(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    message_metadata: Optional[dict] = None

class ConversationHistoryResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]

class ConversationStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    conversation_types: dict
    user_id: int


# Authentication dependency
async def get_current_user(credentials = Depends(get_current_user_from_token), db: Session = Depends(get_db)):
    """Get current authenticated user"""
    return credentials


# Conversation Management Endpoints

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new conversation
    
    Args:
        request: Conversation creation request
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created conversation details
    """
    try:
        user_id = current_user["user"]["id"]
        
        conversation = conversation_service.create_conversation(
            db=db,
            user_id=user_id,
            title=request.title,
            conversation_type=request.conversation_type
        )
        
        return ConversationResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            conversation_type=conversation.conversation_type,
            status=conversation.status,
            created_at=conversation.created_at,
            message_count=0
        )
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/", response_model=List[ConversationResponse])
async def get_user_conversations(
    conversation_type: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all conversations for the current user
    
    Args:
        conversation_type: Filter by conversation type
        limit: Maximum number of conversations to return
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of user conversations
    """
    try:
        user_id = current_user["user"]["id"]
        
        conversations = conversation_service.get_user_conversations(
            db=db,
            user_id=user_id,
            limit=limit,
            conversation_type=conversation_type
        )
        
        # Get message counts for each conversation
        conversation_responses = []
        for conv in conversations:
            message_count = db.query(Message).filter(
                Message.conversation_id == conv.id
            ).count()
            
            conversation_responses.append(ConversationResponse(
                id=conv.id,
                user_id=conv.user_id,
                title=conv.title,
                conversation_type=conv.conversation_type,
                status=conv.status,
                created_at=conv.created_at,
                message_count=message_count
            ))
        
        return conversation_responses
        
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get conversation with full message history
    
    Args:
        conversation_id: Conversation identifier
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Conversation with all messages
    """
    try:
        user_id = current_user["user"]["id"]
        
        # Get conversation
        conversation = conversation_service.get_conversation(db, conversation_id, user_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Get messages
        messages = conversation_service.get_conversation_messages(
            db, conversation_id, user_id
        )
        
        # Format response
        conversation_response = ConversationResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            conversation_type=conversation.conversation_type,
            status=conversation.status,
            created_at=conversation.created_at,
            message_count=len(messages)
        )
        
        message_responses = [
            MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                user_id=msg.user_id,
                role=msg.role,
                content=msg.content,
                metadata=msg.message_metadata,
                created_at=msg.created_at
            ) for msg in messages
        ]
        
        return ConversationHistoryResponse(
            conversation=conversation_response,
            messages=message_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@router.post("/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def add_message(
    conversation_id: int,
    request: AddMessageRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a message to a conversation
    
    Args:
        conversation_id: Conversation identifier
        request: Message creation request
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created message details
    """
    try:
        user_id = current_user["user"]["id"]
        
        # Validate role
        if request.role not in ['user', 'assistant']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be 'user' or 'assistant'"
            )
        
        message = conversation_service.add_message(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            role=request.role,
            content=request.content,
            message_metadata=request.message_metadata
        )
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return MessageResponse(
            id=message.id,
            conversation_id=message.conversation_id,
            user_id=message.user_id,
            role=message.role,
            content=message.content,
            metadata=message.message_metadata,
            created_at=message.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add message: {str(e)}"
        )


@router.put("/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: int,
    title: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update conversation title
    
    Args:
        conversation_id: Conversation identifier
        title: New title
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    try:
        user_id = current_user["user"]["id"]
        
        success = conversation_service.update_conversation_title(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            title=title
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": "Conversation title updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation title: {str(e)}"
        )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a conversation (soft delete)
    
    Args:
        conversation_id: Conversation identifier
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    try:
        user_id = current_user["user"]["id"]
        
        success = conversation_service.delete_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@router.get("/stats/overview", response_model=ConversationStatsResponse)
async def get_conversation_stats(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get conversation statistics for the current user
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Conversation statistics
    """
    try:
        user_id = current_user["user"]["id"]
        
        stats = conversation_service.get_conversation_stats(db, user_id)
        
        return ConversationStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation stats: {str(e)}"
        )
