# models/conversation.py
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

# Import the shared base
from . import Base

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500))
    conversation_type = Column(String(50), default='general', index=True)
    status = Column(String(50), default='active', index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Relationship to user
    user = relationship("User", back_populates="conversations")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False, index=True)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON)  # Store search results and other metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User")
