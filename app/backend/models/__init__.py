# models/__init__.py
from sqlalchemy.ext.declarative import declarative_base

# Shared base for all models
Base = declarative_base()

# Import all models to ensure they're registered with the base
from .user import User, AuthToken
from .conversation import Conversation, Message

__all__ = ['Base', 'User', 'AuthToken', 'Conversation', 'Message']
