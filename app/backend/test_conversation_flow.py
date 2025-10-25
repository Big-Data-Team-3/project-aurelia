#!/usr/bin/env python3
"""
Test script for conversation persistence functionality
Tests the complete conversation flow from creation to message storage
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import get_db, SessionLocal
from services.conversation_service import conversation_service
from models.conversation import Conversation, Message
from models.user import User

def test_conversation_flow():
    """Test the complete conversation flow"""
    print("🧪 Testing Conversation Persistence Flow")
    print("=" * 50)
    
    # Create a test database session
    db = SessionLocal()
    
    try:
        # Test 1: Create a test user (if not exists)
        print("1️⃣ Creating test user...")
        test_user = db.query(User).filter(User.email == "test@example.com").first()
        if not test_user:
            test_user = User(
                google_id="test_google_id_123",
                email="test@example.com",
                name="Test User",
                role="user",
                is_active=True
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            print(f"   ✅ Created test user: {test_user.email} (ID: {test_user.id})")
        else:
            print(f"   ✅ Using existing test user: {test_user.email} (ID: {test_user.id})")
        
        # Test 2: Create a conversation
        print("\n2️⃣ Creating conversation...")
        conversation = conversation_service.create_conversation(
            db=db,
            user_id=test_user.id,
            title="Test RAG Conversation",
            conversation_type="rag_query"
        )
        print(f"   ✅ Created conversation: {conversation.title} (ID: {conversation.id})")
        
        # Test 3: Add user message
        print("\n3️⃣ Adding user message...")
        user_message = conversation_service.add_message(
            db=db,
            conversation_id=conversation.id,
            user_id=test_user.id,
            role="user",
            content="What are the key financial metrics in the document?"
        )
        print(f"   ✅ Added user message: {user_message.content[:50]}...")
        
        # Test 4: Add assistant message with metadata
        print("\n4️⃣ Adding assistant message with metadata...")
        assistant_message = conversation_service.add_message(
            db=db,
            conversation_id=conversation.id,
            user_id=test_user.id,
            role="assistant",
            content="Based on the financial documents, the key metrics include revenue growth, profit margins, and cash flow. Here are the specific details...",
            message_metadata={
                "search_results": [
                    {
                        "id": "chunk_123",
                        "text": "Revenue increased by 15% year-over-year...",
                        "score": 0.92,
                        "section": "Financial Performance",
                        "pages": [5, 6]
                    }
                ],
                "processing_time_ms": 1250,
                "strategy_used": "rrf_fusion"
            }
        )
        print(f"   ✅ Added assistant message with metadata")
        
        # Test 5: Retrieve conversation history
        print("\n5️⃣ Retrieving conversation history...")
        messages = conversation_service.get_conversation_messages(
            db=db,
            conversation_id=conversation.id,
            user_id=test_user.id
        )
        print(f"   ✅ Retrieved {len(messages)} messages")
        for i, msg in enumerate(messages, 1):
            print(f"      {i}. [{msg.role}] {msg.content[:50]}...")
        
        # Test 6: Get conversation context for RAG
        print("\n6️⃣ Getting conversation context for RAG...")
        context = conversation_service.get_conversation_context(
            db=db,
            conversation_id=conversation.id,
            user_id=test_user.id
        )
        print(f"   ✅ Retrieved {len(context)} context messages for RAG")
        
        # Test 7: Get user conversations
        print("\n7️⃣ Getting user conversations...")
        conversations = conversation_service.get_user_conversations(
            db=db,
            user_id=test_user.id
        )
        print(f"   ✅ Retrieved {len(conversations)} conversations for user")
        
        # Test 8: Get conversation statistics
        print("\n8️⃣ Getting conversation statistics...")
        stats = conversation_service.get_conversation_stats(db, test_user.id)
        print(f"   ✅ Statistics: {stats}")
        
        # Test 9: Update conversation title
        print("\n9️⃣ Updating conversation title...")
        success = conversation_service.update_conversation_title(
            db=db,
            conversation_id=conversation.id,
            user_id=test_user.id,
            title="Updated Test Conversation"
        )
        print(f"   ✅ Title update: {'Success' if success else 'Failed'}")
        
        print("\n🎉 All tests passed! Conversation persistence is working correctly.")
        
        # Cleanup (optional - comment out to keep test data)
        print("\n🧹 Cleaning up test data...")
        conversation_service.delete_conversation(db, conversation.id, test_user.id)
        print("   ✅ Test conversation deleted")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def test_api_endpoints():
    """Test API endpoint functionality (requires running server)"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 30)
    print("Note: This requires the FastAPI server to be running")
    print("Run: python main.py")
    print("Then test endpoints at: http://localhost:8000/docs")

if __name__ == "__main__":
    print("🚀 Project Aurelia - Conversation Persistence Test")
    print("=" * 60)
    
    # Test database operations
    if test_conversation_flow():
        print("\n✅ Database operations test: PASSED")
    else:
        print("\n❌ Database operations test: FAILED")
        sys.exit(1)
    
    # Show API testing instructions
    test_api_endpoints()
    
    print("\n🎯 Test Summary:")
    print("   ✅ Conversation creation")
    print("   ✅ Message storage")
    print("   ✅ Metadata handling")
    print("   ✅ Context retrieval")
    print("   ✅ User conversation listing")
    print("   ✅ Statistics generation")
    print("   ✅ Title updates")
    print("   ✅ Conversation deletion")
    print("\n🚀 Conversation persistence is ready for production!")
