# Conversation Persistence System

This document describes the conversation persistence system that replaces Redis-based session management with CloudSQL-based conversation storage.

## Overview

The conversation persistence system provides:
- **Persistent Storage**: Conversations and messages stored in CloudSQL PostgreSQL
- **User Isolation**: Each user has their own conversation history
- **Metadata Support**: Rich metadata storage for search results and processing information
- **RAG Integration**: Seamless integration with the RAG system for context-aware responses
- **Authentication**: JWT token-based authentication for all endpoints

## Database Schema

### Conversations Table
```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    conversation_type VARCHAR(50) DEFAULT 'general',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    message_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes
- `idx_messages_conversation_id` on `messages(conversation_id)`
- `idx_messages_created_at` on `messages(created_at)`

## API Endpoints

### Conversation Management

#### Create Conversation
```http
POST /conversations/
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "title": "My RAG Conversation",
    "conversation_type": "rag_query"
}
```

#### Get User Conversations
```http
GET /conversations/?conversation_type=rag_query&limit=50
Authorization: Bearer <jwt_token>
```

#### Get Conversation History
```http
GET /conversations/{conversation_id}
Authorization: Bearer <jwt_token>
```

#### Add Message to Conversation
```http
POST /conversations/{conversation_id}/messages
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "role": "user",
    "content": "What are the key financial metrics?",
    "message_metadata": {
        "search_results": [...],
        "processing_time_ms": 1250
    }
}
```

#### Update Conversation Title
```http
PUT /conversations/{conversation_id}/title?title=New Title
Authorization: Bearer <jwt_token>
```

#### Delete Conversation
```http
DELETE /conversations/{conversation_id}
Authorization: Bearer <jwt_token>
```

#### Get Conversation Statistics
```http
GET /conversations/stats/overview
Authorization: Bearer <jwt_token>
```

### RAG Integration

#### Standard RAG Query (Auto-creates conversation)
```http
POST /rag/query
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "query": "What are the key financial metrics?",
    "strategy": "rrf_fusion",
    "top_k": 10,
    "include_sources": true
}
```

#### Conversation-based RAG Query
```http
POST /rag/conversation/{conversation_id}/query
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
    "query": "Can you elaborate on the revenue growth?",
    "strategy": "rrf_fusion",
    "top_k": 10
}
```

## Usage Examples

### 1. Create a New RAG Conversation

```python
import requests

# Create conversation
response = requests.post(
    "http://localhost:8000/conversations/",
    headers={"Authorization": "Bearer <your_jwt_token>"},
    json={
        "title": "Financial Analysis Discussion",
        "conversation_type": "rag_query"
    }
)
conversation = response.json()
conversation_id = conversation["id"]

# Ask a question
response = requests.post(
    f"http://localhost:8000/rag/conversation/{conversation_id}/query",
    headers={"Authorization": "Bearer <your_jwt_token>"},
    json={
        "query": "What are the key financial metrics?",
        "strategy": "rrf_fusion"
    }
)
result = response.json()
print(result["answer"])
```

### 2. Continue a Conversation

```python
# Ask a follow-up question in the same conversation
response = requests.post(
    f"http://localhost:8000/rag/conversation/{conversation_id}/query",
    headers={"Authorization": "Bearer <your_jwt_token>"},
    json={
        "query": "Can you explain the revenue growth in more detail?",
        "strategy": "rrf_fusion"
    }
)
result = response.json()
print(result["answer"])
```

### 3. Retrieve Conversation History

```python
# Get full conversation history
response = requests.get(
    f"http://localhost:8000/conversations/{conversation_id}",
    headers={"Authorization": "Bearer <your_jwt_token>"}
)
history = response.json()

for message in history["messages"]:
    print(f"[{message['role']}]: {message['content']}")
    if message.get("metadata", {}).get("search_results"):
        print(f"  Sources: {len(message['metadata']['search_results'])} results")
```

## Migration from Redis Sessions

The system has been designed to replace the Redis-based session management:

### Before (Redis Sessions)
- Temporary storage with TTL
- In-memory session tracking
- No persistent conversation history
- Limited metadata storage

### After (CloudSQL Conversations)
- Persistent storage in PostgreSQL
- Full conversation history
- Rich metadata support
- User-specific conversation isolation
- Better scalability and reliability

## Setup Instructions

### 1. Run Database Migration

```bash
cd app/backend
python migrations/create_conversation_tables.py
```

### 2. Test the System

```bash
python test_conversation_flow.py
```

### 3. Start the Server

```bash
python main.py
```

### 4. Test API Endpoints

Visit `http://localhost:8000/docs` to test the endpoints interactively.

## Key Features

### 1. **User Isolation**
- Each user can only access their own conversations
- JWT token authentication ensures security
- User ID is automatically extracted from the token

### 2. **Conversation Types**
- `general`: General conversations
- `rag_query`: RAG-based queries with document search

### 3. **Rich Metadata**
- Search results stored in message metadata
- Processing times and performance metrics
- Strategy used for RAG queries
- Cache hit/miss information

### 4. **Context Management**
- Conversation history used for RAG context
- Automatic context window management
- Optimized for multi-turn conversations

### 5. **Performance**
- Indexed database queries for fast retrieval
- Efficient conversation context loading
- Minimal overhead for conversation persistence

## Error Handling

The system includes comprehensive error handling:

- **Authentication Errors**: 401 for invalid/expired tokens
- **Authorization Errors**: 403 for accessing other users' conversations
- **Not Found Errors**: 404 for non-existent conversations
- **Validation Errors**: 400 for invalid request data
- **Server Errors**: 500 for internal processing errors

## Monitoring and Analytics

### Conversation Statistics
- Total conversations per user
- Total messages per user
- Conversation type breakdown
- Usage patterns and trends

### Performance Metrics
- Message processing times
- Cache hit rates
- Search result quality
- User engagement metrics

## Security Considerations

1. **Authentication**: All endpoints require valid JWT tokens
2. **Authorization**: Users can only access their own conversations
3. **Data Isolation**: Complete separation between user data
4. **Input Validation**: All inputs are validated and sanitized
5. **SQL Injection Protection**: Using SQLAlchemy ORM prevents SQL injection

## Future Enhancements

1. **Conversation Search**: Full-text search across conversation history
2. **Conversation Sharing**: Ability to share conversations with other users
3. **Conversation Export**: Export conversations to various formats
4. **Advanced Analytics**: More detailed usage analytics and insights
5. **Conversation Templates**: Pre-defined conversation templates for common use cases

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check CloudSQL connection settings
   - Verify database credentials
   - Ensure CloudSQL proxy is running (for development)

2. **Authentication Errors**
   - Verify JWT token is valid and not expired
   - Check token format in Authorization header
   - Ensure user exists and is active

3. **Conversation Not Found**
   - Verify conversation ID is correct
   - Check if conversation belongs to the authenticated user
   - Ensure conversation status is 'active'

### Debug Mode

Enable debug logging by setting the log level to DEBUG in your configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about conversation operations, database queries, and error conditions.
