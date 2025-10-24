"""
RAG API Endpoints
FastAPI router for RAG functionality
"""
# region Imports
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import logging
# endregion
# region Models
from models.rag_models import (
    RAGQuery, RAGResponse, SearchOnlyQuery, SearchResponse,
    RerankQuery, RerankResponse, RAGHealthCheck, RAGConfig,
    ErrorResponse, SearchStrategy, ChatMessage, StreamingChunk,
    CreateSessionRequest, CreateSessionResponse, SessionMessageRequest,
    SessionMessageResponse, SessionListResponse, ChatSession
)
# endregion
# region Services
from services.rag_service import rag_service
from services.reranking import reranking_service
from services.session_service import session_service
from services.context_manager import context_manager
from config.rag_config import rag_config, SearchStrategies
from services.cache_service import cache_service
# endregion
# region Config
from config.rag_config import rag_config, SearchStrategies
# endregion
# region Logger
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])
# endregion
# Dependency for authentication (placeholder)
async def get_current_user():
    """Placeholder for user authentication"""
    return {"user_id": "demo_user", "username": "demo"}

# region Endpoints
# RAG query endpoint for standard query (non-streaming)
@router.post("/query", response_model=RAGResponse)
async def rag_query(
    query: RAGQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Enhanced RAG endpoint with automatic session management
    
    Performs document retrieval using the specified strategy,
    applies Wikipedia fallback if needed, and generates a response.
    Automatically creates and manages sessions for multi-turn conversations.
    """
    logger.info(f"RAG query received: {query.query[:100]}...")
    try:
        # Set user_id from current_user if not provided
        if not query.user_id:
            query.user_id = current_user.get("user_id", "anonymous")
        
        response = await rag_service.query(query)
        return response
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )

# Streaming RAG query endpoint
@router.post("/query/stream")  # Remove response_model=StreamingResponse
async def streaming_rag_query(
    query: RAGQuery,
    current_user: dict = Depends(get_current_user)
):
    """Streaming RAG query endpoint"""
    try:
        import json
        import asyncio
        
        async def generate_stream():
            try:
                async for chunk in rag_service.streaming_query(query):
                    # Convert StreamingChunk to JSON properly
                    chunk_data = {
                        "type": chunk.type,
                        "content": chunk.content,
                        "is_final": chunk.is_final
                    }
                    
                    # Handle SearchResult objects in content
                    if chunk.type == "sources" and isinstance(chunk.content, list):
                        chunk_data["content"] = [
                            source.dict() if hasattr(source, 'dict') else str(source)
                            for source in chunk.content
                        ]
                    
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e), 'is_final': True})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache", 
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"Streaming RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Streaming RAG query failed: {str(e)}"
        )
    
# Search-only endpoint without generation
@router.post("/search", response_model=SearchResponse)
async def search_only(
    search_query: SearchOnlyQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Search-only endpoint without generation
    
    Retrieves relevant documents using the specified strategy
    without generating a response.
    """
    try:
        logger.info(f"Search query received: {search_query.query[:100]}...")
        
        response = await rag_service.search_only(
            query=search_query.query,
            strategy=search_query.strategy,
            top_k=search_query.top_k
        )
        
        logger.info(f"Search completed with {response.total_results} results")
        return response
        
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search query failed: {str(e)}"
        )

# Health check endpoint for RAG system
@router.get("/health", response_model=RAGHealthCheck)
async def health_check():
    """
    Health check endpoint for RAG system
    
    Checks the status of all RAG components including
    Pinecone, OpenAI, reranking models, and Wikipedia service.
    """
    try:
        health_status = await rag_service.health_check()
        
        response = RAGHealthCheck(
            status="healthy" if health_status.get('overall', False) else "unhealthy",
            pinecone_connected=health_status.get('vector_search', False),
            openai_connected=health_status.get('generation', False),
            reranker_loaded=health_status.get('reranking', False),
            wikipedia_available=health_status.get('wikipedia', False)
        )
        
        if not response.status == "healthy":
            logger.warning(f"RAG system health check failed: {health_status}")
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

# Get RAG system configuration
@router.get("/config", response_model=RAGConfig)
async def get_config():
    """
    Get RAG system configuration
    
    Returns current configuration including available strategies,
    models being used, and enabled features.
    """
    try:
        reranker_info = reranking_service.get_model_info()
        
        config = RAGConfig(
            available_strategies=list(SearchStrategy),
            default_strategy=SearchStrategy.RRF_FUSION,
            max_top_k=rag_config.max_top_k,
            reranker_model=rag_config.reranker_model,
            embedding_model=rag_config.embedding_model,
            generation_model=rag_config.generation_model,
            features={
                "reranking_enabled": rag_config.enable_reranking,
                "wikipedia_fallback": rag_config.enable_wikipedia_fallback,
                "caching_enabled": rag_config.enable_caching,
                "query_logging": rag_config.enable_query_logging
            }
        )
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration: {str(e)}"
        )

# Get available search strategies and their configurations
@router.get("/strategies")
async def get_search_strategies():
    """
    Get available search strategies and their configurations
    
    Returns detailed information about each search strategy
    including what components they use.
    """
    try:
        strategies = {}
        
        for strategy in SearchStrategies.get_all_strategies():
            config = SearchStrategies.get_strategy_config(strategy)
            strategies[strategy] = {
                "name": strategy,
                "description": _get_strategy_description(strategy),
                "config": config,
                "components": _get_strategy_components(config)
            }
        
        return {
            "strategies": strategies,
            "default": SearchStrategies.RRF_FUSION,
            "recommended": SearchStrategies.RRF_FUSION
        }
        
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search strategies: {str(e)}"
        )

# Batch RAG queries for processing multiple questions
@router.post("/batch/query")
async def batch_rag_query(
    queries: List[RAGQuery],
    max_concurrent: int = Query(5, ge=1, le=10),
    current_user: dict = Depends(get_current_user)
):
    """
    Batch RAG queries for processing multiple questions
    
    Processes multiple RAG queries concurrently with
    configurable concurrency limits.
    """
    try:
        if len(queries) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 queries allowed per batch"
            )
        
        logger.info(f"Batch RAG query with {len(queries)} queries")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_query(query: RAGQuery):
            async with semaphore:
                return await rag_service.query(query)
        
        # Process queries concurrently
        tasks = [process_query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = RAGResponse(
                    query=queries[i].query,
                    answer=f"Error processing query: {str(response)}",
                    sources=[],
                    strategy_used=queries[i].strategy,
                    metadata={"error": str(response)}
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return {"responses": results, "total_queries": len(queries)}
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch query failed: {str(e)}"
        )

# Get cache statistics and performance metrics
@router.get("/cache/stats")
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    """Get cache statistics and performance metrics"""
    try:
        stats = await cache_service.get_cache_stats()
        return {
            "cache_enabled": rag_config.enable_caching,
            "redis_stats": stats,
            "cache_config": {
                "response_ttl": rag_config.response_cache_ttl,
                "embedding_ttl": rag_config.embedding_cache_ttl,
                "search_ttl": rag_config.search_cache_ttl,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

# Clear cache by type
@router.delete("/cache/clear")
async def clear_cache(
    cache_type: str = Query("all", description="Cache type to clear: all, responses, embeddings, search, sessions"),
    current_user: dict = Depends(get_current_user)
):
    """Clear cache by type"""
    try:
        if cache_type == "all":
            patterns = ["response", "embedding", "search", "session"]
        else:
            patterns = [cache_type]
        
        total_cleared = 0
        for pattern in patterns:
            cleared = await cache_service.clear_cache_pattern(pattern)
            total_cleared += cleared
        
        return {
            "message": f"Cleared {total_cleared} cache entries",
            "cache_type": cache_type
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

# Check cache system health
@router.get("/cache/health")
async def cache_health_check():
    """Check cache system health"""
    try:
        is_healthy = await cache_service.health_check()
        return {
            "cache_enabled": rag_config.enable_caching,
            "redis_healthy": is_healthy,
            "redis_url": rag_config.redis_url.split('@')[-1] if '@' in rag_config.redis_url else rag_config.redis_url
        }
    except Exception as e:
        return {
            "cache_enabled": rag_config.enable_caching,
            "redis_healthy": False,
            "error": str(e)
        }

# Create a new chat session for persistent conversation
@router.post("/chat/session", response_model=CreateSessionResponse)
async def create_chat_session(
    request: CreateSessionRequest = CreateSessionRequest(),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new chat session for persistent conversation
    
    Creates a session with configurable TTL and context window management.
    """
    try:
        # Use current user ID if not provided in request
        user_id = request.user_id or current_user.get("user_id", "anonymous")
        
        session = await session_service.create_session(
            user_id=user_id,
            ttl_hours=request.ttl_hours
        )
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        
        return CreateSessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            expires_at=session.expires_at,
            created_at=session.created_at
        )
        
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )

# Send a message to an existing chat session
@router.post("/chat/session/{session_id}/message", response_model=SessionMessageResponse)
async def send_session_message(
    session_id: str,
    message_request: SessionMessageRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Send a message to an existing chat session
    
    Processes the message through PII filtering, intent classification,
    and routes to RAG or ChatGPT based on analysis.
    """
    try:
        logger.info(f"Processing message for session {session_id}")
        
        response = await rag_service.session_message(session_id, message_request)
        
        logger.info(f"Session message processed: {response.routing_decision} routing")
        return response
        
    except ValueError as e:
        # Session not found or expired
        logger.warning(f"Session error: {e}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Session message processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Message processing failed: {str(e)}"
        )

# Get session details and conversation history
@router.get("/chat/session/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get session details and conversation history
    
    Returns the complete session with all messages and metadata.
    """
    try:
        session = await session_service.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired"
            )
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session: {str(e)}"
        )

# Delete a chat session and all its data
@router.delete("/chat/session/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a chat session and all its data
    
    Permanently removes the session and conversation history.
    """
    try:
        deleted = await session_service.delete_session(session_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Deleted session {session_id}")
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

# List all sessions for the current user
@router.get("/chat/sessions", response_model=SessionListResponse)
async def list_user_sessions(
    active_only: bool = Query(True, description="Only return active (non-expired) sessions"),
    current_user: dict = Depends(get_current_user)
):
    """
    List all sessions for the current user
    
    Returns sessions ordered by most recently updated first.
    """
    try:
        user_id = current_user.get("user_id", "anonymous")
        sessions = await session_service.get_user_sessions(user_id, active_only)
        
        return SessionListResponse(
            sessions=sessions,
            total_count=len(sessions)
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions for user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )
# endregion
# region Helper functions

# Get human-readable description for a search strategy
def _get_strategy_description(strategy: str) -> str:
    """Get human-readable description for a search strategy"""
    descriptions = {
        SearchStrategies.VECTOR_ONLY: "Pure semantic vector search using embeddings",
        SearchStrategies.HYBRID: "Combines vector search with keyword matching using BM25",
        SearchStrategies.RERANKED: "Vector search results reranked using cross-encoder models",
        SearchStrategies.RRF_FUSION: "Full hybrid approach with RRF fusion and neural reranking"
    }
    return descriptions.get(strategy, "Unknown strategy")

# Get list of components used by a strategy
def _get_strategy_components(config: Dict[str, bool]) -> List[str]:
    """Get list of components used by a strategy"""
    components = []
    if config.get("use_vector"):
        components.append("Vector Search")
    if config.get("use_keyword"):
        components.append("Keyword Search (BM25)")
    if config.get("use_reranking"):
        components.append("Neural Reranking")
    if config.get("use_rrf"):
        components.append("RRF Fusion")
    return components

# Context Management Endpoints
@router.get("/context/analytics/{session_id}")
async def get_context_analytics(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get context analytics for a session
    
    Returns analytics about conversation context, including:
    - Message counts and lengths
    - Token usage estimates
    - Context strategy being used
    - Session activity metrics
    """
    try:
        # Get session
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired"
            )
        
        # Get context analytics
        analytics = await context_manager.get_context_analytics(session)
        
        return {
            "session_id": session_id,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context analytics failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get context analytics: {str(e)}"
        )

@router.post("/context/optimize/{session_id}")
async def optimize_session_context(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Manually optimize context for a session
    
    This endpoint can be used to force context optimization
    and see what the optimized context would look like.
    """
    try:
        # Get session
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or expired"
            )
        
        # Get optimized context
        context_window = await context_manager.get_optimized_context(session)
        
        return {
            "session_id": session_id,
            "context_window": {
                "window_type": context_window.window_type,
                "message_count": len(context_window.messages),
                "token_count": context_window.token_count,
                "has_summary": context_window.summary is not None,
                "created_at": context_window.created_at.isoformat()
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                for msg in context_window.messages
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize context: {str(e)}"
        )

# endregion