"""
RAG API Endpoints
FastAPI router for RAG functionality
"""
# region Imports
import asyncio
from datetime import datetime
import time
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
from services.query_cache_service import query_cache_service
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
    Enhanced RAG endpoint with automatic session management and cache checking
    
    Performs document retrieval using the specified strategy,
    applies Wikipedia fallback if needed, and generates a response.
    Automatically creates and manages sessions for multi-turn conversations.
    
    Cache Behavior:
    1. First checks for cached response
    2. If cache hit: Returns cached response instantly
    3. If cache miss: Processes query normally and caches the result
    """
    logger.info(f"RAG query received: {query.query[:100]}...")
    start_time = time.time()
    
    try:
        # Set user_id from current_user if not provided
        #Strip whitespaces from query
        query.query = query.query.strip()
        if not query.user_id:
            query.user_id = current_user.get("user_id", "anonymous")
        
        # Step 1: Check for cached response first
        cached_result = await query_cache_service.get_cached_response(query)
        if cached_result:
            cached_response, perf_data = cached_result
            processing_time = time.time() - start_time
            
            # Add cache performance metadata
            cached_response.metadata = cached_response.metadata or {}
            cached_response.metadata.update({
                'cache_hit': True,
                'cache_retrieval_time_ms': round(processing_time * 1000, 2),
                'original_processing_time_ms': perf_data.get('processing_time_ms', 0),
                'cache_speedup': f"{perf_data.get('processing_time_ms', 0) / (processing_time * 1000):.1f}x faster" if processing_time > 0 else "instant",
                'cached_at': perf_data.get('cached_at', 'unknown')
            })
            
            logger.info(f"Cache HIT for query: {query.query[:50]}... (retrieved in {processing_time*1000:.2f}ms)")
            return cached_response
        
        # Step 2: Cache miss - process query normally
        logger.info(f"Cache MISS for query: {query.query[:50]}... - processing normally")
        response = await rag_service.query(query)
        
        # Step 3: Cache the response for future use
        processing_time = time.time() - start_time
        await query_cache_service.cache_response(
            query, 
            response, 
            processing_time * 1000  # Convert to milliseconds
        )
        
        # Add cache miss metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            'cache_hit': False,
            'processing_time_ms': round(processing_time * 1000, 2),
            'cached_for_future': True
        })
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )

# Cached RAG query endpoint
@router.post("/query/cached", response_model=RAGResponse)
async def cached_rag_query(
    query: RAGQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Cached RAG endpoint with instant responses for repeated queries
    
    This endpoint provides the same functionality as the standard /query endpoint,
    but with intelligent caching for performance optimization:
    
    1. First time: Processes query normally and caches the response
    2. Subsequent times: Returns cached response instantly with performance comparison
    
    Features:
    - Instant responses for cached queries
    - Performance metrics showing speedup over standard processing
    - Automatic cache management with TTL
    - Cache statistics and analytics
    """
    logger.info(f"Cached RAG query received: {query.query[:100]}...")
    start_time = time.time()
    
    try:
        # Set user_id from current_user if not provided
        if not query.user_id:
            query.user_id = current_user.get("user_id", "anonymous")
        
        # Step 1: Check for cached response
        cached_result = await query_cache_service.get_cached_response(query)
        
        if cached_result:
            # Cache HIT - Return cached response with performance comparison
            cached_response, perf_data = cached_result
            
            # Calculate current response time (should be very fast)
            current_time_ms = (time.time() - start_time) * 1000
            original_time_ms = perf_data.get('processing_time_ms', 0)
            
            # Calculate speedup
            speedup = original_time_ms / current_time_ms if current_time_ms > 0 else 0
            
            # Update response with cache information
            cached_response.metadata.update({
                'cache_hit': True,
                'cached_at': perf_data.get('cached_at'),
                'original_processing_time_ms': original_time_ms,
                'cached_response_time_ms': current_time_ms,
                'speedup_factor': round(speedup, 2),
                'cache_performance': f"{speedup:.1f}x faster than original",
                'cache_key': f"query_cache:{hash(query.query + str(query.strategy) + str(query.top_k))}"
            })
            
            # Update timing information
            cached_response.retrieval_time_ms = 0.0  # No retrieval for cached response
            cached_response.generation_time_ms = 0.0  # No generation for cached response
            cached_response.total_time_ms = current_time_ms
            
            logger.info(f"Cache HIT: {query.query[:50]}... (Speedup: {speedup:.1f}x)")
            return cached_response
        
        else:
            # Cache MISS - Process query normally and cache the result
            logger.info(f"Cache MISS: {query.query[:50]}... - Processing normally")
            
            # Process query using standard RAG service
            response = await rag_service.query(query)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache the response for future use
            cache_success = await query_cache_service.cache_response(
                query, response, processing_time_ms
            )
            
            # Add cache information to response
            response.metadata.update({
                'cache_hit': False,
                'cache_stored': cache_success,
                'processing_time_ms': processing_time_ms,
                'cache_note': 'Response cached for future instant retrieval'
            })
            
            logger.info(f"Cache MISS processed and cached: {query.query[:50]}... ({processing_time_ms:.2f}ms)")
            return response
            
    except Exception as e:
        logger.error(f"Cached RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cached RAG query failed: {str(e)}"
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

# Query cache management endpoints
@router.get("/cache/stats")
async def get_cache_stats():
    """Get query cache statistics and performance metrics"""
    try:
        stats = await query_cache_service.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@router.delete("/cache/clear")
async def clear_query_cache(
    query_pattern: str = Query(None, description="Optional pattern to match specific queries")
):
    """Clear query cache entries"""
    try:
        success = await query_cache_service.clear_cache(query_pattern)
        
        if success:
            message = f"Cache cleared successfully"
            if query_pattern:
                message += f" for queries matching: {query_pattern}"
            
            return {
                "status": "success",
                "message": message,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear cache"
            )
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/cache/health/detailed")
async def detailed_cache_health():
    """Detailed cache health check including query cache"""
    try:
        # Basic cache health
        basic_health = await cache_service.health_check()
        
        # Query cache health
        query_cache_health = await query_cache_service.health_check()
        
        # Cache stats
        cache_stats = await query_cache_service.get_cache_stats()
        
        return {
            "overall_healthy": basic_health and query_cache_health,
            "basic_cache": {
                "enabled": rag_config.enable_caching,
                "redis_healthy": basic_health,
                "redis_url": rag_config.redis_url.split('@')[-1] if '@' in rag_config.redis_url else rag_config.redis_url
            },
            "query_cache": {
                "healthy": query_cache_health,
                "stats": cache_stats
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Detailed cache health check failed: {e}")
        return {
            "overall_healthy": False,
            "error": str(e),
            "timestamp": time.time()
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