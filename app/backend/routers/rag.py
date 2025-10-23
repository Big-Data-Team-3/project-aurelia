"""
RAG API Endpoints
FastAPI router for RAG functionality
"""

import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
# from sse_starlette.sse import EventSourceResponse  # Commented out for now
import json
import logging

from models.rag_models import (
    RAGQuery, RAGResponse, SearchOnlyQuery, SearchResponse,
    RerankQuery, RerankResponse, RAGHealthCheck, RAGConfig,
    ErrorResponse, SearchStrategy, ChatMessage, StreamingChunk,
    CreateSessionRequest, CreateSessionResponse, SessionMessageRequest,
    SessionMessageResponse, SessionListResponse, ChatSession
)
from services.rag_service import rag_service
from services.reranking import reranking_service
from services.session_service import session_service
from config.rag_config import rag_config, SearchStrategies


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


# Dependency for authentication (placeholder)
async def get_current_user():
    """Placeholder for user authentication"""
    return {"user_id": "demo_user", "username": "demo"}


@router.post("/query", response_model=RAGResponse)
async def rag_query(
    query: RAGQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Main RAG endpoint combining retrieval and generation
    
    Performs document retrieval using the specified strategy,
    applies Wikipedia fallback if needed, and generates a response.
    """
    try:
        logger.info(f"RAG query received: {query.query[:100]}...")
        
        response = await rag_service.query(query)
        
        logger.info(f"RAG query completed successfully in {response.total_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


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


@router.post("/rerank", response_model=RerankResponse)
async def rerank_results(
    rerank_query: RerankQuery,
    current_user: dict = Depends(get_current_user)
):
    """
    Rerank search results using cross-encoder models
    
    Takes a list of search results and reranks them based on
    relevance to the query using neural reranking models.
    """
    try:
        logger.info(f"Rerank query received with {len(rerank_query.results)} results")
        
        if not rag_config.enable_reranking:
            raise HTTPException(
                status_code=400,
                detail="Reranking is disabled in the current configuration"
            )
        
        reranked_results, rerank_time = await reranking_service.rerank_results(
            query=rerank_query.query,
            results=rerank_query.results,
            top_k=rerank_query.top_k
        )
        
        response = RerankResponse(
            query=rerank_query.query,
            reranked_results=reranked_results,
            original_count=len(rerank_query.results),
            reranked_count=len(reranked_results),
            rerank_time_ms=rerank_time
        )
        
        logger.info(f"Reranking completed in {rerank_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}"
        )


# Streaming endpoint temporarily disabled due to sse_starlette dependency
# @router.post("/query/stream")
# async def rag_query_stream(
#     query: RAGQuery,
#     current_user: dict = Depends(get_current_user)
# ):
#     """
#     Streaming RAG endpoint for real-time responses
#     
#     Returns Server-Sent Events (SSE) stream with incremental
#     response data including sources and generated text.
#     """
#     try:
#         logger.info(f"Streaming RAG query received: {query.query[:100]}...")
#         
#         async def event_generator():
#             try:
#                 async for chunk in rag_service.streaming_query(query):
#                     # Convert chunk to JSON
#                     chunk_data = {
#                         "type": chunk.type,
#                         "content": chunk.content,
#                         "is_final": chunk.is_final
#                     }
#                     
#                     # Handle different content types for JSON serialization
#                     if chunk.type == "sources" and isinstance(chunk.content, list):
#                         # Convert SearchResult objects to dicts
#                         chunk_data["content"] = [
#                             result.dict() for result in chunk.content
#                         ]
#                     
#                     yield {
#                         "event": "chunk",
#                         "data": json.dumps(chunk_data)
#                     }
#                     
#                     if chunk.is_final:
#                         break
#                         
#             except Exception as e:
#                 logger.error(f"Streaming error: {e}")
#                 yield {
#                     "event": "error",
#                     "data": json.dumps({"error": str(e)})
#                 }
#         
#         return EventSourceResponse(event_generator())
#         
#     except Exception as e:
#         logger.error(f"Streaming RAG query failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Streaming query failed: {str(e)}"
#         )


@router.post("/conversation", response_model=RAGResponse)
async def conversational_query(
    query: str = Query(..., description="Current user query"),
    conversation_history: List[ChatMessage] = [],
    strategy: SearchStrategy = SearchStrategy.RRF_FUSION,
    top_k: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """
    RAG query with conversation context
    
    Processes a query considering the conversation history
    for more contextual and coherent responses.
    """
    try:
        logger.info(f"Conversational query with {len(conversation_history)} history messages")
        
        response = await rag_service.conversational_query(
            query=query,
            conversation_history=conversation_history,
            strategy=strategy,
            top_k=top_k
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Conversational query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversational query failed: {str(e)}"
        )


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


def _get_strategy_description(strategy: str) -> str:
    """Get human-readable description for a search strategy"""
    descriptions = {
        SearchStrategies.VECTOR_ONLY: "Pure semantic vector search using embeddings",
        SearchStrategies.HYBRID: "Combines vector search with keyword matching using BM25",
        SearchStrategies.RERANKED: "Vector search results reranked using cross-encoder models",
        SearchStrategies.RRF_FUSION: "Full hybrid approach with RRF fusion and neural reranking"
    }
    return descriptions.get(strategy, "Unknown strategy")


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


# Session-based chat endpoints
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


@router.post("/chat/session/{session_id}/extend")
async def extend_session_ttl(
    session_id: str,
    additional_hours: int = Query(24, ge=1, le=168, description="Hours to add to TTL"),
    current_user: dict = Depends(get_current_user)
):
    """
    Extend session time-to-live
    
    Adds additional time to prevent session expiration.
    """
    try:
        session = await session_service.extend_session_ttl(session_id, additional_hours)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(f"Extended session {session_id} TTL by {additional_hours} hours")
        return {
            "message": f"Session TTL extended by {additional_hours} hours",
            "new_expires_at": session.expires_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extend session TTL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extend session: {str(e)}"
        )


@router.get("/chat/stats")
async def get_session_stats(current_user: dict = Depends(get_current_user)):
    """
    Get session statistics for monitoring and debugging
    
    Returns system-wide session statistics.
    """
    try:
        stats = await session_service.get_session_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get session stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Exception handlers removed - APIRouter doesn't support exception_handler
# These would need to be added to the main FastAPI app in main.py if needed
