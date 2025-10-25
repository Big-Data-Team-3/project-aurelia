"""
RAG API Endpoints
FastAPI router for RAG functionality
"""
# region Imports
import time
from fastapi import APIRouter, HTTPException, Depends
import logging
# endregion
# region Models
from models.rag_models import (
    RAGQuery, RAGResponse, RAGHealthCheck
)
# endregion
# region Services
from services.rag_service import rag_service
from services.query_cache_service import query_cache_service
from services.cache_service import cache_service
from config.rag_config import rag_config
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




# Get cache statistics and performance metrics
@router.get("/cache/stats")
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    """Get cache statistics and performance metrics"""
    try:
        # Get Redis stats if available
        redis_stats = await cache_service.get_cache_stats()
        
        # Get memory cache stats
        memory_stats = cache_service.get_memory_cache_stats()
        
        return {
            "cache_enabled": rag_config.enable_caching,
            "redis_stats": redis_stats,
            "memory_cache_stats": memory_stats,
            "cache_config": {
                "response_ttl": rag_config.response_cache_ttl,
                "embedding_ttl": rag_config.embedding_cache_ttl,
                "search_ttl": rag_config.search_cache_ttl,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")


# endregion