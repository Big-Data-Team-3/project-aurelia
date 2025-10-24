"""
Redis-based caching service for RAG system
Handles response caching, embedding caching, and session storage
"""

import redis
import json
import pickle
import hashlib
import logging
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from config.rag_config import rag_config
from models.rag_models import RAGResponse, SearchResult, ChatSession

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service with connection pooling and error handling"""
    
    def __init__(self):
        self.redis_pool = None
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with pool"""
        try:
            # Create connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                rag_config.redis_url,
                password=rag_config.redis_password if rag_config.redis_password else None,
                max_connections=rag_config.redis_max_connections,
                retry_on_timeout=True
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _create_key(self, prefix: str, *args) -> str:
        """Create a standardized cache key"""
        key_parts = [rag_config.cache_prefix, prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def _hash_object(self, obj: Any) -> str:
        """Create a hash for complex objects"""
        if isinstance(obj, (dict, list)):
            obj_str = json.dumps(obj, sort_keys=True)
        else:
            obj_str = str(obj)
        return hashlib.md5(obj_str.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or rag_config.cache_ttl
            data = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    # RAG Response Caching
    async def cache_rag_response(self, query_hash: str, response: RAGResponse) -> bool:
        """Cache RAG response"""
        key = self._create_key("response", query_hash)
        return await self.set(key, response, rag_config.response_cache_ttl)
    
    async def get_cached_rag_response(self, query_hash: str) -> Optional[RAGResponse]:
        """Get cached RAG response"""
        key = self._create_key("response", query_hash)
        return await self.get(key)
    
    # Embedding Caching
    async def cache_embedding(self, text_hash: str, embedding: List[float]) -> bool:
        """Cache text embedding"""
        key = self._create_key("embedding", text_hash)
        return await self.set(key, embedding, rag_config.embedding_cache_ttl)
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._create_key("embedding", text_hash)
        return await self.get(key)
    
    # Search Results Caching
    async def cache_search_results(self, search_hash: str, results: List[SearchResult]) -> bool:
        """Cache search results"""
        key = self._create_key("search", search_hash)
        # Convert SearchResult objects to dicts for caching
        results_data = [result.dict() for result in results]
        return await self.set(key, results_data, rag_config.search_cache_ttl)
    
    async def get_cached_search_results(self, search_hash: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        key = self._create_key("search", search_hash)
        results_data = await self.get(key)
        if results_data:
            # Convert back to SearchResult objects
            return [SearchResult(**result) for result in results_data]
        return None
    
    # Session Caching (Alternative to in-memory sessions)
    async def cache_session(self, session_id: str, session: ChatSession) -> bool:
        """Cache chat session"""
        key = self._create_key("session", session_id)
        return await self.set(key, session.dict(), rag_config.session_cache_ttl)
    
    async def get_cached_session(self, session_id: str) -> Optional[ChatSession]:
        """Get cached session"""
        key = self._create_key("session", session_id)
        session_data = await self.get(key)
        if session_data:
            return ChatSession(**session_data)
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete cached session"""
        key = self._create_key("session", session_id)
        return await self.delete(key)
    
    # Cache Management
    async def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(f"{rag_config.cache_prefix}:{pattern}*")
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error for pattern {pattern}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            info = self.redis_client.info()
            
            # Count keys by type
            key_counts = {}
            for prefix in ["response", "embedding", "search", "session"]:
                pattern = f"{rag_config.cache_prefix}:{prefix}:*"
                keys = self.redis_client.keys(pattern)
                key_counts[f"{prefix}_keys"] = len(keys)
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                **key_counts
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            if not self.redis_client:
                # Try to reconnect to redis
                self._initialize_redis()
            return self.redis_client.ping() if self.redis_client else False
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

# Global cache service instance
cache_service = CacheService()


# Utility functions for creating cache keys
def create_query_hash(query: str, strategy: str, top_k: int, **kwargs) -> str:
    """Create hash for RAG query caching"""
    query_data = {
        "query": query.lower().strip(),
        "strategy": strategy,
        "top_k": top_k,
        **kwargs
    }
    return cache_service._hash_object(query_data)


def create_text_hash(text: str, model: str = None) -> str:
    """Create hash for text embedding caching"""
    text_data = {
        "text": text.strip(),
        "model": model or rag_config.embedding_model
    }
    return cache_service._hash_object(text_data)


def create_search_hash(query_embedding: List[float], top_k: int, filters: Dict = None) -> str:
    """Create hash for search results caching"""
    # Use first and last few embedding values for hash (full embedding is too large)
    embedding_sample = query_embedding[:5] + query_embedding[-5:]
    search_data = {
        "embedding_sample": embedding_sample,
        "top_k": top_k,
        "filters": filters or {}
    }
    return cache_service._hash_object(search_data)