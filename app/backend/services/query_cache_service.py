"""
Query Cache Service
Handles caching of complete query responses with performance tracking
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta

from models.rag_models import RAGQuery, RAGResponse
from services.cache_service import cache_service
from config.rag_config import rag_config

logger = logging.getLogger(__name__)

class QueryCacheService:
    """Service for caching complete query responses with performance metrics"""
    
    def __init__(self):
        self.cache_prefix = "query_cache"
        self.default_ttl = 3600  # 1 hour default TTL
        self.performance_ttl = 86400  # 24 hours for performance data
        
    def _create_query_hash(self, query: str, strategy: str = None, top_k: int = None) -> str:
        """Create a deterministic hash for the query"""
        # Include key parameters that affect the response
        cache_key = f"{query}|{strategy}|{top_k}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _create_cache_key(self, query_hash: str) -> str:
        """Create the full cache key"""
        return f"{self.cache_prefix}:{query_hash}"
    
    def _create_performance_key(self, query_hash: str) -> str:
        """Create performance tracking key"""
        return f"{self.cache_prefix}:perf:{query_hash}"
    
    async def get_cached_response(self, rag_query: RAGQuery) -> Optional[Tuple[RAGResponse, Dict[str, Any]]]:
        """
        Get cached response for a query
        
        Args:
            rag_query: The RAG query to check for cache
            
        Returns:
            Tuple of (cached_response, performance_data) if found, None otherwise
        """
        try:
            query_hash = self._create_query_hash(
                rag_query.query, 
                rag_query.strategy.value if rag_query.strategy else None,
                rag_query.top_k
            )
            
            cache_key = self._create_cache_key(query_hash)
            perf_key = self._create_performance_key(query_hash)
            
            # Get cached response
            cached_data = await cache_service.get(cache_key)
            if not cached_data:
                logger.info(f"No cached response found for query: {rag_query.query[:50]}...")
                return None
            
            # Get performance data
            perf_data = await cache_service.get(perf_key)
            if not perf_data:
                logger.warning(f"Cached response found but no performance data for query: {rag_query.query[:50]}...")
                perf_data = {}
            
            # Reconstruct RAGResponse from cached data
            cached_response = RAGResponse(**cached_data)
            
            logger.info(f"Cache HIT for query: {rag_query.query[:50]}...")
            return cached_response, perf_data
            
        except Exception as e:
            logger.error(f"Error retrieving cached response: {e}")
            return None
    
    async def cache_response(
        self, 
        rag_query: RAGQuery, 
        response: RAGResponse, 
        processing_time_ms: float,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a query response with performance data
        
        Args:
            rag_query: The original query
            response: The response to cache
            processing_time_ms: Time taken to generate the response
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if caching was successful
        """
        try:
            query_hash = self._create_query_hash(
                rag_query.query,
                rag_query.strategy.value if rag_query.strategy else None,
                rag_query.top_k
            )
            
            cache_key = self._create_cache_key(query_hash)
            perf_key = self._create_performance_key(query_hash)
            
            # Prepare response data for caching (convert to dict)
            response_data = response.dict()
            
            # Prepare performance data
            performance_data = {
                'query': rag_query.query,
                'strategy': rag_query.strategy.value if rag_query.strategy else None,
                'top_k': rag_query.top_k,
                'processing_time_ms': processing_time_ms,
                'cached_at': datetime.now().isoformat(),
                'response_size': len(json.dumps(response_data)),
                'sources_count': len(response.sources) if response.sources else 0,
                'query_type': response.metadata.get('query_classification', 'unknown'),
                'wikipedia_fallback_used': response.metadata.get('wikipedia_fallback_used', False)
            }
            
            # Cache the response
            cache_ttl = ttl or self.default_ttl
            response_cached = await cache_service.set(cache_key, response_data, cache_ttl)
            
            # Cache performance data (longer TTL)
            perf_cached = await cache_service.set(perf_key, performance_data, self.performance_ttl)
            
            if response_cached and perf_cached:
                logger.info(f"Cached response for query: {rag_query.query[:50]}... (TTL: {cache_ttl}s)")
                return True
            else:
                logger.error(f"Failed to cache response for query: {rag_query.query[:50]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            # Get all cache keys
            cache_keys = await cache_service.get_all_keys(self.cache_prefix)
            perf_keys = await cache_service.get_all_keys(f"{self.cache_prefix}:perf")
            
            # Count cached responses
            cached_responses = len([k for k in cache_keys if not k.endswith(':perf')])
            
            # Calculate average performance data
            total_processing_time = 0
            total_responses = 0
            query_types = {}
            wikipedia_fallbacks = 0
            
            for perf_key in perf_keys:
                perf_data = await cache_service.get(perf_key)
                if perf_data:
                    total_processing_time += perf_data.get('processing_time_ms', 0)
                    total_responses += 1
                    
                    query_type = perf_data.get('query_type', 'unknown')
                    query_types[query_type] = query_types.get(query_type, 0) + 1
                    
                    if perf_data.get('wikipedia_fallback_used', False):
                        wikipedia_fallbacks += 1
            
            avg_processing_time = total_processing_time / total_responses if total_responses > 0 else 0
            
            return {
                'total_cached_responses': cached_responses,
                'total_performance_records': len(perf_keys),
                'average_processing_time_ms': round(avg_processing_time, 2),
                'query_type_distribution': query_types,
                'wikipedia_fallback_usage': wikipedia_fallbacks,
                'cache_hit_potential': f"{cached_responses} queries ready for instant response"
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    async def clear_cache(self, query_pattern: Optional[str] = None) -> bool:
        """
        Clear cache entries
        
        Args:
            query_pattern: Optional pattern to match specific queries
            
        Returns:
            True if clearing was successful
        """
        try:
            if query_pattern:
                # Clear specific queries matching pattern
                cache_keys = await cache_service.get_all_keys(self.cache_prefix)
                pattern_keys = [k for k in cache_keys if query_pattern.lower() in k.lower()]
                
                for key in pattern_keys:
                    await cache_service.delete(key)
                
                logger.info(f"Cleared {len(pattern_keys)} cache entries matching pattern: {query_pattern}")
            else:
                # Clear all cache entries
                cache_keys = await cache_service.get_all_keys(self.cache_prefix)
                
                for key in cache_keys:
                    await cache_service.delete(key)
                
                logger.info(f"Cleared all {len(cache_keys)} cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if the cache service is working"""
        try:
            # Test cache operations
            test_key = f"{self.cache_prefix}:health_test"
            test_data = {"test": True, "timestamp": time.time()}
            
            # Test set
            set_result = await cache_service.set(test_key, test_data, 60)
            if not set_result:
                return False
            
            # Test get
            retrieved_data = await cache_service.get(test_key)
            if not retrieved_data or retrieved_data.get("test") != True:
                return False
            
            # Clean up
            await cache_service.delete(test_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Query cache health check failed: {e}")
            return False


# Global instance
query_cache_service = QueryCacheService()
