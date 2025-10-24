import os
import hashlib
import logging
from typing import Optional
from openai import OpenAI

from config.rag_config import rag_config
from services.cache_service import cache_service

logger = logging.getLogger(__name__)

QUERY_REWRITING_PROMPT = """Rewrite this query to be more effective for financial document retrieval. Focus on making it more specific, adding relevant financial terminology, and improving search effectiveness for vector and hybrid search.

Guidelines:
- Keep the core meaning and intent of the original query
- Add relevant financial terminology and concepts
- Make the query more specific and searchable
- Consider MATLAB Financial Toolbox functions and concepts
- Maintain natural language flow

Return only the rewritten query, no explanations."""

class QueryRewriter:
    """Service for rewriting queries with Redis caching"""
    
    def __init__(self):
        self.client = OpenAI(api_key=rag_config.openai_api_key)
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    def _create_query_hash(self, query: str) -> str:
        """Create a hash for the original query for caching"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    async def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite a query to be more effective for financial document retrieval
        
        Args:
            original_query: The original user query
            
        Returns:
            Rewritten query optimized for retrieval
        """
        if not original_query or not original_query.strip():
            return original_query
        
        try:
            # Create cache key from original query
            cache_key = f"query_rewrite:{self._create_query_hash(original_query)}"
            
            # Check cache first
            cached_rewrite = await cache_service.get(cache_key)
            if cached_rewrite:
                logger.info(f"Using cached rewrite for query: {original_query[:50]}...")
                return cached_rewrite
            
            # Rewrite query using OpenAI
            logger.info(f"Rewriting query: {original_query[:50]}...")
            response = self.client.chat.completions.create(
                model=rag_config.generation_model,
                messages=[
                    {"role": "system", "content": QUERY_REWRITING_PROMPT},
                    {"role": "user", "content": original_query}
                ],
                temperature=0.3,
                max_tokens=200,
                timeout=rag_config.request_timeout
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            # Validate rewrite
            if not rewritten_query or rewritten_query == original_query:
                logger.warning(f"Query rewrite returned empty or unchanged result for: {original_query}")
                rewritten_query = original_query
            
            # Cache the result
            await cache_service.set(cache_key, rewritten_query, self.cache_ttl)
            
            logger.info(f"Query rewritten successfully: {original_query[:30]}... -> {rewritten_query[:30]}...")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Query rewriting failed for '{original_query}': {e}")
            # Fallback to original query
            return original_query
    
    async def get_rewrite_stats(self) -> dict:
        """Get statistics about query rewriting"""
        try:
            # This would require additional cache tracking
            return {
                "cache_enabled": cache_service.redis_client is not None,
                "cache_ttl": self.cache_ttl
            }
        except Exception as e:
            logger.error(f"Failed to get rewrite stats: {e}")
            return {"error": str(e)}


# Global query rewriter instance
query_rewriter = QueryRewriter()