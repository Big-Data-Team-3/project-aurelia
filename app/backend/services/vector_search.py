"""
Vector Search Service using Pinecone
Handles vector similarity search with ANN optimizations
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import logging

from config.rag_config import rag_config
from models.rag_models import SearchResult, SourceType


logger = logging.getLogger(__name__)


class VectorSearchService:
    """Service for vector-based similarity search using Pinecone"""
    
    def __init__(self, config=None):
        self.config = config or rag_config
        self.openai_client = None
        self.pinecone_client = None
        self.index = None
        self._initialized = False
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenAI and Pinecone clients"""
        try:
            # Check if API keys are available
            if not self.config.openai_api_key:
                logger.warning("OpenAI API key not found - vector search will be disabled")
                return
                
            if not self.config.pinecone_api_key:
                logger.warning("Pinecone API key not found - vector search will be disabled")
                return
            
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=self.config.pinecone_api_key)
            self.index = self.pinecone_client.Index(name=self.config.pinecone_index_name)
            logger.info(f"Pinecone client initialized for index: {self.config.pinecone_index_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            # Don't raise - allow service to start without vector search
            self._initialized = False
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text using OpenAI"""
        if not self._initialized or not self.openai_client:
            raise RuntimeError("Vector search service not initialized - missing API keys")
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.config.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts in batch"""
        if not self._initialized or not self.openai_client:
            raise RuntimeError("Vector search service not initialized - missing API keys")
            
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.config.embedding_model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to create batch embeddings: {e}")
            raise
    
    async def vector_search(
        self, 
        query: str, 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            Tuple of (search results, search time in ms)
        """
        if not self._initialized or not self.index:
            logger.warning("Vector search service not initialized - returning empty results")
            return [], 0.0
            
        start_time = time.time()
        
        try:
            # Create embedding for query
            query_embedding = await self.create_embedding(query)
            
            # Perform vector search with ANN optimization
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            results = []
            for match in search_results.matches:
                metadata = match.metadata or {}
                
                # Extract text content
                text = metadata.get('text', '')
                
                # Parse JSON fields if they exist
                pages = self._safe_json_parse(metadata.get('pages', '[]'))
                content_types = self._safe_json_parse(metadata.get('content_types', '[]'))
                hierarchy = self._safe_json_parse(metadata.get('hierarchy', '[]'))
                
                result = SearchResult(
                    id=match.id,
                    text=text,
                    score=float(match.score),
                    source_type=SourceType.DOCUMENT,
                    metadata={
                        'strategy': metadata.get('strategy', ''),
                        'section': metadata.get('section', ''),
                        'content_types': content_types,
                        'hierarchy': hierarchy,
                        'is_special': metadata.get('is_special', False),
                        'full_text_length': metadata.get('full_text_length', 0)
                    },
                    section=metadata.get('section'),
                    pages=pages,
                    strategy=metadata.get('strategy')
                )
                results.append(result)
            
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Vector search completed in {search_time:.2f}ms, found {len(results)} results")
            
            return results, search_time
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def _safe_json_parse(self, json_str: str) -> List:
        """Safely parse JSON string, return empty list if parsing fails"""
        try:
            if isinstance(json_str, str):
                return json.loads(json_str)
            elif isinstance(json_str, list):
                return json_str
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []
    
    async def similarity_search_with_score_threshold(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform vector search with score threshold filtering
        
        Args:
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filter_dict: Optional metadata filters
            
        Returns:
            Tuple of (filtered search results, search time in ms)
        """
        results, search_time = await self.vector_search(
            query=query,
            top_k=top_k * 2,  # Get more results to filter
            filter_dict=filter_dict
        )
        
        # Filter by score threshold
        filtered_results = [
            result for result in results 
            if result.score >= score_threshold
        ][:top_k]  # Limit to top_k after filtering
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} above threshold {score_threshold}")
        
        return filtered_results, search_time
    
    async def multi_vector_search(
        self,
        queries: List[str],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[SearchResult]], float]:
        """
        Perform multiple vector searches concurrently
        
        Args:
            queries: List of search query texts
            top_k: Number of results per query
            filter_dict: Optional metadata filters
            
        Returns:
            Tuple of (list of search results for each query, total search time in ms)
        """
        start_time = time.time()
        
        # Create embeddings for all queries in batch
        query_embeddings = await self.create_embeddings_batch(queries)
        
        # Perform searches concurrently
        search_tasks = []
        for embedding in query_embeddings:
            task = asyncio.create_task(
                self._vector_search_with_embedding(embedding, top_k, filter_dict)
            )
            search_tasks.append(task)
        
        all_results = await asyncio.gather(*search_tasks)
        
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Multi-vector search completed in {search_time:.2f}ms for {len(queries)} queries")
        
        return all_results, search_time
    
    async def _vector_search_with_embedding(
        self,
        embedding: List[float],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Internal method to search with pre-computed embedding"""
        try:
            search_results = self.index.query(
                vector=embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            
            results = []
            for match in search_results.matches:
                metadata = match.metadata or {}
                text = metadata.get('text', '')
                pages = self._safe_json_parse(metadata.get('pages', '[]'))
                
                result = SearchResult(
                    id=match.id,
                    text=text,
                    score=float(match.score),
                    source_type=SourceType.DOCUMENT,
                    metadata=metadata,
                    section=metadata.get('section'),
                    pages=pages,
                    strategy=metadata.get('strategy')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search with embedding failed: {e}")
            return []
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check if the vector search service is healthy"""
        if not self._initialized:
            logger.warning("Vector search service not initialized - health check failed")
            return False
            
        try:
            # Test embedding creation
            test_embedding = await self.create_embedding("test query")
            if not test_embedding or len(test_embedding) != self.config.embedding_dimension:
                return False
            
            # Test index connection
            stats = await self.get_index_stats()
            if not stats:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global instance - lazy initialization
_vector_search_service = None

def get_vector_search_service():
    """Get or create the vector search service instance"""
    global _vector_search_service
    if _vector_search_service is None:
        _vector_search_service = VectorSearchService()
    return _vector_search_service

# For backward compatibility - this will be replaced with lazy initialization
class LazyVectorSearchService:
    def __getattr__(self, name):
        return getattr(get_vector_search_service(), name)

vector_search_service = LazyVectorSearchService()
