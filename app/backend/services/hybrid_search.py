"""
Hybrid Search Service combining Vector and Keyword Search
Implements BM25 keyword search and combines with vector search using fusion algorithms
"""

import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import logging
import re
from collections import defaultdict

from config.rag_config import rag_config
from models.rag_models import SearchResult, SourceType
from services.vector_search import vector_search_service
from services.fusion import fusion_service


logger = logging.getLogger(__name__)


class HybridSearchService:
    """Service for hybrid search combining vector similarity and keyword matching"""
    
    def __init__(self):
        self.bm25_index = None
        self.document_corpus = []
        self.document_metadata = []
        self._index_initialized = False
    
    async def initialize_bm25_index(self, force_rebuild: bool = False):
        """
        Initialize BM25 index from Pinecone data
        
        Args:
            force_rebuild: Whether to force rebuild the index
        """
        if self._index_initialized and not force_rebuild:
            logger.info("BM25 index already initialized")
            return
        
        try:
            logger.info("Initializing BM25 index from Pinecone data...")
            start_time = time.time()
            
            # Check if vector search service is healthy first
            if not await vector_search_service.health_check():
                logger.error("Vector search service is not healthy - cannot initialize BM25 index")
                return
            
            # Get all documents from Pinecone index
            stats = await vector_search_service.get_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            logger.info(f"Found {total_vectors} vectors in Pinecone index")
            
            if total_vectors == 0:
                logger.warning("No vectors found in Pinecone index - BM25 index will be empty")
                # Still mark as initialized to prevent repeated attempts
                self._index_initialized = True
                return
            
            # For now, we'll build the BM25 index on-demand during searches
            # In production, consider pre-building and caching the index
            logger.info(f"BM25 index initialization completed in {(time.time() - start_time):.2f}s")
            self._index_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            # Don't raise the exception - allow the service to continue with vector-only search
            logger.warning("Hybrid search will fall back to vector-only search")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into tokens and remove empty strings
        tokens = [token for token in text.split() if token and len(token) > 1]
        
        return tokens
    
    async def keyword_search(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: int = 10
    ) -> Tuple[List[SearchResult], float]:
        """
        Perform BM25 keyword search on provided documents
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results to return
            
        Returns:
            Tuple of (search results, search time in ms)
        """
        start_time = time.time()
        
        if not documents:
            return [], 0.0
        
        try:
            # Prepare corpus for BM25
            corpus = []
            for doc in documents:
                tokens = self._preprocess_text(doc.text)
                corpus.append(tokens)
            
            # Create BM25 index
            bm25 = BM25Okapi(corpus)
            
            # Preprocess query
            query_tokens = self._preprocess_text(query)
            
            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)
            
            # Create results with BM25 scores
            scored_results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                if score > 0:  # Only include documents with positive scores
                    keyword_result = SearchResult(
                        id=doc.id,
                        text=doc.text,
                        score=float(score),
                        source_type=doc.source_type,
                        metadata={
                            **doc.metadata,
                            'original_score': doc.score,
                            'bm25_score': float(score),
                            'search_method': 'bm25'
                        },
                        section=doc.section,
                        pages=doc.pages,
                        strategy=doc.strategy,
                        title=doc.title,
                        url=doc.url
                    )
                    scored_results.append(keyword_result)
            
            # Sort by BM25 score and return top_k
            scored_results.sort(key=lambda x: x.score, reverse=True)
            final_results = scored_results[:top_k]
            
            search_time = (time.time() - start_time) * 1000
            logger.info(f"BM25 keyword search completed in {search_time:.2f}ms, found {len(final_results)} results")
            
            return final_results, search_time
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return [], (time.time() - start_time) * 1000
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = None,
        keyword_weight: float = None,
        fusion_method: str = 'rrf'
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            top_k: Number of final results to return
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            fusion_method: Method to combine results ('rrf', 'weighted_score', 'borda_count')
            
        Returns:
            Tuple of (hybrid search results, timing information)
        """
        start_time = time.time()
        
        # Use config defaults if weights not provided
        v_weight = vector_weight if vector_weight is not None else rag_config.vector_weight
        k_weight = keyword_weight if keyword_weight is not None else rag_config.keyword_weight
        
        try:
            # Perform vector search first to get candidate documents
            vector_results, vector_time = await vector_search_service.vector_search(
                query=query,
                top_k=top_k * 2  # Get more results for keyword search
            )
            
            if not vector_results:
                logger.warning("No vector search results found")
                return [], {'total_time': (time.time() - start_time) * 1000}
            
            # Perform keyword search on the vector search results
            keyword_results, keyword_time = await self.keyword_search(
                query=query,
                documents=vector_results,
                top_k=top_k * 2
            )
            
            # Combine results using specified fusion method
            if fusion_method == 'rrf':
                combined_results, fusion_time = await fusion_service.reciprocal_rank_fusion(
                    result_lists=[vector_results[:top_k], keyword_results[:top_k]],
                    weights=[v_weight, k_weight]
                )
            elif fusion_method == 'weighted_score':
                combined_results, fusion_time = await fusion_service.weighted_score_fusion(
                    result_lists=[vector_results[:top_k], keyword_results[:top_k]],
                    weights=[v_weight, k_weight]
                )
            elif fusion_method == 'borda_count':
                combined_results, fusion_time = await fusion_service.borda_count_fusion(
                    result_lists=[vector_results[:top_k], keyword_results[:top_k]],
                    weights=[v_weight, k_weight]
                )
            else:
                logger.warning(f"Unknown fusion method: {fusion_method}, using RRF")
                combined_results, fusion_time = await fusion_service.reciprocal_rank_fusion(
                    result_lists=[vector_results[:top_k], keyword_results[:top_k]],
                    weights=[v_weight, k_weight]
                )
            
            # Return top_k results
            final_results = combined_results[:top_k]
            
            total_time = (time.time() - start_time) * 1000
            
            timing_info = {
                'vector_time': vector_time,
                'keyword_time': keyword_time,
                'fusion_time': fusion_time,
                'total_time': total_time
            }
            
            logger.info(f"Hybrid search completed in {total_time:.2f}ms with {len(final_results)} results")
            
            return final_results, timing_info
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return [], {'total_time': (time.time() - start_time) * 1000}
    
    async def multi_strategy_search(
        self,
        query: str,
        strategies: List[str],
        top_k: int = 10
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform search using multiple strategies and combine results
        
        Args:
            query: Search query
            strategies: List of search strategies to use
            top_k: Number of final results to return
            
        Returns:
            Tuple of (combined results, detailed timing and metadata)
        """
        start_time = time.time()
        
        available_strategies = {
            'vector': self._vector_only_search,
            'keyword': self._keyword_only_search,
            'hybrid_rrf': lambda q, k: self.hybrid_search(q, k, fusion_method='rrf'),
            'hybrid_weighted': lambda q, k: self.hybrid_search(q, k, fusion_method='weighted_score'),
            'hybrid_borda': lambda q, k: self.hybrid_search(q, k, fusion_method='borda_count')
        }
        
        # Filter valid strategies
        valid_strategies = [s for s in strategies if s in available_strategies]
        if not valid_strategies:
            logger.warning(f"No valid strategies found in: {strategies}")
            return [], {}
        
        try:
            # Execute all strategies concurrently
            strategy_tasks = []
            for strategy in valid_strategies:
                task = asyncio.create_task(
                    available_strategies[strategy](query, top_k)
                )
                strategy_tasks.append((strategy, task))
            
            # Collect results from all strategies
            strategy_results = []
            strategy_metadata = {}
            
            for strategy, task in strategy_tasks:
                try:
                    results, metadata = await task
                    strategy_results.append(results)
                    strategy_metadata[strategy] = metadata
                except Exception as e:
                    logger.error(f"Strategy {strategy} failed: {e}")
                    strategy_results.append([])
                    strategy_metadata[strategy] = {'error': str(e)}
            
            # Combine all results using RRF
            if strategy_results:
                combined_results, fusion_time = await fusion_service.reciprocal_rank_fusion(
                    result_lists=strategy_results
                )
                final_results = combined_results[:top_k]
            else:
                final_results = []
                fusion_time = 0.0
            
            total_time = (time.time() - start_time) * 1000
            
            metadata = {
                'strategies_used': valid_strategies,
                'strategy_metadata': strategy_metadata,
                'fusion_time': fusion_time,
                'total_time': total_time,
                'total_results': len(final_results)
            }
            
            logger.info(f"Multi-strategy search completed in {total_time:.2f}ms using {len(valid_strategies)} strategies")
            
            return final_results, metadata
            
        except Exception as e:
            logger.error(f"Multi-strategy search failed: {e}")
            return [], {'error': str(e), 'total_time': (time.time() - start_time) * 1000}
    
    async def _vector_only_search(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict[str, float]]:
        """Helper method for vector-only search"""
        results, search_time = await vector_search_service.vector_search(query, top_k)
        return results, {'search_time': search_time}
    
    async def _keyword_only_search(self, query: str, top_k: int) -> Tuple[List[SearchResult], Dict[str, float]]:
        """Helper method for keyword-only search"""
        # First get documents from vector search, then apply keyword search
        vector_results, _ = await vector_search_service.vector_search(query, top_k * 3)
        keyword_results, search_time = await self.keyword_search(query, vector_results, top_k)
        return keyword_results, {'search_time': search_time}
    
    async def get_search_statistics(self, query: str) -> Dict[str, Any]:
        """
        Get detailed statistics about search performance across different methods
        
        Args:
            query: Search query to analyze
            
        Returns:
            Dictionary with performance statistics
        """
        try:
            # Test different search methods
            vector_results, vector_time = await vector_search_service.vector_search(query, 10)
            
            if vector_results:
                keyword_results, keyword_time = await self.keyword_search(query, vector_results, 10)
                hybrid_results, hybrid_timing = await self.hybrid_search(query, 10)
            else:
                keyword_results, keyword_time = [], 0.0
                hybrid_results, hybrid_timing = [], {}
            
            stats = {
                'query': query,
                'vector_search': {
                    'results_count': len(vector_results),
                    'search_time_ms': vector_time,
                    'avg_score': sum(r.score for r in vector_results) / len(vector_results) if vector_results else 0
                },
                'keyword_search': {
                    'results_count': len(keyword_results),
                    'search_time_ms': keyword_time,
                    'avg_score': sum(r.score for r in keyword_results) / len(keyword_results) if keyword_results else 0
                },
                'hybrid_search': {
                    'results_count': len(hybrid_results),
                    'timing': hybrid_timing,
                    'avg_score': sum(r.score for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> bool:
        """Check if the hybrid search service is healthy"""
        try:
            # Test vector search component
            vector_healthy = await vector_search_service.health_check()
            
            # Test fusion component
            fusion_healthy = await fusion_service.health_check()
            
            # Test BM25 functionality
            test_docs = [
                SearchResult(id="1", text="test document one", score=0.9, source_type=SourceType.DOCUMENT, metadata={}),
                SearchResult(id="2", text="another test document", score=0.8, source_type=SourceType.DOCUMENT, metadata={})
            ]
            
            bm25_results, _ = await self.keyword_search("test", test_docs, 2)
            bm25_healthy = len(bm25_results) > 0
            
            return vector_healthy and fusion_healthy and bm25_healthy
            
        except Exception as e:
            logger.error(f"Hybrid search service health check failed: {e}")
            return False


# Global instance
hybrid_search_service = HybridSearchService()
