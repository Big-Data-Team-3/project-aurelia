"""
Reranking Service using Cross-Encoder Models
Improves search result relevance through neural reranking
"""

import time
import asyncio
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import logging
import numpy as np

from config.rag_config import rag_config
from models.rag_models import SearchResult


logger = logging.getLogger(__name__)


class RerankingService:
    """Service for reranking search results using cross-encoder models"""
    
    def __init__(self):
        self.model = None
        self.model_name = rag_config.reranker_model
        self._model_loading = False
        self._model_loaded = False
        # Don't load model during __init__ to avoid blocking startup
        if rag_config.enable_reranking:
            # Start loading model in background
            asyncio.create_task(self._load_model_async())
    
    async def _load_model_async(self):
        """Load the reranking model asynchronously"""
        if self._model_loading or self._model_loaded:
            return
        
        self._model_loading = True
        try:
            logger.info(f"Loading reranking model asynchronously: {self.model_name}")
            import os
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'  # Disable telemetry
            
            # Load model in a separate thread to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(CrossEncoder, self.model_name)
                self.model = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            self._model_loaded = True
            logger.info("Reranking model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            logger.warning("Reranking will be disabled - search results will not be reranked")
            self.model = None
        finally:
            self._model_loading = False
    
    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        Rerank search results using cross-encoder model
        
        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return (None for all)
            
        Returns:
            Tuple of (reranked results, reranking time in ms)
        """
        start_time = time.time()
        
        # Wait for model to load if it's still loading
        if self._model_loading and not self._model_loaded:
            logger.info("Waiting for reranking model to load...")
            while self._model_loading and not self._model_loaded:
                await asyncio.sleep(0.1)
        
        if not self.model or not results:
            logger.warning("Reranking model not available or no results to rerank")
            return results[:top_k] if top_k else results, 0.0
        
        try:
            # Prepare query-document pairs for the cross-encoder
            query_doc_pairs = []
            for result in results:
                # Use the text content for reranking
                query_doc_pairs.append([query, result.text])
            
            # Get relevance scores from cross-encoder
            relevance_scores = self.model.predict(query_doc_pairs)
            
            # Create list of (result, score) tuples
            result_score_pairs = list(zip(results, relevance_scores))
            
            # Sort by relevance score (descending)
            result_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update results with new scores and return top_k
            reranked_results = []
            for result, score in result_score_pairs:
                # Create new result with updated score
                reranked_result = SearchResult(
                    id=result.id,
                    text=result.text,
                    score=float(score),  # Use reranking score
                    source_type=result.source_type,
                    metadata={
                        **result.metadata,
                        'original_score': result.score,  # Keep original score
                        'rerank_score': float(score)
                    },
                    section=result.section,
                    pages=result.pages,
                    strategy=result.strategy,
                    title=result.title,
                    url=result.url
                )
                reranked_results.append(reranked_result)
            
            # Return top_k results if specified
            final_results = reranked_results[:top_k] if top_k else reranked_results
            
            rerank_time = (time.time() - start_time) * 1000
            logger.info(f"Reranked {len(results)} results in {rerank_time:.2f}ms, returning top {len(final_results)}")
            
            return final_results, rerank_time
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results if reranking fails
            return results[:top_k] if top_k else results, 0.0
    
    async def batch_rerank(
        self,
        queries: List[str],
        result_lists: List[List[SearchResult]],
        top_k: Optional[int] = None
    ) -> Tuple[List[List[SearchResult]], float]:
        """
        Rerank multiple result lists for multiple queries
        
        Args:
            queries: List of search queries
            result_lists: List of result lists corresponding to each query
            top_k: Number of top results to return per query
            
        Returns:
            Tuple of (list of reranked result lists, total reranking time in ms)
        """
        start_time = time.time()
        
        if not self.model:
            logger.warning("Reranking model not available")
            return result_lists, 0.0
        
        try:
            # Process all queries concurrently
            rerank_tasks = []
            for query, results in zip(queries, result_lists):
                task = asyncio.create_task(
                    self.rerank_results(query, results, top_k)
                )
                rerank_tasks.append(task)
            
            # Wait for all reranking tasks to complete
            reranked_data = await asyncio.gather(*rerank_tasks)
            
            # Extract results and sum up times
            reranked_lists = [data[0] for data in reranked_data]
            total_time = sum(data[1] for data in reranked_data)
            
            logger.info(f"Batch reranked {len(queries)} query result lists in {total_time:.2f}ms")
            
            return reranked_lists, total_time
            
        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            return result_lists, 0.0
    
    async def get_relevance_scores(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Get relevance scores for texts without reranking SearchResult objects
        
        Args:
            query: Search query
            texts: List of text documents
            
        Returns:
            List of relevance scores
        """
        if not self.model or not texts:
            return [0.0] * len(texts)
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [[query, text] for text in texts]
            
            # Get relevance scores
            scores = self.model.predict(query_doc_pairs)
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Failed to get relevance scores: {e}")
            return [0.0] * len(texts)
    
    async def filter_by_relevance_threshold(
        self,
        query: str,
        results: List[SearchResult],
        threshold: float = 0.5
    ) -> Tuple[List[SearchResult], float]:
        """
        Filter results by relevance threshold using cross-encoder
        
        Args:
            query: Search query
            results: List of search results
            threshold: Minimum relevance threshold
            
        Returns:
            Tuple of (filtered results, processing time in ms)
        """
        start_time = time.time()
        
        if not self.model or not results:
            return results, 0.0
        
        try:
            # Get relevance scores
            texts = [result.text for result in results]
            scores = await self.get_relevance_scores(query, texts)
            
            # Filter results by threshold
            filtered_results = []
            for result, score in zip(results, scores):
                if score >= threshold:
                    # Update result with relevance score
                    filtered_result = SearchResult(
                        id=result.id,
                        text=result.text,
                        score=float(score),
                        source_type=result.source_type,
                        metadata={
                            **result.metadata,
                            'original_score': result.score,
                            'relevance_score': float(score)
                        },
                        section=result.section,
                        pages=result.pages,
                        strategy=result.strategy,
                        title=result.title,
                        url=result.url
                    )
                    filtered_results.append(filtered_result)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Filtered {len(results)} to {len(filtered_results)} results above threshold {threshold}")
            
            return filtered_results, processing_time
            
        except Exception as e:
            logger.error(f"Relevance filtering failed: {e}")
            return results, 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the loaded reranking model"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'reranking_enabled': rag_config.enable_reranking
        }
    
    async def health_check(self) -> bool:
        """Check if the reranking service is healthy"""
        try:
            if not rag_config.enable_reranking:
                return True  # Service is "healthy" if disabled
            
            # If model is still loading, consider it healthy
            if self._model_loading and not self._model_loaded:
                logger.info("Reranking model is still loading - service is healthy but reranking not yet available")
                return True
            
            # If model failed to load, consider it healthy but disabled
            if not self.model:
                logger.warning("Reranking model not available - service will work without reranking")
                return True  # Return True to not fail overall health check
            
            # Test with a simple reranking task
            test_query = "test query"
            test_texts = ["relevant text", "irrelevant content"]
            scores = await self.get_relevance_scores(test_query, test_texts)
            
            return len(scores) == 2 and all(isinstance(s, (int, float)) for s in scores)
            
        except Exception as e:
            logger.error(f"Reranking service health check failed: {e}")
            # Don't fail the health check if reranking is unavailable
            return True


# Global instance
reranking_service = RerankingService()
