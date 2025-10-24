"""
Fusion Service implementing RRF (Reciprocal Rank Fusion) and other fusion algorithms
Combines results from multiple retrieval methods
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

from config.rag_config import rag_config
from models.rag_models import SearchResult


logger = logging.getLogger(__name__)


class FusionService:
    """Service for combining results from multiple retrieval methods"""
    
    def __init__(self):
        self.rrf_k = rag_config.rrf_k
    
    async def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        k: Optional[int] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion (RRF)
        
        RRF Score = sum(weight_i / (k + rank_i)) for each list where item appears
        
        Args:
            result_lists: List of search result lists to combine
            weights: Optional weights for each result list (default: equal weights)
            k: RRF parameter (default: from config)
            
        Returns:
            Tuple of (fused results, fusion time in ms)
        """
        start_time = time.time()
        
        if not result_lists or all(not results for results in result_lists):
            return [], 0.0
        
        k_param = k if k is not None else self.rrf_k
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            logger.warning(f"Weight count ({len(weights)}) doesn't match result list count ({len(result_lists)})")
            weights = [1.0] * len(result_lists)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / len(result_lists)] * len(result_lists)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        result_map = {}  # Map from result ID to SearchResult object
        
        for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
            for rank, result in enumerate(results, 1):
                result_id = result.id
                
                # Calculate RRF contribution from this list
                rrf_contribution = weight / (k_param + rank)
                rrf_scores[result_id] += rrf_contribution
                
                # Store the result object (prefer higher-ranked occurrences)
                if result_id not in result_map or rank < result_map[result_id][1]:
                    result_map[result_id] = (result, rank, list_idx)
        
        # Sort by RRF score and create final results
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for result_id, rrf_score in sorted_items:
            original_result, best_rank, source_list = result_map[result_id]
            
            # Create new result with RRF score
            fused_result = SearchResult(
                id=original_result.id,
                text=original_result.text,
                score=float(rrf_score),
                source_type=original_result.source_type,
                metadata={
                    **original_result.metadata,
                    'original_score': original_result.score,
                    'rrf_score': float(rrf_score),
                    'best_rank': best_rank,
                    'source_list': source_list,
                    'fusion_method': 'rrf'
                },
                section=original_result.section,
                pages=original_result.pages,
                strategy=original_result.strategy,
                title=original_result.title,
                url=original_result.url
            )
            fused_results.append(fused_result)
        
        fusion_time = (time.time() - start_time) * 1000
        logger.info(f"RRF fusion completed in {fusion_time:.2f}ms, combined {len(result_lists)} lists into {len(fused_results)} results")
        
        return fused_results, fusion_time
    
    async def weighted_score_fusion(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None,
        normalize_scores: bool = True
    ) -> Tuple[List[SearchResult], float]:
        """
        Combine results using weighted score fusion
        
        Args:
            result_lists: List of search result lists to combine
            weights: Optional weights for each result list
            normalize_scores: Whether to normalize scores before fusion
            
        Returns:
            Tuple of (fused results, fusion time in ms)
        """
        start_time = time.time()
        
        if not result_lists or all(not results for results in result_lists):
            return [], 0.0
        
        # Set default weights
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            weights = [1.0] * len(result_lists)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        # Normalize scores within each list if requested
        if normalize_scores:
            normalized_lists = []
            for results in result_lists:
                if not results:
                    normalized_lists.append([])
                    continue
                
                scores = [r.score for r in results]
                min_score, max_score = min(scores), max(scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized_results = []
                    for result in results:
                        normalized_score = (result.score - min_score) / score_range
                        normalized_result = SearchResult(
                            id=result.id,
                            text=result.text,
                            score=normalized_score,
                            source_type=result.source_type,
                            metadata=result.metadata,
                            section=result.section,
                            pages=result.pages,
                            strategy=result.strategy,
                            title=result.title,
                            url=result.url
                        )
                        normalized_results.append(normalized_result)
                    normalized_lists.append(normalized_results)
                else:
                    normalized_lists.append(results)
            result_lists = normalized_lists
        
        # Calculate weighted scores
        weighted_scores = defaultdict(float)
        result_map = {}
        
        for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
            for result in results:
                result_id = result.id
                weighted_scores[result_id] += result.score * weight
                
                # Store result object (prefer first occurrence)
                if result_id not in result_map:
                    result_map[result_id] = (result, list_idx)
        
        # Sort by weighted score and create final results
        sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for result_id, weighted_score in sorted_items:
            original_result, source_list = result_map[result_id]
            
            fused_result = SearchResult(
                id=original_result.id,
                text=original_result.text,
                score=float(weighted_score),
                source_type=original_result.source_type,
                metadata={
                    **original_result.metadata,
                    'original_score': original_result.score,
                    'weighted_score': float(weighted_score),
                    'source_list': source_list,
                    'fusion_method': 'weighted_score'
                },
                section=original_result.section,
                pages=original_result.pages,
                strategy=original_result.strategy,
                title=original_result.title,
                url=original_result.url
            )
            fused_results.append(fused_result)
        
        fusion_time = (time.time() - start_time) * 1000
        logger.info(f"Weighted score fusion completed in {fusion_time:.2f}ms")
        
        return fused_results, fusion_time
    
    async def borda_count_fusion(
        self,
        result_lists: List[List[SearchResult]],
        weights: Optional[List[float]] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        Combine results using Borda count method
        
        Args:
            result_lists: List of search result lists to combine
            weights: Optional weights for each result list
            
        Returns:
            Tuple of (fused results, fusion time in ms)
        """
        start_time = time.time()
        
        if not result_lists or all(not results for results in result_lists):
            return [], 0.0
        
        # Set default weights
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            weights = [1.0] * len(result_lists)
        
        # Calculate Borda scores
        borda_scores = defaultdict(float)
        result_map = {}
        
        for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
            list_length = len(results)
            
            for rank, result in enumerate(results):
                result_id = result.id
                
                # Borda score: (list_length - rank) * weight
                borda_contribution = (list_length - rank) * weight
                borda_scores[result_id] += borda_contribution
                
                # Store result object
                if result_id not in result_map:
                    result_map[result_id] = (result, list_idx)
        
        # Sort by Borda score and create final results
        sorted_items = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for result_id, borda_score in sorted_items:
            original_result, source_list = result_map[result_id]
            
            fused_result = SearchResult(
                id=original_result.id,
                text=original_result.text,
                score=float(borda_score),
                source_type=original_result.source_type,
                metadata={
                    **original_result.metadata,
                    'original_score': original_result.score,
                    'borda_score': float(borda_score),
                    'source_list': source_list,
                    'fusion_method': 'borda_count'
                },
                section=original_result.section,
                pages=original_result.pages,
                strategy=original_result.strategy,
                title=original_result.title,
                url=original_result.url
            )
            fused_results.append(fused_result)
        
        fusion_time = (time.time() - start_time) * 1000
        logger.info(f"Borda count fusion completed in {fusion_time:.2f}ms")
        
        return fused_results, fusion_time
    
    async def combine_with_deduplication(
        self,
        result_lists: List[List[SearchResult]],
        similarity_threshold: float = 0.9
    ) -> Tuple[List[SearchResult], float]:
        """
        Combine results with deduplication based on text similarity
        
        Args:
            result_lists: List of search result lists to combine
            similarity_threshold: Threshold for considering results as duplicates
            
        Returns:
            Tuple of (deduplicated results, processing time in ms)
        """
        start_time = time.time()
        
        if not result_lists:
            return [], 0.0
        
        # Flatten all results
        all_results = []
        for results in result_lists:
            all_results.extend(results)
        
        if not all_results:
            return [], 0.0
        
        # Simple deduplication based on text similarity
        deduplicated_results = []
        seen_texts = set()
        
        for result in all_results:
            # Use first 100 characters for similarity check
            text_signature = result.text[:100].lower().strip()
            
            if text_signature not in seen_texts:
                seen_texts.add(text_signature)
                deduplicated_results.append(result)
        
        # Sort by original scores
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Deduplication completed in {processing_time:.2f}ms, reduced {len(all_results)} to {len(deduplicated_results)} results")
        
        return deduplicated_results, processing_time
    
    def get_fusion_methods(self) -> List[str]:
        """Get list of available fusion methods"""
        return ['rrf', 'weighted_score', 'borda_count', 'deduplication']
    
    async def health_check(self) -> bool:
        """Check if the fusion service is healthy"""
        try:
            # Test with simple data
            test_results1 = [
                SearchResult(id="1", text="test1", score=0.9, source_type="document", metadata={})
            ]
            test_results2 = [
                SearchResult(id="2", text="test2", score=0.8, source_type="document", metadata={})
            ]
            
            fused, _ = await self.reciprocal_rank_fusion([test_results1, test_results2])
            return len(fused) == 2
            
        except Exception as e:
            logger.error(f"Fusion service health check failed: {e}")
            return False


# Global instance
fusion_service = FusionService()
