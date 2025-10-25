"""
AURELIA Lab 5 Evaluation Service
Core evaluation logic for quality, performance, and cost metrics
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the app directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from backend.services.generation import generation_service
from evaluation_models import (
    ConceptQualityMetrics, LatencyMetrics, TokenCostMetrics,
    EvaluationSummary, EvaluationResult, EvaluationConfig
)


class EvaluationService:
    """Core evaluation service for RAG system performance"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.embedding_model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model for semantic similarity"""
        try:
            # Use OpenAI embeddings for semantic similarity
            import openai
            self.embedding_model = "text-embedding-3-large"
        except ImportError:
            print("Warning: OpenAI not available, using TF-IDF for similarity")
            self.embedding_model = None
    
    async def evaluate_quality(
        self,
        generated_response: str,
        ground_truth: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> ConceptQualityMetrics:
        """
        Evaluate the quality of a generated response against ground truth
        
        Args:
            generated_response: Generated response from RAG system
            ground_truth: Ground truth data for the concept
            sources: Sources used in generation
            
        Returns:
            Quality metrics for the response
        """
        
        # 1. Accuracy Evaluation (40% weight)
        accuracy_score, accuracy_explanation = await self._evaluate_accuracy(
            generated_response, ground_truth
        )
        
        # 2. Completeness Evaluation (30% weight)
        completeness_score, completeness_details = await self._evaluate_completeness(
            generated_response, ground_truth
        )
        
        # 3. Citation Fidelity Evaluation (30% weight)
        citation_score, citation_details = await self._evaluate_citation_fidelity(
            sources, ground_truth
        )
        
        # Calculate overall quality score
        overall_score = (
            accuracy_score * self.config.accuracy_weight +
            completeness_score * self.config.completeness_weight +
            citation_score * self.config.citation_weight
        ) * 100
        
        return ConceptQualityMetrics(
            accuracy_score=accuracy_score,
            accuracy_explanation=accuracy_explanation,
            completeness_score=completeness_score,
            has_definition=completeness_details["has_definition"],
            has_key_components=completeness_details["has_key_components"],
            has_example=completeness_details["has_example"],
            has_use_cases=completeness_details["has_use_cases"],
            has_formula=completeness_details["has_formula"],
            completeness_explanation=completeness_details["explanation"],
            citation_fidelity_score=citation_score,
            expected_pages=citation_details["expected_pages"],
            actual_pages=citation_details["actual_pages"],
            citation_precision=citation_details["precision"],
            citation_recall=citation_details["recall"],
            citation_f1=citation_details["f1"],
            citation_explanation=citation_details["explanation"],
            overall_score=overall_score
        )
    
    async def _evaluate_accuracy(
        self, 
        generated_response: str, 
        ground_truth: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Evaluate semantic accuracy using embeddings"""
        
        try:
            # Extract definition from ground truth
            ground_truth_definition = ground_truth.get("definition", "")
            
            if not ground_truth_definition or not generated_response:
                return 0.0, "Missing ground truth definition or generated response"
            
            # Use OpenAI embeddings for semantic similarity
            if self.embedding_model:
                similarity_score = await self._calculate_semantic_similarity(
                    generated_response, ground_truth_definition
                )
            else:
                # Fallback to TF-IDF similarity
                similarity_score = self._calculate_tfidf_similarity(
                    generated_response, ground_truth_definition
                )
            
            explanation = f"Semantic similarity: {similarity_score:.3f} (0-1 scale)"
            return similarity_score, explanation
            
        except Exception as e:
            return 0.0, f"Accuracy evaluation failed: {str(e)}"
    
    async def _calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """Calculate semantic similarity using OpenAI embeddings"""
        try:
            import openai
            import os
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = openai.OpenAI(api_key=api_key)
            
            # Get embeddings
            response1 = client.embeddings.create(
                model=self.embedding_model,
                input=text1
            )
            response2 = client.embeddings.create(
                model=self.embedding_model,
                input=text2
            )
            
            embedding1 = np.array(response1.data[0].embedding)
            embedding2 = np.array(response2.data[0].embedding)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1), 
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"Semantic similarity calculation failed: {e}")
            # Fallback to TF-IDF
            return self._calculate_tfidf_similarity(text1, text2)
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF vectors"""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    async def _evaluate_completeness(
        self, 
        generated_response: str, 
        ground_truth: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate completeness of generated response"""
        
        try:
            # Check for key components
            has_definition = len(generated_response) >= 20  # Basic length check
            
            # Check for key components (look for lists, bullet points, etc.)
            has_key_components = self._check_for_key_components(generated_response)
            
            # Check for examples
            has_example = self._check_for_examples(generated_response)
            
            # Check for use cases
            has_use_cases = self._check_for_use_cases(generated_response)
            
            # Check for formulas (if applicable)
            has_formula = self._check_for_formulas(generated_response, ground_truth)
            
            # Calculate completeness score
            components_checked = [has_definition, has_key_components, has_example, has_use_cases]
            if ground_truth.get("formulas"):  # Only check formulas if they exist in ground truth
                components_checked.append(has_formula)
            
            completeness_score = sum(components_checked) / len(components_checked)
            
            explanation = f"Found {sum(components_checked)}/{len(components_checked)} required components"
            
            return completeness_score, {
                "has_definition": has_definition,
                "has_key_components": has_key_components,
                "has_example": has_example,
                "has_use_cases": has_use_cases,
                "has_formula": has_formula,
                "explanation": explanation
            }
            
        except Exception as e:
            return 0.0, {
                "has_definition": False,
                "has_key_components": False,
                "has_example": False,
                "has_use_cases": False,
                "has_formula": False,
                "explanation": f"Completeness evaluation failed: {str(e)}"
            }
    
    def _check_for_key_components(self, text: str) -> bool:
        """Check if text contains key components"""
        # Look for patterns that indicate structured information
        patterns = [
            r'\d+\.\s+',  # Numbered lists
            r'•\s+',      # Bullet points
            r'-\s+',      # Dashes
            r'\*\s+',     # Asterisks
            r'components?',  # Word "component"
            r'factors?',     # Word "factor"
            r'aspects?',     # Word "aspect"
            r'characteristics?',  # Word "characteristic"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for multiple sentences (indicating detailed explanation)
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if len(s.strip()) > 10]) >= 3
    
    def _check_for_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        example_indicators = [
            r'for example',
            r'for instance',
            r'such as',
            r'example:',
            r'instance:',
            r'consider',
            r'suppose',
            r'imagine',
            r'case in point'
        ]
        
        for indicator in example_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return True
        
        # Check for specific examples (numbers, percentages, etc.)
        if re.search(r'\d+%|\$\d+|\d+\.\d+', text):
            return True
        
        return False
    
    def _check_for_use_cases(self, text: str) -> bool:
        """Check if text contains use cases"""
        use_case_indicators = [
            r'use case',
            r'application',
            r'used for',
            r'useful for',
            r'purpose',
            r'benefit',
            r'advantage',
            r'when to use',
            r'where to apply',
            r'practical use'
        ]
        
        for indicator in use_case_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return True
        
        return False
    
    def _check_for_formulas(self, text: str, ground_truth: Dict[str, Any]) -> bool:
        """Check if text contains formulas (if applicable)"""
        if not ground_truth.get("formulas"):
            return True  # No formulas expected
        
        # Look for mathematical expressions
        formula_patterns = [
            r'[A-Za-z]\s*=\s*[A-Za-z0-9+\-*/()^√]+',  # Variable assignments
            r'[A-Za-z]\s*=\s*[A-Za-z0-9+\-*/()^√]+',  # Mathematical expressions
            r'[A-Za-z]\s*=\s*[A-Za-z0-9+\-*/()^√]+',  # Formulas
            r'formula',
            r'equation',
            r'calculation',
            r'compute',
            r'calculate'
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    async def _evaluate_citation_fidelity(
        self, 
        sources: List[Dict[str, Any]], 
        ground_truth: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate citation fidelity against ground truth page references"""
        
        try:
            # Extract expected pages from ground truth
            expected_pages = set(ground_truth.get("page_references", []))
            
            # Extract actual pages from sources
            actual_pages = set()
            for source in sources:
                if source.get("pages"):
                    if isinstance(source["pages"], list):
                        actual_pages.update(source["pages"])
                    else:
                        actual_pages.add(source["pages"])
            
            # Calculate precision and recall
            if not expected_pages:
                # No page references expected
                precision = 1.0 if not actual_pages else 0.0
                recall = 1.0
                f1 = 1.0 if not actual_pages else 0.0
            else:
                if not actual_pages:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                else:
                    true_positives = len(expected_pages.intersection(actual_pages))
                    precision = true_positives / len(actual_pages)
                    recall = true_positives / len(expected_pages)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            explanation = f"Expected pages: {sorted(expected_pages)}, Actual pages: {sorted(actual_pages)}"
            
            return f1, {
                "expected_pages": sorted(expected_pages),
                "actual_pages": sorted(actual_pages),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "explanation": explanation
            }
            
        except Exception as e:
            return 0.0, {
                "expected_pages": [],
                "actual_pages": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "explanation": f"Citation evaluation failed: {str(e)}"
            }
    
    async def estimate_token_cost(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> TokenCostMetrics:
        """Estimate token usage and costs"""
        
        try:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            query_tokens = len(query) // 4
            response_tokens = len(response) // 4
            
            # Estimate source tokens
            source_tokens = 0
            for source in sources:
                source_text = source.get("text", "")
                source_tokens += len(source_text) // 4
            
            total_tokens = query_tokens + response_tokens + source_tokens
            input_tokens = query_tokens + source_tokens
            output_tokens = response_tokens
            
            # Calculate costs
            embedding_cost = (total_tokens / 1000) * self.config.embedding_cost_per_1k_tokens
            generation_input_cost = (input_tokens / 1000) * self.config.generation_input_cost_per_1k_tokens
            generation_output_cost = (output_tokens / 1000) * self.config.generation_output_cost_per_1k_tokens
            
            total_cost = embedding_cost + generation_input_cost + generation_output_cost
            
            return TokenCostMetrics(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                embedding_cost_usd=embedding_cost,
                generation_cost_usd=generation_input_cost + generation_output_cost,
                total_cost_usd=total_cost,
                embedding_model=self.embedding_model or "text-embedding-3-large",
                generation_model="gpt-4o-mini"
            )
            
        except Exception as e:
            return TokenCostMetrics(
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                embedding_cost_usd=0.0,
                generation_cost_usd=0.0,
                total_cost_usd=0.0,
                embedding_model=self.embedding_model or "text-embedding-3-large",
                generation_model="gpt-4o-mini"
            )
    
    async def aggregate_results(
        self, 
        evaluation_results: List[EvaluationResult]
    ) -> EvaluationSummary:
        """Aggregate evaluation results into summary statistics"""
        
        try:
            successful_results = [r for r in evaluation_results if r.status.value == "success"]
            failed_results = [r for r in evaluation_results if r.status.value == "failed"]
            
            total_concepts = len(evaluation_results)
            successful_evaluations = len(successful_results)
            failed_evaluations = len(failed_results)
            
            if not successful_results:
                return EvaluationSummary(
                    evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    total_concepts=total_concepts,
                    successful_evaluations=successful_evaluations,
                    failed_evaluations=failed_evaluations,
                    overall_quality_score=0.0,
                    average_accuracy=0.0,
                    average_completeness=0.0,
                    average_citation_fidelity=0.0,
                    average_latency_ms=0.0,
                    average_retrieval_time_ms=0.0,
                    average_generation_time_ms=0.0,
                    cache_hit_rate=0.0,
                    cache_speedup_factor=1.0,
                    total_cost_usd=0.0,
                    average_cost_per_query_usd=0.0,
                    total_tokens_used=0,
                    duration_minutes=0.0
                )
            
            # Quality metrics
            quality_scores = [r.quality_metrics.overall_score for r in successful_results]
            accuracy_scores = [r.quality_metrics.accuracy_score for r in successful_results]
            completeness_scores = [r.quality_metrics.completeness_score for r in successful_results]
            citation_scores = [r.quality_metrics.citation_fidelity_score for r in successful_results]
            
            overall_quality_score = np.mean(quality_scores)
            average_accuracy = np.mean(accuracy_scores)
            average_completeness = np.mean(completeness_scores)
            average_citation_fidelity = np.mean(citation_scores)
            
            # Performance metrics
            latency_times = [r.latency_metrics.total_time_ms for r in successful_results]
            retrieval_times = [r.latency_metrics.retrieval_time_ms for r in successful_results]
            generation_times = [r.latency_metrics.generation_time_ms for r in successful_results]
            cache_hits = [r.latency_metrics.cache_hit for r in successful_results]
            
            average_latency_ms = np.mean(latency_times)
            average_retrieval_time_ms = np.mean(retrieval_times)
            average_generation_time_ms = np.mean(generation_times)
            cache_hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0.0
            
            # Calculate cache speedup
            cached_times = [r.latency_metrics.total_time_ms for r in successful_results if r.latency_metrics.cache_hit]
            fresh_times = [r.latency_metrics.total_time_ms for r in successful_results if not r.latency_metrics.cache_hit]
            
            if cached_times and fresh_times:
                cache_speedup_factor = np.mean(fresh_times) / np.mean(cached_times)
            else:
                cache_speedup_factor = 1.0
            
            # Cost metrics
            total_costs = [r.token_cost_metrics.total_cost_usd for r in successful_results]
            total_tokens = [r.token_cost_metrics.total_tokens for r in successful_results]
            
            total_cost_usd = sum(total_costs)
            average_cost_per_query_usd = np.mean(total_costs)
            total_tokens_used = sum(total_tokens)
            
            # Quality distribution
            high_quality_count = sum(1 for score in quality_scores if score >= self.config.high_quality_threshold)
            medium_quality_count = sum(1 for score in quality_scores if self.config.medium_quality_threshold <= score < self.config.high_quality_threshold)
            low_quality_count = sum(1 for score in quality_scores if score < self.config.medium_quality_threshold)
            
            # Performance distribution
            fast_responses_count = sum(1 for time in latency_times if time < self.config.fast_response_threshold_ms)
            medium_responses_count = sum(1 for time in latency_times if self.config.fast_response_threshold_ms <= time < self.config.slow_response_threshold_ms)
            slow_responses_count = sum(1 for time in latency_times if time >= self.config.slow_response_threshold_ms)
            
            return EvaluationSummary(
                evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                total_concepts=total_concepts,
                successful_evaluations=successful_evaluations,
                failed_evaluations=failed_evaluations,
                overall_quality_score=overall_quality_score,
                average_accuracy=average_accuracy,
                average_completeness=average_completeness,
                average_citation_fidelity=average_citation_fidelity,
                average_latency_ms=average_latency_ms,
                average_retrieval_time_ms=average_retrieval_time_ms,
                average_generation_time_ms=average_generation_time_ms,
                cache_hit_rate=cache_hit_rate,
                cache_speedup_factor=cache_speedup_factor,
                total_cost_usd=total_cost_usd,
                average_cost_per_query_usd=average_cost_per_query_usd,
                total_tokens_used=total_tokens_used,
                high_quality_count=high_quality_count,
                medium_quality_count=medium_quality_count,
                low_quality_count=low_quality_count,
                fast_responses_count=fast_responses_count,
                medium_responses_count=medium_responses_count,
                slow_responses_count=slow_responses_count,
                duration_minutes=0.0,  # Will be set by caller
                evaluation_config=self.config.dict()
            )
            
        except Exception as e:
            print(f"Error aggregating results: {e}")
            # Return minimal summary
            return EvaluationSummary(
                evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                total_concepts=len(evaluation_results),
                successful_evaluations=0,
                failed_evaluations=len(evaluation_results),
                overall_quality_score=0.0,
                average_accuracy=0.0,
                average_completeness=0.0,
                average_citation_fidelity=0.0,
                average_latency_ms=0.0,
                average_retrieval_time_ms=0.0,
                average_generation_time_ms=0.0,
                cache_hit_rate=0.0,
                cache_speedup_factor=1.0,
                total_cost_usd=0.0,
                average_cost_per_query_usd=0.0,
                total_tokens_used=0,
                duration_minutes=0.0
            )
    
    async def generate_markdown_report(self, summary: EvaluationSummary) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""# AURELIA Lab 5 Evaluation Report

**Evaluation ID:** {summary.evaluation_id}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Duration:** {summary.duration_minutes:.1f} minutes

## Executive Summary

- **Overall Quality Score:** {summary.overall_quality_score:.1f}/100
- **Concepts Evaluated:** {summary.total_concepts}
- **Success Rate:** {summary.successful_evaluations}/{summary.total_concepts} ({summary.successful_evaluations/summary.total_concepts*100:.1f}%)
- **Average Latency:** {summary.average_latency_ms:.0f}ms
- **Total Cost:** ${summary.total_cost_usd:.4f}

## Quality Metrics

### Overall Quality Distribution
- **High Quality (>80):** {summary.high_quality_count} concepts
- **Medium Quality (60-80):** {summary.medium_quality_count} concepts  
- **Low Quality (<60):** {summary.low_quality_count} concepts

### Quality Breakdown
- **Accuracy:** {summary.average_accuracy:.3f} (semantic similarity)
- **Completeness:** {summary.average_completeness:.3f} (field coverage)
- **Citation Fidelity:** {summary.average_citation_fidelity:.3f} (page reference accuracy)

## Performance Metrics

### Latency Distribution
- **Fast Responses (<1000ms):** {summary.fast_responses_count} queries
- **Medium Responses (1000-3000ms):** {summary.medium_responses_count} queries
- **Slow Responses (>3000ms):** {summary.slow_responses_count} queries

### Performance Breakdown
- **Average Total Time:** {summary.average_latency_ms:.0f}ms
- **Average Retrieval Time:** {summary.average_retrieval_time_ms:.0f}ms
- **Average Generation Time:** {summary.average_generation_time_ms:.0f}ms
- **Cache Hit Rate:** {summary.cache_hit_rate:.1%}
- **Cache Speedup Factor:** {summary.cache_speedup_factor:.1f}x

## Cost Analysis

- **Total Tokens Used:** {summary.total_tokens_used:,}
- **Average Cost per Query:** ${summary.average_cost_per_query_usd:.4f}
- **Total Estimated Cost:** ${summary.total_cost_usd:.4f}

## Recommendations

### Quality Improvements
"""
        
        if summary.average_accuracy < 0.8:
            report += "- **Accuracy:** Improve semantic similarity through better retrieval and generation\n"
        
        if summary.average_completeness < 0.8:
            report += "- **Completeness:** Ensure responses include all required fields (definition, components, examples, use cases)\n"
        
        if summary.average_citation_fidelity < 0.7:
            report += "- **Citations:** Improve page reference accuracy and source attribution\n"
        
        report += """
### Performance Optimizations
"""
        
        if summary.average_latency_ms > 3000:
            report += "- **Latency:** Consider optimizing retrieval and generation pipelines\n"
        
        if summary.cache_hit_rate < 0.5:
            report += "- **Caching:** Improve cache hit rates for better performance\n"
        
        report += """
### Cost Management
"""
        
        if summary.average_cost_per_query_usd > 0.05:
            report += "- **Cost:** Consider optimizing token usage and model selection\n"
        
        report += f"""
## Technical Details

### Evaluation Configuration
- **Accuracy Weight:** {self.config.accuracy_weight}
- **Completeness Weight:** {self.config.completeness_weight}
- **Citation Weight:** {self.config.citation_weight}
- **High Quality Threshold:** {self.config.high_quality_threshold}
- **Fast Response Threshold:** {self.config.fast_response_threshold_ms}ms

### Model Information
- **Embedding Model:** {self.embedding_model or 'text-embedding-3-large'}
- **Generation Model:** gpt-4o-mini
- **Evaluation Method:** Semantic similarity + field coverage + citation fidelity

---
*Report generated by AURELIA Lab 5 Evaluation System v1.0.0*
"""
        
        return report
