"""
AURELIA Lab 5 Evaluation Models
Pydantic models for evaluation data structures and metrics
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class EvaluationStatus(str, Enum):
    """Status of evaluation results"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class ConceptQualityMetrics(BaseModel):
    """Quality metrics for a single concept evaluation"""
    
    # Accuracy metrics (40% weight)
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    accuracy_explanation: str = Field("", description="Explanation of accuracy assessment")
    
    # Completeness metrics (30% weight)
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Field coverage completeness")
    has_definition: bool = Field(False, description="Has definition field")
    has_key_components: bool = Field(False, description="Has key components field")
    has_example: bool = Field(False, description="Has example field")
    has_use_cases: bool = Field(False, description="Has use cases field")
    has_formula: bool = Field(False, description="Has formula field")
    completeness_explanation: str = Field("", description="Explanation of completeness assessment")
    
    # Citation fidelity metrics (30% weight)
    citation_fidelity_score: float = Field(..., ge=0.0, le=1.0, description="Citation accuracy score")
    expected_pages: List[int] = Field(default_factory=list, description="Expected page references")
    actual_pages: List[int] = Field(default_factory=list, description="Actual page references")
    citation_precision: float = Field(0.0, description="Citation precision")
    citation_recall: float = Field(0.0, description="Citation recall")
    citation_f1: float = Field(0.0, description="Citation F1 score")
    citation_explanation: str = Field("", description="Explanation of citation assessment")
    
    # Overall quality score
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")


class LatencyMetrics(BaseModel):
    """Performance metrics for query processing"""
    
    total_time_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
    retrieval_time_ms: float = Field(0.0, ge=0.0, description="Document retrieval time")
    generation_time_ms: float = Field(0.0, ge=0.0, description="Response generation time")
    cache_hit: bool = Field(False, description="Whether response was served from cache")
    
    # Component breakdown
    vector_search_time_ms: float = Field(0.0, ge=0.0, description="Vector search time")
    reranking_time_ms: float = Field(0.0, ge=0.0, description="Reranking time")
    wikipedia_fallback_time_ms: float = Field(0.0, ge=0.0, description="Wikipedia fallback time")
    
    # Performance indicators
    speedup_factor: float = Field(1.0, ge=0.0, description="Speedup factor vs baseline")


class TokenCostMetrics(BaseModel):
    """Token usage and cost metrics"""
    
    total_tokens: int = Field(0, ge=0, description="Total tokens used")
    input_tokens: int = Field(0, ge=0, description="Input tokens")
    output_tokens: int = Field(0, ge=0, description="Output tokens")
    
    # Cost breakdown
    embedding_cost_usd: float = Field(0.0, ge=0.0, description="Embedding API cost")
    generation_cost_usd: float = Field(0.0, ge=0.0, description="Generation API cost")
    total_cost_usd: float = Field(0.0, ge=0.0, description="Total estimated cost")
    
    # Model information
    embedding_model: str = Field("text-embedding-3-large", description="Embedding model used")
    generation_model: str = Field("gpt-4o-mini", description="Generation model used")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single concept"""
    
    concept_name: str = Field(..., description="Name of the concept evaluated")
    query: str = Field(..., description="Query used for evaluation")
    generated_response: str = Field(..., description="Generated response from RAG system")
    ground_truth: Dict[str, Any] = Field(..., description="Ground truth data for the concept")
    
    # Evaluation metrics
    quality_metrics: ConceptQualityMetrics = Field(..., description="Quality evaluation metrics")
    latency_metrics: LatencyMetrics = Field(..., description="Performance metrics")
    token_cost_metrics: TokenCostMetrics = Field(..., description="Cost metrics")
    
    # Additional data
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")
    
    # Timestamps
    evaluated_at: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    status: EvaluationStatus = Field(EvaluationStatus.SUCCESS, description="Evaluation status")


class EvaluationSummary(BaseModel):
    """Summary of complete evaluation run"""
    
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    total_concepts: int = Field(..., ge=0, description="Total number of concepts evaluated")
    successful_evaluations: int = Field(..., ge=0, description="Number of successful evaluations")
    failed_evaluations: int = Field(..., ge=0, description="Number of failed evaluations")
    
    # Overall quality metrics
    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
    average_accuracy: float = Field(..., ge=0.0, le=1.0, description="Average accuracy score")
    average_completeness: float = Field(..., ge=0.0, le=1.0, description="Average completeness score")
    average_citation_fidelity: float = Field(..., ge=0.0, le=1.0, description="Average citation fidelity")
    
    # Performance metrics
    average_latency_ms: float = Field(..., ge=0.0, description="Average total latency")
    average_retrieval_time_ms: float = Field(..., ge=0.0, description="Average retrieval time")
    average_generation_time_ms: float = Field(..., ge=0.0, description="Average generation time")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    cache_speedup_factor: float = Field(..., ge=0.0, description="Average cache speedup")
    
    # Cost metrics
    total_cost_usd: float = Field(..., ge=0.0, description="Total estimated cost")
    average_cost_per_query_usd: float = Field(..., ge=0.0, description="Average cost per query")
    total_tokens_used: int = Field(..., ge=0, description="Total tokens used")
    
    # Quality distribution
    high_quality_count: int = Field(0, ge=0, description="Number of high-quality responses (>80)")
    medium_quality_count: int = Field(0, ge=0, description="Number of medium-quality responses (60-80)")
    low_quality_count: int = Field(0, ge=0, description="Number of low-quality responses (<60)")
    
    # Performance distribution
    fast_responses_count: int = Field(0, ge=0, description="Number of fast responses (<1000ms)")
    medium_responses_count: int = Field(0, ge=0, description="Number of medium responses (1000-3000ms)")
    slow_responses_count: int = Field(0, ge=0, description="Number of slow responses (>3000ms)")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now, description="Evaluation start time")
    completed_at: datetime = Field(default_factory=datetime.now, description="Evaluation completion time")
    duration_minutes: float = Field(..., ge=0.0, description="Total evaluation duration in minutes")
    
    # Configuration
    evaluation_config: Dict[str, Any] = Field(default_factory=dict, description="Evaluation configuration")


class GroundTruthConcept(BaseModel):
    """Ground truth data structure for a financial concept"""
    
    name: str = Field(..., description="Concept name")
    definition: str = Field(..., description="Concept definition")
    key_components: List[str] = Field(default_factory=list, description="Key components")
    formulas: List[str] = Field(default_factory=list, description="Relevant formulas")
    examples: List[str] = Field(default_factory=list, description="Examples")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    page_references: List[int] = Field(default_factory=list, description="Page references")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in ground truth")
    source_document: str = Field("fintbx.pdf", description="Source document")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs"""
    
    # Quality evaluation weights
    accuracy_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for accuracy metric")
    completeness_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for completeness metric")
    citation_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for citation metric")
    
    # Quality thresholds
    high_quality_threshold: float = Field(80.0, ge=0.0, le=100.0, description="High quality threshold")
    medium_quality_threshold: float = Field(60.0, ge=0.0, le=100.0, description="Medium quality threshold")
    
    # Performance thresholds
    fast_response_threshold_ms: float = Field(1000.0, ge=0.0, description="Fast response threshold")
    slow_response_threshold_ms: float = Field(3000.0, ge=0.0, description="Slow response threshold")
    
    # Evaluation settings
    enable_caching: bool = Field(True, description="Enable response caching")
    force_refresh: bool = Field(False, description="Force fresh generation")
    max_retries: int = Field(3, ge=1, description="Maximum retry attempts")
    timeout_seconds: int = Field(60, ge=1, description="Request timeout")
    
    # Report settings
    generate_markdown_report: bool = Field(True, description="Generate markdown report")
    upload_to_gcs: bool = Field(True, description="Upload results to GCS")
    gcs_bucket: str = Field("aurelia-evaluations", description="GCS bucket name")
    
    # Cost estimation
    embedding_cost_per_1k_tokens: float = Field(0.13, ge=0.0, description="Embedding cost per 1K tokens")
    generation_input_cost_per_1k_tokens: float = Field(0.15, ge=0.0, description="Generation input cost per 1K tokens")
    generation_output_cost_per_1k_tokens: float = Field(0.60, ge=0.0, description="Generation output cost per 1K tokens")
