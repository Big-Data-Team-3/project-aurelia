"""
Pydantic models for RAG system requests and responses
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class SearchStrategy(str, Enum):
    """Available search strategies"""
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    RERANKED = "reranked"
    RRF_FUSION = "rrf_fusion"


class SourceType(str, Enum):
    """Types of sources for retrieved content"""
    DOCUMENT = "document"
    WIKIPEDIA = "wikipedia"


class SearchResult(BaseModel):
    """Individual search result from vector database or Wikipedia"""
    id: str = Field(..., description="Unique identifier for the result")
    text: str = Field(..., description="Text content of the result")
    score: float = Field(..., description="Relevance score (0-1)")
    source_type: SourceType = Field(..., description="Type of source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Document-specific fields
    section: Optional[str] = Field(None, description="Document section")
    pages: Optional[List[int]] = Field(None, description="Page numbers")
    strategy: Optional[str] = Field(None, description="Chunking strategy used")
    
    # Wikipedia-specific fields
    title: Optional[str] = Field(None, description="Wikipedia article title")
    url: Optional[str] = Field(None, description="Source URL")


class RAGQuery(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    strategy: SearchStrategy = Field(SearchStrategy.RRF_FUSION, description="Search strategy to use")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    rerank_top_k: int = Field(5, ge=1, le=20, description="Number of results to rerank")
    include_sources: bool = Field(True, description="Include source information in response")
    enable_wikipedia_fallback: bool = Field(True, description="Enable Wikipedia fallback if results are insufficient")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Generation temperature")
    max_tokens: int = Field(1000, ge=100, le=2000, description="Maximum tokens in response")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class SearchOnlyQuery(BaseModel):
    """Request model for search-only queries (no generation)"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    strategy: SearchStrategy = Field(SearchStrategy.RRF_FUSION, description="Search strategy to use")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    rerank_top_k: int = Field(5, ge=1, le=20, description="Number of results to rerank")
    enable_wikipedia_fallback: bool = Field(True, description="Enable Wikipedia fallback")


class RerankQuery(BaseModel):
    """Request model for reranking queries"""
    query: str = Field(..., min_length=1, max_length=1000, description="Query for reranking")
    results: List[SearchResult] = Field(..., min_items=1, description="Results to rerank")
    top_k: int = Field(5, ge=1, le=20, description="Number of top results to return")


class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SearchResult] = Field(default_factory=list, description="Source documents used")
    strategy_used: SearchStrategy = Field(..., description="Search strategy that was used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Performance metrics
    retrieval_time_ms: Optional[float] = Field(None, description="Time taken for retrieval in milliseconds")
    generation_time_ms: Optional[float] = Field(None, description="Time taken for generation in milliseconds")
    total_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")


class SearchResponse(BaseModel):
    """Response model for search-only queries"""
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    strategy_used: SearchStrategy = Field(..., description="Search strategy that was used")
    total_results: int = Field(..., description="Total number of results found")
    retrieval_time_ms: Optional[float] = Field(None, description="Time taken for retrieval in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class RerankResponse(BaseModel):
    """Response model for reranking queries"""
    query: str = Field(..., description="Original query")
    reranked_results: List[SearchResult] = Field(default_factory=list, description="Reranked results")
    original_count: int = Field(..., description="Number of original results")
    reranked_count: int = Field(..., description="Number of reranked results")
    rerank_time_ms: Optional[float] = Field(None, description="Time taken for reranking in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class RAGHealthCheck(BaseModel):
    """Health check response for RAG system"""
    status: str = Field(..., description="System status")
    pinecone_connected: bool = Field(..., description="Pinecone connection status")
    openai_connected: bool = Field(..., description="OpenAI connection status")
    reranker_loaded: bool = Field(..., description="Reranker model status")
    wikipedia_available: bool = Field(..., description="Wikipedia service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class RAGConfig(BaseModel):
    """Configuration response model"""
    available_strategies: List[SearchStrategy] = Field(..., description="Available search strategies")
    default_strategy: SearchStrategy = Field(..., description="Default search strategy")
    max_top_k: int = Field(..., description="Maximum allowed top_k value")
    reranker_model: str = Field(..., description="Reranker model being used")
    embedding_model: str = Field(..., description="Embedding model being used")
    generation_model: str = Field(..., description="Generation model being used")
    features: Dict[str, bool] = Field(..., description="Enabled features")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Streaming response models for real-time chat
class StreamingChunk(BaseModel):
    """Individual chunk in a streaming response"""
    type: str = Field(..., description="Chunk type: 'text', 'sources', 'metadata'")
    content: Union[str, List[SearchResult], Dict[str, Any]] = Field(..., description="Chunk content")
    is_final: bool = Field(False, description="Whether this is the final chunk")


class ChatMessage(BaseModel):
    """Chat message model for conversation history"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    sources: Optional[List[SearchResult]] = Field(None, description="Sources used (for assistant messages)")


class ChatSession(BaseModel):
    """Chat session model"""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation history")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
