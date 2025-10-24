"""
RAG System Configuration
Centralized configuration for all RAG components
"""

import os
from typing import Dict, List, Optional, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class RAGConfig(BaseSettings):
    """RAG system configuration with environment variable support"""
    
    # API Keys
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    pinecone_api_key: str = Field("", env="PINECONE_API_KEY")
    # Pinecone Configuration
    pinecone_index_name: str = Field("fintbx-embeddings", env="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field("us-east-1", env="PINECONE_ENVIRONMENT")
    
    # OpenAI Configuration
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    generation_model: str = Field("gpt-4o-mini", env="GENERATION_MODEL")
    embedding_dimension: int = Field(3072, env="EMBEDDING_DIMENSION")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_password: str = Field("", env="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, env="REDIS_SSL")
    redis_max_connections: int = Field(10, env="REDIS_MAX_CONNECTIONS")
    cache_prefix: str = Field("aurelia", env="CACHE_PREFIX")
    response_cache_ttl: int = Field(1800, env="RESPONSE_CACHE_TTL")
    embedding_cache_ttl: int = Field(86400, env="EMBEDDING_CACHE_TTL")
    search_cache_ttl: int = Field(3600, env="SEARCH_CACHE_TTL")
    session_cache_ttl: int = Field(86400, env="SESSION_CACHE_TTL")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour
    
    # Search Configuration
    default_top_k: int = Field(10, env="DEFAULT_TOP_K")
    max_top_k: int = Field(50, env="MAX_TOP_K")
    rerank_top_k: int = Field(5, env="RERANK_TOP_K")
    
    # Reranking Configuration
    reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")
    enable_reranking: bool = Field(True, env="ENABLE_RERANKING")
    
    # RRF Configuration
    rrf_k: int = Field(60, env="RRF_K")  # RRF parameter
    vector_weight: float = Field(0.7, env="VECTOR_WEIGHT")
    keyword_weight: float = Field(0.3, env="KEYWORD_WEIGHT")
    
    # Generation Configuration
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    # Wikipedia Fallback
    enable_wikipedia_fallback: bool = Field(True, env="ENABLE_WIKIPEDIA_FALLBACK")
    wikipedia_top_k: int = Field(3, env="WIKIPEDIA_TOP_K")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    
    # Performance Configuration
    request_timeout: int = Field(60, env="REQUEST_TIMEOUT")  # Increased from 30 to 60 seconds
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_query_logging: bool = Field(True, env="ENABLE_QUERY_LOGGING")
    
    # Session Management Configuration
    session_default_ttl_hours: int = Field(24, env="SESSION_DEFAULT_TTL_HOURS")
    session_max_context_messages: int = Field(20, env="SESSION_MAX_CONTEXT_MESSAGES")
    session_cleanup_interval_minutes: int = Field(60, env="SESSION_CLEANUP_INTERVAL_MINUTES")
    
    # Financial Enhancement Configuration
    enable_financial_enhancement: bool = Field(True, env="ENABLE_FINANCIAL_ENHANCEMENT")
    financial_enhancement_delay: float = Field(0.5, env="FINANCIAL_ENHANCEMENT_DELAY")  # seconds
    max_concurrent_enhancements: int = Field(3, env="MAX_CONCURRENT_ENHANCEMENTS")
    
    # Timeout Configuration
    instructor_classification_timeout: int = Field(30, env="INSTRUCTOR_CLASSIFICATION_TIMEOUT")  # seconds
    instructor_enhancement_timeout: int = Field(45, env="INSTRUCTOR_ENHANCEMENT_TIMEOUT")  # seconds
    frontend_request_timeout: int = Field(120, env="FRONTEND_REQUEST_TIMEOUT")  # seconds
    
    # PII Detection Configuration
    enable_pii_detection: bool = Field(True, env="ENABLE_PII_DETECTION")
    pii_strict_mode: bool = Field(False, env="PII_STRICT_MODE")
    pii_confidence_threshold: float = Field(0.7, env="PII_CONFIDENCE_THRESHOLD")
    pii_rejection_threshold: float = Field(0.9, env="PII_REJECTION_THRESHOLD")
    
    # Query Processing Configuration
    enable_query_processing: bool = Field(True, env="ENABLE_QUERY_PROCESSING")
    enable_intent_classification: bool = Field(True, env="ENABLE_INTENT_CLASSIFICATION")
    enable_query_expansion: bool = Field(True, env="ENABLE_QUERY_EXPANSION")
    intent_confidence_threshold: float = Field(0.6, env="INTENT_CONFIDENCE_THRESHOLD")
    
    # Routing Configuration
    default_routing_strategy: str = Field("auto", env="DEFAULT_ROUTING_STRATEGY")  # auto, rag, chatgpt
    rag_keyword_threshold: float = Field(0.1, env="RAG_KEYWORD_THRESHOLD")
    force_rag_for_factual: bool = Field(True, env="FORCE_RAG_FOR_FACTUAL")
    
    class Config:
        # Use absolute path to ensure .env file is found regardless of working directory
        env_file = Path(__file__).parent.parent.parent / ".env"
        case_sensitive = False
        extra = "ignore"


class PromptTemplates:
    """Centralized prompt templates for RAG system"""
    
    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in financial technology and business analysis. 
You have access to relevant context from financial documents and can provide accurate, detailed responses.

Guidelines:
- Use the provided context to answer questions accurately
- If the context doesn't contain enough information, clearly state this
- Provide specific details and examples when available
- Cite relevant sections or pages when possible
- If asked about topics not covered in the context, indicate this clearly
"""

    RAG_USER_PROMPT = """Context information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain sufficient information to answer the question, please state this clearly."""

    QUERY_EXPANSION_PROMPT = """Given the following query, generate 2-3 alternative phrasings or related questions that might help find relevant information:

Original query: {query}

Alternative queries:"""

    CONTEXT_RELEVANCE_PROMPT = """Rate the relevance of the following context to the given question on a scale of 1-10:

Question: {question}
Context: {context}

Relevance score (1-10):"""


class SearchStrategies:
    """Available search strategies and their configurations"""
    
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    RERANKED = "reranked"
    RRF_FUSION = "rrf_fusion"
    
    @classmethod
    def get_all_strategies(cls) -> List[str]:
        return [cls.VECTOR_ONLY, cls.HYBRID, cls.RERANKED, cls.RRF_FUSION]
    
    @classmethod
    def get_strategy_config(cls, strategy: str) -> Dict:
        """Get configuration for a specific search strategy"""
        configs = {
            cls.VECTOR_ONLY: {
                "use_vector": True,
                "use_keyword": False,
                "use_reranking": False,
                "use_rrf": False
            },
            cls.HYBRID: {
                "use_vector": True,
                "use_keyword": True,
                "use_reranking": False,
                "use_rrf": False
            },
            cls.RERANKED: {
                "use_vector": True,
                "use_keyword": False,
                "use_reranking": True,
                "use_rrf": False
            },
            cls.RRF_FUSION: {
                "use_vector": True,
                "use_keyword": True,
                "use_reranking": True,
                "use_rrf": True
            }
        }
        return configs.get(strategy, configs[cls.VECTOR_ONLY])


class QueryProcessingConfig:
    """Configuration for query processing pipeline"""
    
    # PII patterns and their configurations
    PII_PATTERNS = {
        "email": {
            "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "replacement": "[EMAIL]",
            "confidence": 0.95,
            "description": "Email address"
        },
        "phone_us": {
            "pattern": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "replacement": "[PHONE]",
            "confidence": 0.90,
            "description": "US phone number"
        },
        "ssn": {
            "pattern": r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b',
            "replacement": "[SSN]",
            "confidence": 0.95,
            "description": "Social Security Number"
        }
    }
    
    # Intent classification keywords
    INTENT_KEYWORDS = {
        "factual_query": [
            "what", "how", "when", "where", "why", "who", "which", "define", "explain",
            "describe", "calculate", "analyze", "compare", "difference", "between"
        ],
        "conversational": [
            "think", "opinion", "feel", "believe", "prefer", "like", "dislike",
            "recommend", "suggest", "advice", "help me", "assist", "support"
        ],
        "greeting": [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "greetings", "howdy", "what's up", "how are you", "nice to meet"
        ]
    }
    
    # RAG routing keywords
    RAG_KEYWORDS = [
        "document", "report", "paper", "study", "research", "analysis",
        "financial", "finance", "investment", "revenue", "profit", "market",
        "according to", "based on", "mentioned in", "states that", "shows that"
    ]


class SessionConfig:
    """Configuration for session management"""
    
    DEFAULT_TTL_HOURS = 24
    MAX_CONTEXT_MESSAGES = 20
    CLEANUP_INTERVAL_MINUTES = 60
    MAX_SESSIONS_PER_USER = 10
    
    @classmethod
    def get_session_settings(cls, config: 'RAGConfig') -> Dict[str, Any]:
        """Get session settings from RAG config"""
        return {
            "default_ttl_hours": config.session_default_ttl_hours,
            "max_context_messages": config.session_max_context_messages,
            "cleanup_interval_minutes": config.session_cleanup_interval_minutes
        }


# Load environment variables explicitly if needed
def load_env_if_exists():
    """Explicitly load .env file if it exists"""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            return True
        except ImportError:
            print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
            return False
    return False

# Load environment variables
load_env_if_exists()

# Global configuration instance
rag_config = RAGConfig()
