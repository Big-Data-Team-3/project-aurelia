"""
RAG System Configuration
Centralized configuration for all RAG components
"""

import os
from typing import Dict, List, Optional
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
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_query_logging: bool = Field(True, env="ENABLE_QUERY_LOGGING")
    
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
