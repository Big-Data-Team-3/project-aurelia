# Project Aurelia RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with FastAPI and Streamlit, featuring advanced search algorithms, neural reranking, and Wikipedia fallback.

## Features

### 🔍 Advanced Search Strategies
- **Vector Search**: Pure semantic similarity using embeddings
- **Hybrid Search**: Combines vector search with BM25 keyword matching
- **Neural Reranking**: Cross-encoder models for result refinement
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result combination

### 🧠 AI-Powered Generation
- OpenAI GPT models for response generation
- Context-aware responses with source citations
- Streaming responses for real-time chat
- Conversation history support

### 📚 Knowledge Sources
- Pinecone vector database with existing financial documents
- Wikipedia fallback for comprehensive coverage
- Multiple chunking strategies (hierarchical, hybrid, type-aware)

### 🚀 Modern Architecture
- FastAPI backend with async support
- Streamlit chat interface
- Modular service architecture
- Comprehensive health monitoring

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp app/.env.example app/.env

# Edit .env with your API keys
# - OPENAI_API_KEY: Your OpenAI API key
# - PINECONE_API_KEY: Your Pinecone API key
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Backend

```bash
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Streamlit App

```bash
cd app/streamlit
streamlit run app.py --server.port 8501
```

### 5. Access the Applications

- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit App**: http://localhost:8501
- **RAG Chat**: Navigate to "RAG Chat" in the Streamlit sidebar

## API Endpoints

### Core RAG Endpoints

- `POST /rag/query` - Main RAG endpoint with generation
- `POST /rag/search` - Search-only without generation
- `POST /rag/rerank` - Standalone reranking
- `POST /rag/query/stream` - Streaming responses
- `POST /rag/conversation` - Conversational queries

### System Endpoints

- `GET /rag/health` - System health check
- `GET /rag/config` - Current configuration
- `GET /rag/strategies` - Available search strategies

### Batch Operations

- `POST /rag/batch/query` - Batch processing of multiple queries

## Search Strategies

### 1. Vector Only (`vector_only`)
Pure semantic search using embeddings. Best for conceptual queries.

### 2. Hybrid (`hybrid`)
Combines vector search with BM25 keyword matching. Good balance of semantic and lexical matching.

### 3. Reranked (`reranked`)
Vector search results refined using cross-encoder neural reranking. Higher precision for complex queries.

### 4. RRF Fusion (`rrf_fusion`) - **Recommended**
Full pipeline with hybrid search, RRF fusion, and neural reranking. Best overall performance.

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Models
EMBEDDING_MODEL=text-embedding-3-large
GENERATION_MODEL=gpt-4o-mini
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Search Parameters
DEFAULT_TOP_K=10
RERANK_TOP_K=5
VECTOR_WEIGHT=0.7
KEYWORD_WEIGHT=0.3

# Features
ENABLE_RERANKING=true
ENABLE_WIKIPEDIA_FALLBACK=true
```

### Streamlit Configuration

The chat interface allows real-time configuration:
- Search strategy selection
- Number of sources to retrieve
- Generation parameters (temperature, max tokens)
- Wikipedia fallback toggle
- Source citation display

## Architecture

### Service Layer (`app/backend/services/`)

- **`rag_service.py`** - Main orchestration service
- **`vector_search.py`** - Pinecone integration
- **`generation.py`** - OpenAI text generation
- **`reranking.py`** - Cross-encoder reranking
- **`hybrid_search.py`** - Hybrid search with BM25
- **`fusion.py`** - RRF and other fusion algorithms
- **`wikipedia.py`** - Wikipedia fallback service

### API Layer (`app/backend/routers/`)

- **`rag.py`** - FastAPI endpoints for RAG functionality

### Models (`app/backend/models/`)

- **`rag_models.py`** - Pydantic models for requests/responses

### Configuration (`app/backend/config/`)

- **`rag_config.py`** - Centralized configuration management

## Usage Examples

### Python API Client

```python
import requests

# Basic RAG query
response = requests.post("http://localhost:8000/rag/query", json={
    "query": "What are the key financial metrics discussed?",
    "strategy": "rrf_fusion",
    "top_k": 10,
    "include_sources": True
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

### Search Only

```python
# Search without generation
response = requests.post("http://localhost:8000/rag/search", json={
    "query": "financial risk assessment",
    "strategy": "hybrid",
    "top_k": 5
})

results = response.json()
for result in results['results']:
    print(f"Score: {result['score']:.3f} - {result['text'][:100]}...")
```

### Health Check

```python
# Check system health
response = requests.get("http://localhost:8000/rag/health")
health = response.json()
print(f"System Status: {health['status']}")
print(f"Pinecone: {health['pinecone_connected']}")
print(f"OpenAI: {health['openai_connected']}")
```

## Performance Optimization

### Search Performance
- ANN (Approximate Nearest Neighbor) optimizations in Pinecone
- Configurable batch sizes for embedding creation
- Concurrent processing for multi-query operations

### Generation Performance
- Streaming responses for real-time chat
- Configurable token limits and temperature
- Request timeout management

### Caching
- Optional response caching (configurable TTL)
- BM25 index caching for repeated queries

## Monitoring and Logging

### Health Monitoring
- Comprehensive health checks for all services
- Component-level status reporting
- Performance metrics tracking

### Logging
- Configurable log levels
- Query logging for analysis
- Error tracking and reporting

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file has correct API keys
   - Check API key permissions and quotas

2. **Pinecone Connection Issues**
   - Verify index name and environment
   - Check Pinecone API key and permissions

3. **Model Loading Errors**
   - Ensure sufficient memory for reranking models
   - Check internet connection for model downloads

4. **Performance Issues**
   - Adjust batch sizes in configuration
   - Enable caching for repeated queries
   - Consider reducing `top_k` values

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

### Health Checks

Use the health endpoint to diagnose issues:
```bash
curl http://localhost:8000/rag/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
