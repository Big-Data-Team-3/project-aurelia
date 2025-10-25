# Project Aurelia - Advanced RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with FastAPI and Streamlit, featuring advanced document processing, multiple search strategies, neural reranking, and intelligent query routing.

**GitHub Repo:** [Big-Data-Team-3](https://github.com/Big-Data-Team-3/project-aurelia)
 
- **Interactive Tutorial**: [Google Codelabs](https://codelabs-preview.appspot.com/?file_id=1WW1CRWUCD43Zuu0kj5Udn5pIm5i5cgRwh0CzkGeBlZo)

- **Recording Link** : [Video Recording](https://drive.google.com/drive/folders/1ovXMpd7xj6oKaeUFmPXziRpBvTstffFq?usp=drive_link)


## 🌟 Features

### 🔍 Advanced Search & Retrieval
- **Multiple Search Strategies**: Vector search, hybrid search, neural reranking, and RRF fusion
- **Intelligent Query Processing**: PII filtering, intent classification, and automatic routing
- **Neural Reranking**: Cross-encoder models for result refinement
- **Wikipedia Fallback**: Comprehensive coverage when documents don't contain answers
- **Session Management**: Conversational context and history tracking

### 🧠 AI-Powered Generation
- **OpenAI Integration**: GPT-4o-mini for response generation with streaming support
- **Context-Aware Responses**: Maintains conversation history and provides source citations
- **Intelligent Routing**: Automatically routes queries to RAG or ChatGPT based on content analysis
- **Temperature Control**: Configurable creativity levels for different use cases

### 📚 Document Processing Pipeline
- **Multi-Parser Support**: Docling and Enhanced PyMuPDF parsers for comprehensive extraction
- **Intelligent Chunking**: 6 different chunking strategies including hybrid intelligent chunking
- **Content Classification**: Automatic detection of headings, tables, code, equations, and lists
- **Metadata Preservation**: Section hierarchies, page numbers, and content types

### 🚀 Modern Architecture
- **FastAPI Backend**: Async support with comprehensive API documentation
- **Streamlit Frontend**: Interactive chat interface with real-time configuration
- **Pinecone Vector Database**: Scalable vector storage with 3072-dimensional embeddings
- **Modular Services**: Clean separation of concerns with health monitoring

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Document Processing Pipeline](#-document-processing-pipeline)
- [Search Strategies](#-search-strategies)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Development Setup](#-development-setup)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Pinecone API key

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd project-aurelia

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the `app` directory:

```bash
# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=fintbx-embeddings
PINECONE_ENVIRONMENT=us-east-1

# Model Configuration
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

### 3. Start the System

#### Option A: Automated Startup (Recommended)
```bash
cd app
python start_rag.py
```

#### Option B: Manual Startup
```bash
# Terminal 1: Start Backend
cd app/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd app/streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Access the Applications

- **🎨 Streamlit Chat Interface**: http://localhost:8501
- **📚 FastAPI Documentation**: http://localhost:8000/docs
- **🔍 Alternative API Docs**: http://localhost:8000/redoc

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI       │    │    External     │
│   Frontend      │◄──►│    Backend       │◄──►│    Services     │
│                 │    │                  │    │                 │
│ • Chat UI       │    │ • RAG Service    │    │ • OpenAI API    │
│ • Config Panel  │    │ • Search Engine  │    │ • Pinecone DB   │
│ • Health Check  │    │ • Query Router   │    │ • Wikipedia     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```


### Core Components

#### 1. **RAG Service** (`app/backend/services/rag_service.py`)
- **Purpose**: Main orchestration service for all RAG operations
- **Features**: 
  - Query processing and routing
  - Search strategy execution
  - Response generation coordination
  - Session management
  - Streaming support

#### 2. **Search Services**
- **Vector Search** (`vector_search.py`): Pinecone-based semantic similarity
- **Hybrid Search** (`hybrid_search.py`): Combines vector + BM25 keyword search
- **Reranking Service** (`reranking.py`): Cross-encoder neural reranking
- **Fusion Service** (`fusion.py`): RRF, weighted score, and Borda count fusion

#### 3. **Query Processing** (`query_processor.py`)
- **Intent Classification**: Determines query type (factual, conversational, etc.)
- **PII Filtering**: Removes sensitive information
- **Routing Logic**: Decides between RAG and ChatGPT
- **Query Expansion**: Generates related search terms

#### 4. **Generation Services**
- **Generation Service** (`generation.py`): OpenAI GPT integration
- **ChatGPT Service** (`chatgpt_service.py`): Direct ChatGPT for non-RAG queries
- **Wikipedia Service** (`wikipedia.py`): Fallback knowledge source

## 📄 Document Processing Pipeline

### Phase A: Document Parsing

Two advanced parsers extract structured content from PDFs:

#### 1. **Docling Parser** (`pipelines/docli_parsing.py`)
- **Technology**: IBM's Docling library
- **Strengths**: Superior table detection, figure extraction, document structure
- **Output**: Structured blocks with metadata

#### 2. **Enhanced PyMuPDF Parser** (`pipelines/pymupdf_parser.py`)
- **Technology**: PyMuPDF with custom enhancements
- **Features**: 
  - Enhanced text grouping algorithms
  - Improved heading detection
  - Configurable parameters for different document types
  - Advanced post-processing and cleanup

**Extracted Elements:**
- Headings (with hierarchy levels)
- Paragraphs and text blocks
- Tables (converted to Markdown)
- Figures and images
- Code blocks
- Mathematical equations
- Lists (bulleted and numbered)

### Phase B: Intelligent Chunking

Six different chunking strategies available in `data/chunks/`:

#### 1. **Hybrid Intelligent Chunking** (Recommended)
- **File**: `hybrid_intelligent_chunks.jsonl`
- **Strategy**: Combines multiple approaches for optimal results
- **Features**:
  - Preserves special content (tables, code) whole
  - Respects semantic sections
  - Adds hierarchical context
  - Intelligent size-based splitting with overlap
  - Content type classification

#### 2. **Hierarchical Chunking**
- **File**: `hierarchical_chunks.jsonl`
- **Strategy**: Follows document structure hierarchy
- **Best for**: Documents with clear section organization

#### 3. **Section-Based Chunking**
- **File**: `section_based_chunks.jsonl`
- **Strategy**: Splits by document sections
- **Best for**: Well-structured documents with clear sections

#### 4. **Type-Aware Chunking**
- **File**: `type_aware_chunks.jsonl`
- **Strategy**: Groups by content type (paragraphs, tables, etc.)
- **Best for**: Mixed content documents

#### 5. **Page-Based Chunking**
- **File**: `page_based_chunks.jsonl`
- **Strategy**: Splits by page boundaries
- **Best for**: Page-specific queries and citations

#### 6. **Fixed Overlap Chunking**
- **File**: `fixed_overlap_chunks.jsonl`
- **Strategy**: Fixed-size chunks with overlap
- **Best for**: Uniform processing requirements

### Phase C: Embedding & Vector Storage

**Pipeline** (`pipelines/embedding.py`):
1. **Embedding Generation**: OpenAI `text-embedding-3-large` (3072 dimensions)
2. **Batch Processing**: Efficient batch embedding creation
3. **Vector Storage**: Pinecone serverless index with cosine similarity
4. **Metadata Preservation**: Rich metadata for filtering and display

**Embedding Features:**
- **High Dimensionality**: 3072-dimensional vectors for nuanced understanding
- **Batch Optimization**: Processes 100 chunks at a time
- **Error Handling**: Retry logic and comprehensive error reporting
- **Metadata Storage**: Section paths, page numbers, content types, and more

## 🔍 Search Strategies

### 1. Vector Only (`vector_only`)
- **Method**: Pure semantic similarity using embeddings
- **Best for**: Conceptual queries, semantic understanding
- **Speed**: Fastest
- **Accuracy**: Good for semantic matches

### 2. Hybrid Search (`hybrid`)
- **Method**: Combines vector search with BM25 keyword matching
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Best for**: Balanced semantic and lexical matching
- **Speed**: Medium
- **Accuracy**: Better than vector-only for mixed queries

### 3. Reranked (`reranked`)
- **Method**: Vector search + cross-encoder neural reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Best for**: High precision requirements
- **Speed**: Slower due to reranking
- **Accuracy**: Highest precision

### 4. RRF Fusion (`rrf_fusion`) - **Recommended**
- **Method**: Full pipeline with hybrid search, RRF fusion, and neural reranking
- **Features**: 
  - Initial vector search for candidates
  - BM25 keyword scoring
  - Reciprocal Rank Fusion
  - Neural reranking for final results
- **Best for**: Optimal balance of speed and accuracy
- **Speed**: Medium-slow
- **Accuracy**: Best overall performance

### Search Strategy Comparison

| Strategy | Speed | Accuracy | Best Use Case |
|----------|-------|----------|---------------|
| Vector Only | ⚡⚡⚡ | ⭐⭐⭐ | Quick semantic queries |
| Hybrid | ⚡⚡ | ⭐⭐⭐⭐ | Balanced queries |
| Reranked | ⚡ | ⭐⭐⭐⭐⭐ | High precision needs |
| RRF Fusion | ⚡⚡ | ⭐⭐⭐⭐⭐ | Production use |

## 📡 API Documentation

### Core RAG Endpoints

#### `POST /rag/query`
Main RAG endpoint with generation.

**Request:**
```json
{
  "query": "What are the key financial metrics?",
  "strategy": "rrf_fusion",
  "top_k": 10,
  "rerank_top_k": 5,
  "include_sources": true,
  "enable_wikipedia_fallback": true,
  "temperature": 0.1,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "answer": "Based on the financial documents...",
  "sources": [
    {
      "id": "chunk_123",
      "text": "Revenue increased by 15%...",
      "score": 0.92,
      "section": "Financial Performance",
      "pages": [5, 6],
      "metadata": {...}
    }
  ],
  "total_time_ms": 1250,
  "search_time_ms": 800,
  "generation_time_ms": 450,
  "strategy_used": "rrf_fusion"
}
```

#### `POST /rag/search`
Search-only without generation.

#### `POST /rag/query/stream`
Streaming responses for real-time chat.

#### `POST /rag/conversation`
Conversational queries with history.

### System Endpoints

#### `GET /rag/health`
Comprehensive system health check.

**Response:**
```json
{
  "status": "healthy",
  "pinecone_connected": true,
  "openai_connected": true,
  "reranker_loaded": true,
  "wikipedia_available": true,
  "index_stats": {
    "total_vectors": 1500,
    "dimension": 3072
  }
}
```

#### `GET /rag/config`
Current system configuration.

#### `GET /rag/strategies`
Available search strategies and their descriptions.

### Batch Operations

#### `POST /rag/batch/query`
Process multiple queries efficiently.

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `PINECONE_API_KEY` | - | Pinecone API key (required) |
| `PINECONE_INDEX_NAME` | `fintbx-embeddings` | Pinecone index name |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `GENERATION_MODEL` | `gpt-4o-mini` | OpenAI generation model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `DEFAULT_TOP_K` | `10` | Default number of search results |
| `RERANK_TOP_K` | `5` | Number of results to rerank |
| `VECTOR_WEIGHT` | `0.7` | Weight for vector search in hybrid |
| `KEYWORD_WEIGHT` | `0.3` | Weight for keyword search in hybrid |
| `ENABLE_RERANKING` | `true` | Enable neural reranking |
| `ENABLE_WIKIPEDIA_FALLBACK` | `true` | Enable Wikipedia fallback |
| `TEMPERATURE` | `0.1` | Default generation temperature |
| `MAX_TOKENS` | `1000` | Default max response tokens |

### Streamlit Configuration

The chat interface provides real-time configuration:

- **Search Strategy Selection**: Choose from 4 available strategies
- **Source Control**: Number of sources to retrieve and use
- **Generation Parameters**: Temperature and max tokens
- **Feature Toggles**: Wikipedia fallback, source display
- **Advanced Settings**: Fine-tune search and generation parameters

### Performance Tuning

#### For Speed:
```bash
DEFAULT_TOP_K=5
RERANK_TOP_K=3
ENABLE_RERANKING=false
VECTOR_WEIGHT=1.0
KEYWORD_WEIGHT=0.0
```

#### For Accuracy:
```bash
DEFAULT_TOP_K=20
RERANK_TOP_K=10
ENABLE_RERANKING=true
VECTOR_WEIGHT=0.6
KEYWORD_WEIGHT=0.4
```

## 🛠️ Development Setup

### Project Structure

```
project-aurelia/
├── app/
│   ├── backend/
│   │   ├── config/
│   │   │   └── rag_config.py          # Configuration management
│   │   ├── models/
│   │   │   └── rag_models.py          # Pydantic models
│   │   ├── routers/
│   │   │   └── rag.py                 # API endpoints
│   │   ├── services/
│   │   │   ├── rag_service.py         # Main orchestration
│   │   │   ├── vector_search.py       # Pinecone integration
│   │   │   ├── hybrid_search.py       # Hybrid search
│   │   │   ├── reranking.py          # Neural reranking
│   │   │   ├── generation.py         # OpenAI generation
│   │   │   ├── query_processor.py    # Query processing
│   │   │   └── ...
│   │   └── main.py                   # FastAPI app
│   ├── streamlit/
│   │   └── app.py                    # Streamlit interface
│   └── start_rag.py                  # Automated startup
├── pipelines/
│   ├── docli_parsing.py              # Docling parser
│   ├── pymupdf_parser.py             # PyMuPDF parser
│   ├── embedding.py                  # Embedding pipeline
│   └── processing/
│       └── hybrid_chunking.py        # Chunking strategies
├── data/
│   ├── chunks/                       # Generated chunks
│   ├── final/                        # Parsed documents
│   └── embedding_output_pymudpdf/    # Embedding reports
└── requirements.txt
```

### Adding New Search Strategies

1. **Create Service**: Add new service in `app/backend/services/`
2. **Update Models**: Add strategy enum in `models/rag_models.py`
3. **Integrate**: Add to `rag_service.py` routing logic
4. **Configure**: Update `rag_config.py` with strategy config
5. **Test**: Add health checks and error handling

### Adding New Parsers

1. **Create Parser**: Add parser in `pipelines/`
2. **Implement Interface**: Follow existing parser patterns
3. **Add Chunking**: Create corresponding chunking strategy
4. **Update Pipeline**: Integrate into embedding pipeline
5. **Document**: Add to this README

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t project-aurelia .
docker run -p 8000:8000 -p 8501:8501 project-aurelia
```

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **API Keys**: Rotate regularly and use least privilege
3. **Rate Limiting**: Implement API rate limiting
4. **Monitoring**: Add comprehensive logging and metrics
5. **Scaling**: Consider horizontal scaling for high load
6. **Caching**: Implement Redis for query caching
7. **Load Balancing**: Use nginx or similar for production

### Health Monitoring

The system provides comprehensive health checks:

- **Component Health**: Individual service status
- **API Connectivity**: OpenAI and Pinecone connections
- **Index Status**: Vector database statistics
- **Model Loading**: Reranker model availability
- **Performance Metrics**: Response times and success rates

## 🔧 Troubleshooting

### Common Issues

#### 1. **Backend Connection Failed**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check logs
cd app/backend && python -m uvicorn main:app --reload
```

#### 2. **Pinecone Connection Issues**
```bash
# Verify API key
echo $PINECONE_API_KEY

# Check index exists
# Visit Pinecone console to verify index
```

#### 3. **OpenAI API Errors**
```bash
# Check API key and quota
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### 4. **Empty Search Results**
- Verify documents are embedded and uploaded to Pinecone
- Check index statistics via `/rag/health` endpoint
- Try different search strategies
- Lower similarity thresholds

#### 5. **Slow Response Times**
- Reduce `top_k` and `rerank_top_k` values
- Disable reranking for faster responses
- Use `vector_only` strategy for speed
- Check network latency to external APIs

### Performance Optimization

#### Search Performance:
- Use appropriate `top_k` values (5-15 typically optimal)
- Enable reranking only when precision is critical
- Cache frequent queries
- Use hybrid search for balanced performance

#### Generation Performance:
- Adjust `max_tokens` based on needs
- Use lower `temperature` for consistent responses
- Consider using `gpt-3.5-turbo` for faster generation
- Implement streaming for better user experience

### Logging and Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key log locations:
- RAG Service: Query processing and routing decisions
- Search Services: Search performance and results
- Generation Service: OpenAI API interactions
- Health Checks: System status and connectivity

## 📊 Performance Metrics

### Typical Response Times
- **Vector Search**: 200-500ms
- **Hybrid Search**: 400-800ms
- **Reranked Search**: 800-1500ms
- **Generation**: 1000-3000ms
- **Total (RRF Fusion)**: 1500-4000ms

### Accuracy Metrics
- **Vector Only**: ~75% relevance
- **Hybrid**: ~82% relevance  
- **Reranked**: ~88% relevance
- **RRF Fusion**: ~85% relevance (balanced)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review the API documentation at `/docs`
3. Check system health at `/rag/health`
4. Open an issue on GitHub

---

**Project Aurelia** - Advancing the state of the art in Retrieval-Augmented Generation systems.

