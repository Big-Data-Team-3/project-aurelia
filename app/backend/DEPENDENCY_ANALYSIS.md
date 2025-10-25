# Backend Dependency Analysis

## Overview
This document provides a comprehensive analysis of all dependencies used in the Project Aurelia backend and identifies missing packages in the original `requirements.txt`.

## Missing Dependencies Found

### Critical AI/ML Dependencies
The following packages are used in the codebase but were missing from the original requirements.txt:

1. **`openai>=1.0.0`** - Used in multiple services:
   - `services/vector_search.py` - For embeddings
   - `services/generation.py` - For text generation
   - `services/chatgpt_service.py` - For ChatGPT integration
   - `services/query_rewriter.py` - For query rewriting
   - `services/instructor_classifier.py` - For AI classification

2. **`pinecone-client>=3.0.0`** - Used in:
   - `services/vector_search.py` - For vector database operations

3. **`sentence-transformers>=2.2.0`** - Used in:
   - `services/reranking.py` - For neural reranking with CrossEncoder

4. **`rank-bm25>=0.2.2`** - Used in:
   - `services/hybrid_search.py` - For BM25 keyword search

5. **`instructor>=0.4.0`** - Used in:
   - `services/instructor_classifier.py` - For structured AI responses

6. **`wikipedia>=1.4.0`** - Used in:
   - `services/wikipedia.py` - For Wikipedia fallback search

7. **`numpy>=1.24.0`** - Used in:
   - `services/reranking.py` - For numerical operations

### Configuration Dependencies
8. **`pydantic-settings>=2.0.0`** - Used in:
   - `config/rag_config.py` - For environment-based configuration

## Updated Requirements.txt

The updated `requirements.txt` now includes all necessary dependencies:

```txt
# Core FastAPI dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Database dependencies
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
alembic>=1.12.0

# Authentication & Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Environment & Configuration
python-dotenv>=1.0.0
pydantic-settings>=2.0.0

# HTTP requests
requests>=2.31.0

# Redis for caching
redis>=5.0.0

# AI/ML Dependencies
openai>=1.0.0
pinecone-client>=3.0.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2
instructor>=0.4.0

# Wikipedia integration
wikipedia>=1.4.0

# Data processing
numpy>=1.24.0

# Development dependencies (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

## Deployment Files Created

### 1. Dockerfile
- Optimized for production deployment
- Uses Python 3.11 slim image
- Multi-stage build for smaller image size
- Non-root user for security
- Health checks included

### 2. docker-compose.yml
- Backend service only (uses existing Redis service)
- Environment variable configuration
- Health checks and restart policies
- Connects to external Redis via REDIS_URL

### 3. .dockerignore
- Optimized to exclude unnecessary files
- Reduces build context size
- Faster Docker builds

### 4. deploy.sh
- Automated deployment script
- Multiple deployment options
- Environment setup assistance
- Log viewing and cleanup utilities

## Deployment Commands

### Quick Deployment
```bash
# Make script executable
chmod +x deploy.sh

# Deploy with Docker Compose
./deploy.sh deploy

# Or use Docker Compose directly
docker-compose up -d
```

### Manual Deployment
```bash
# Build image
docker build -t project-aurelia-backend .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="your_database_url" \
  -e OPENAI_API_KEY="your_openai_key" \
  -e PINECONE_API_KEY="your_pinecone_key" \
  project-aurelia-backend
```

## Environment Variables Required

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/postgres

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=fintbx-embeddings

# Redis Configuration (point to your existing Redis service)
REDIS_URL=redis://your-redis-host:6379
REDIS_PASSWORD=your_redis_password_if_any

# Environment
ENVIRONMENT=production
```

## Performance Optimizations

### Docker Build Optimizations
1. **Multi-stage builds** - Separate build and runtime environments
2. **Layer caching** - Copy requirements.txt first for better caching
3. **Minimal base image** - Using Python 3.11 slim
4. **Non-root user** - Security best practice

### Runtime Optimizations
1. **Health checks** - Automatic container health monitoring
2. **Restart policies** - Automatic recovery from failures
3. **Resource limits** - Prevent resource exhaustion
4. **External Redis** - Uses your existing Redis service

## Security Considerations

1. **Non-root user** - Container runs as non-privileged user
2. **Environment variables** - Sensitive data not in images
3. **Minimal attack surface** - Slim base image with minimal packages
4. **Health checks** - Early detection of issues

## Monitoring and Logging

1. **Health checks** - Built-in health monitoring
2. **Structured logging** - Consistent log format
3. **Log aggregation** - Docker Compose log management
4. **Metrics collection** - Ready for monitoring tools

## Next Steps

1. **Update requirements.txt** - Use the updated version
2. **Set up environment variables** - Configure API keys and database
3. **Test deployment** - Use the deploy script
4. **Monitor performance** - Check logs and health status
5. **Scale as needed** - Adjust resources based on usage

## Troubleshooting

### Common Issues
1. **Missing API keys** - Ensure all environment variables are set
2. **Database connection** - Verify DATABASE_URL is correct
3. **Port conflicts** - Check if port 8000 is available
4. **Memory issues** - Increase Docker memory limits if needed

### Debug Commands
```bash
# View logs
./deploy.sh logs

# Check container status
docker-compose ps

# Access container shell
docker-compose exec backend bash

# View resource usage
docker stats
```
