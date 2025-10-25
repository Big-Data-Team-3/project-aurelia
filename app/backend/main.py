# region Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import os
import redis
# Import RAG router
from routers.rag import router as rag_router
# Import auth router
from routers.auth import router as auth_router 
# Import conversation router
from routers.conversations import router as conversation_router
# Import cache service
from services.cache_service import cache_service
# Import models to ensure they're registered
import models
# endregion

# region Initialize FastAPI app
app = FastAPI(
    title="Project Aurelia - RAG System",
    description="Advanced Retrieval-Augmented Generation system with multiple search strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
# endregion
# region CORS middleware
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# endregion
# region Redis Memorystore setup
REDIS_HOST = os.getenv("REDIS_HOST", "10.0.0.5")  # Replace with your Memorystore IP
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
cache_service.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
# endregion
# region Include RAG router
# Include RAG router
app.include_router(rag_router)
# endregion
# region Include auth router
app.include_router(auth_router)
# endregion
# region Include conversation router
app.include_router(conversation_router)
# endregion
# region Application start time for uptime calculation
# Application start time for uptime calculation
app_start_time = datetime.now()
# endregion
# region Root endpoint
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Project Aurelia RAG System",
        "description": "Advanced Retrieval-Augmented Generation API",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0",
        "rag_endpoints": "/rag/",
        "health_check": "/rag/health"
    }
# endregion
# region Cache test endpoint
@app.get("/cache-test")
def cache_test():
    try:
        cache_service.set("test_key", "hello", ex=60)
        val = cache_service.get("test_key")
        return {"status": "ok", "value": val}
    except Exception as e:
        return {"status": "error", "error": str(e)}
# endregion
# region Startup event
# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Project Aurelia RAG System is starting up...")
    
    # Test Redis connection
    try:
        redis_healthy = await cache_service.health_check()
        if redis_healthy:
            print("✅ Redis connection established successfully")
        else:
            print("⚠️  Redis connection failed - caching will be disabled")
    except Exception as e:
        print(f"⚠️  Redis initialization error: {e}")
    
    print("✅ API is ready to serve requests!")
# endregion
# region Shutdown event
# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Project Aurelia RAG System is shutting down...")
    # Redis connections will be closed automatically by the connection pool
# endregion
# region Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# endregion