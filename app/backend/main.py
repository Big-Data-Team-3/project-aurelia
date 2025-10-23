from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn

# Import RAG router
from routers.rag import router as rag_router

# Initialize FastAPI app
app = FastAPI(
    title="Project Aurelia - RAG System",
    description="Advanced Retrieval-Augmented Generation system with multiple search strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include RAG router
app.include_router(rag_router)

# Application start time for uptime calculation
app_start_time = datetime.now()

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

# Simple system health endpoint (delegates to RAG health for detailed info)
@app.get("/health")
async def system_health():
    """Basic system health check - use /rag/health for detailed RAG system status"""
    uptime = datetime.now() - app_start_time
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": str(uptime),
        "detailed_health": "/rag/health"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Project Aurelia RAG System is starting up...")
    print("✅ API is ready to serve requests!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Project Aurelia RAG System is shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
