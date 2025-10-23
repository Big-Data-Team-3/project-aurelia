from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import uvicorn
import pandas as pd
import numpy as np
from enum import Enum
import json

# Import RAG router
from routers.rag import router as rag_router

# Initialize FastAPI app
app = FastAPI(
    title="Project Aurelia API",
    description="A comprehensive FastAPI template for data processing and analytics",
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

# Pydantic Models
class StatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"

class CategoryEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class DataItem(BaseModel):
    id: int = Field(..., description="Unique identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    category: CategoryEnum = Field(..., description="Item category")
    score: float = Field(..., ge=0, le=100, description="Score between 0 and 100")
    status: StatusEnum = Field(..., description="Item status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DataItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    category: CategoryEnum
    score: float = Field(..., ge=0, le=100)
    status: StatusEnum = StatusEnum.active

class DataItemUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    category: Optional[CategoryEnum] = None
    score: Optional[float] = Field(None, ge=0, le=100)
    status: Optional[StatusEnum] = None

class AnalyticsRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    category: Optional[CategoryEnum] = None
    status: Optional[StatusEnum] = None

class AnalyticsResponse(BaseModel):
    total_items: int
    average_score: float
    category_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    score_statistics: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: str

# In-memory storage (replace with database in production)
data_store: List[DataItem] = []
app_start_time = datetime.now()

# Dependency for getting current user (placeholder)
def get_current_user():
    # Implement your authentication logic here
    return {"user_id": "demo_user", "username": "demo"}

# Utility functions
def generate_sample_data() -> List[DataItem]:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    sample_data = []
    
    for i in range(1, 101):  # Generate 100 sample items
        item = DataItem(
            id=i,
            name=f"Item {i}",
            category=np.random.choice([e.value for e in CategoryEnum]),
            score=round(np.clip(np.random.normal(75, 15), 0, 100), 2),
            status=np.random.choice([e.value for e in StatusEnum], p=[0.7, 0.2, 0.1]),
            created_at=datetime.now()
        )
        sample_data.append(item)
    
    return sample_data

# Initialize with sample data
if not data_store:
    data_store = generate_sample_data()

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Project Aurelia API",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - app_start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=str(uptime)
    )

# CRUD Operations for Data Items
@app.get("/items", response_model=List[DataItem])
async def get_items(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    category: Optional[CategoryEnum] = Query(None, description="Filter by category"),
    status: Optional[StatusEnum] = Query(None, description="Filter by status"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum score"),
    max_score: Optional[float] = Query(None, ge=0, le=100, description="Maximum score")
):
    """Get all items with optional filtering and pagination"""
    filtered_items = data_store
    
    # Apply filters
    if category:
        filtered_items = [item for item in filtered_items if item.category == category]
    if status:
        filtered_items = [item for item in filtered_items if item.status == status]
    if min_score is not None:
        filtered_items = [item for item in filtered_items if item.score >= min_score]
    if max_score is not None:
        filtered_items = [item for item in filtered_items if item.score <= max_score]
    
    # Apply pagination
    return filtered_items[skip:skip + limit]

@app.get("/items/{item_id}", response_model=DataItem)
async def get_item(item_id: int = Path(..., description="Item ID")):
    """Get a specific item by ID"""
    item = next((item for item in data_store if item.id == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/items", response_model=DataItem)
async def create_item(item: DataItemCreate, current_user: dict = Depends(get_current_user)):
    """Create a new item"""
    # Generate new ID
    new_id = max([item.id for item in data_store], default=0) + 1
    
    new_item = DataItem(
        id=new_id,
        name=item.name,
        category=item.category,
        score=item.score,
        status=item.status,
        created_at=datetime.now()
    )
    
    data_store.append(new_item)
    return new_item

@app.put("/items/{item_id}", response_model=DataItem)
async def update_item(
    item_id: int = Path(..., description="Item ID"),
    item_update: DataItemUpdate = ...,
    current_user: dict = Depends(get_current_user)
):
    """Update an existing item"""
    item_index = next((i for i, item in enumerate(data_store) if item.id == item_id), None)
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    existing_item = data_store[item_index]
    update_data = item_update.dict(exclude_unset=True)
    
    # Update only provided fields
    for field, value in update_data.items():
        setattr(existing_item, field, value)
    
    return existing_item

@app.delete("/items/{item_id}")
async def delete_item(
    item_id: int = Path(..., description="Item ID"),
    current_user: dict = Depends(get_current_user)
):
    """Delete an item"""
    item_index = next((i for i, item in enumerate(data_store) if item.id == item_id), None)
    if item_index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    deleted_item = data_store.pop(item_index)
    return {"message": f"Item {item_id} deleted successfully", "deleted_item": deleted_item}

# Analytics endpoints
@app.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: AnalyticsRequest):
    """Get analytics data based on filters"""
    filtered_items = data_store
    
    # Apply date filters (if dates were provided)
    if request.start_date or request.end_date:
        filtered_items = [
            item for item in filtered_items
            if (not request.start_date or item.created_at.date() >= request.start_date)
            and (not request.end_date or item.created_at.date() <= request.end_date)
        ]
    
    # Apply other filters
    if request.category:
        filtered_items = [item for item in filtered_items if item.category == request.category]
    if request.status:
        filtered_items = [item for item in filtered_items if item.status == request.status]
    
    if not filtered_items:
        return AnalyticsResponse(
            total_items=0,
            average_score=0.0,
            category_distribution={},
            status_distribution={},
            score_statistics={}
        )
    
    # Calculate statistics
    scores = [item.score for item in filtered_items]
    categories = [item.category.value for item in filtered_items]
    statuses = [item.status.value for item in filtered_items]
    
    category_dist = {cat: categories.count(cat) for cat in set(categories)}
    status_dist = {status: statuses.count(status) for status in set(statuses)}
    
    score_stats = {
        "mean": round(np.mean(scores), 2),
        "median": round(np.median(scores), 2),
        "std": round(np.std(scores), 2),
        "min": round(np.min(scores), 2),
        "max": round(np.max(scores), 2)
    }
    
    return AnalyticsResponse(
        total_items=len(filtered_items),
        average_score=round(np.mean(scores), 2),
        category_distribution=category_dist,
        status_distribution=status_dist,
        score_statistics=score_stats
    )

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get a quick analytics summary"""
    if not data_store:
        return {"message": "No data available"}
    
    scores = [item.score for item in data_store]
    categories = [item.category.value for item in data_store]
    statuses = [item.status.value for item in data_store]
    
    return {
        "total_items": len(data_store),
        "average_score": round(np.mean(scores), 2),
        "top_category": max(set(categories), key=categories.count),
        "most_common_status": max(set(statuses), key=statuses.count),
        "score_range": {
            "min": round(min(scores), 2),
            "max": round(max(scores), 2)
        }
    }

# Data export endpoint
@app.get("/export/csv")
async def export_csv(
    category: Optional[CategoryEnum] = Query(None, description="Filter by category"),
    status: Optional[StatusEnum] = Query(None, description="Filter by status")
):
    """Export data as CSV"""
    filtered_items = data_store
    
    if category:
        filtered_items = [item for item in filtered_items if item.category == category]
    if status:
        filtered_items = [item for item in filtered_items if item.status == status]
    
    # Convert to DataFrame for CSV export
    df = pd.DataFrame([item.dict() for item in filtered_items])
    csv_content = df.to_csv(index=False)
    
    return JSONResponse(
        content={"csv_data": csv_content},
        headers={"Content-Disposition": "attachment; filename=export.csv"}
    )

# Batch operations
@app.post("/items/batch", response_model=List[DataItem])
async def create_items_batch(
    items: List[DataItemCreate],
    current_user: dict = Depends(get_current_user)
):
    """Create multiple items at once"""
    if len(items) > 100:
        raise HTTPException(status_code=400, detail="Cannot create more than 100 items at once")
    
    created_items = []
    start_id = max([item.id for item in data_store], default=0) + 1
    
    for i, item_data in enumerate(items):
        new_item = DataItem(
            id=start_id + i,
            name=item_data.name,
            category=item_data.category,
            score=item_data.score,
            status=item_data.status,
            created_at=datetime.now()
        )
        data_store.append(new_item)
        created_items.append(new_item)
    
    return created_items

@app.delete("/items/batch")
async def delete_items_batch(
    item_ids: List[int],
    current_user: dict = Depends(get_current_user)
):
    """Delete multiple items at once"""
    deleted_items = []
    not_found_ids = []
    
    for item_id in item_ids:
        item_index = next((i for i, item in enumerate(data_store) if item.id == item_id), None)
        if item_index is not None:
            deleted_items.append(data_store.pop(item_index))
        else:
            not_found_ids.append(item_id)
    
    return {
        "deleted_items": deleted_items,
        "not_found_ids": not_found_ids,
        "message": f"Deleted {len(deleted_items)} items"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Project Aurelia API is starting up...")
    print(f"📊 Loaded {len(data_store)} sample items")
    print("✅ API is ready to serve requests!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Project Aurelia API is shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
