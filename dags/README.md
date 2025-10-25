# RAG Cache Demonstration DAG

This Airflow DAG demonstrates the performance advantages of caching in the RAG API by comparing response times between normal queries and cached queries.

## Overview

The DAG runs three different portfolio-related queries through the following sequence:
1. **Normal Query** - Processes without cache
2. **Cached Query** - Demonstrates cache performance

## Configuration

### Backend API URL
Update the `BACKEND_API` in `seed_dag.py`:
```python
# For local testing
BACKEND_API = "http://localhost:8000"

# For GCP Cloud Run deployment
BACKEND_API = "https://aurelia-backend-445621378597.us-central1.run.app"
```

### Test Queries
The DAG currently tests three portfolio-related queries:
```python
TEST_QUERIES = [
    {
        "query": "Common Operations on the Portfolio Object",
        "strategy": "rrf_fusion",
        "top_k": 10,
        "description": "Common Operations on the Portfolio Object query"
    },
    {
        "query": "Portfolio Optimization Theory",
        "strategy": "rrf_fusion",
        "top_k": 10,
        "description": "Portfolio Optimization Theory query"
    },
    {
        "query": "Role of Convexity in Portfolio Problems",
        "strategy": "rrf_fusion",
        "top_k": 10,
        "description": "Role of Convexity in Portfolio Problems query"
    }
]
```

## DAG Structure

```
start_demo
    ↓
normal_1 → cached_1
    ↓
normal_2 → cached_2
    ↓
normal_3 → cached_3
    ↓
final_report
    ↓
end_demo
```

## What the DAG Demonstrates

### 1. Normal Query Processing
- Runs query on `/rag/query` endpoint
- Processes without cache benefits
- Shows standard processing time

### 2. Cached Query Performance
- Runs same query on `/rag/query/cached` endpoint
- Demonstrates cache performance (may be cache hit or miss)
- Shows processing time with caching

### 3. Performance Comparison
- Compares normal vs cached query response times
- Calculates speedup factors
- Shows time savings and caching effectiveness
- Provides detailed performance metrics in final report

## Expected Results

You should see performance improvements like:
- **Normal query**: ~2000-5000ms (standard processing)
- **Cached query**: ~10-5000ms (depending on cache hit/miss)
- **Speedup**: 2-500x faster (when cache hit occurs)
- **Cache effectiveness**: Demonstrated through response time comparison

## Running the DAG

### 1. Prerequisites
- Airflow instance running
- Backend API accessible
- Redis service running (for caching)

### 2. Deploy DAG
```bash
# Copy DAG file to Airflow DAGs folder
cp seed_dag.py $AIRFLOW_HOME/dags/

# Or upload to Google Cloud Composer
gcloud composer environments storage dags import \
    --environment=ENVIRONMENT_NAME \
    --location=LOCATION \
    --source=seed_dag.py
```

### 3. Trigger DAG
- Open Airflow UI
- Find "rag_cache_demonstration" DAG
- Click "Trigger DAG"

Or trigger via gcloud CLI:
```bash
gcloud composer environments run ENVIRONMENT_NAME \
    --location=LOCATION \
    dags trigger \
    -- rag_cache_demonstration
```

### 4. Monitor Progress
- Watch task execution in Airflow UI
- Check logs for performance metrics
- Review final report for summary

## Log Output Example

```
🔄 Running NORMAL query: Common Operations on the Portfolio Object query
✅ Common Operations on the Portfolio Object query - /rag/query: 3245.67ms

⚡ Running CACHED query: Common Operations on the Portfolio Object query
✅ Common Operations on the Portfolio Object query - /rag/query/cached: 23.45ms

🎯 FINAL CACHING PERFORMANCE REPORT
================================================================================
📈 SUMMARY:
   Total Queries Tested: 3
   Successful Comparisons: 3
   Success Rate: 100.0%

   Common Operations on the Portfolio Object query: Normal 3245.67ms vs Cached 23.45ms (Speedup: 138.40x)
   Portfolio Optimization Theory query: Normal 2987.23ms vs Cached 15.32ms (Speedup: 194.99x)
   Role of Convexity in Portfolio Problems query: Normal 4123.45ms vs Cached 28.91ms (Speedup: 142.65x)

🚀 PERFORMANCE SUMMARY:
   Average Speedup: 158.68x
   Best Speedup: 194.99x
   Worst Speedup: 138.40x

✅ CACHING DEMONSTRATION COMPLETE!
================================================================================
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if backend API is running
   - Verify BACKEND_API URL in config.py

2. **Timeout Errors**
   - Increase timeout in REQUEST_CONFIG
   - Check backend performance

3. **Cache Not Working**
   - Verify Redis is running
   - Check cache configuration in backend

### Debug Mode
Enable debug logging in Airflow:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Customization

### Adding More Queries
Add entries to `TEST_QUERIES` in `seed_dag.py`:
```python
{
    "query": "Your custom query",
    "strategy": "rrf_fusion",
    "top_k": 10,
    "description": "Custom query description"
}
```

### Changing Test Strategy
Modify the DAG to test different scenarios:
- Different query strategies
- Various top_k values
- Different user contexts

## Performance Metrics

The DAG tracks:
- Response times for each query type
- Cache hit/miss ratios
- Speedup factors
- Time savings
- Success rates

## Integration with Monitoring

The DAG can be integrated with:
- Prometheus metrics
- Grafana dashboards
- Alert systems
- Performance monitoring tools
