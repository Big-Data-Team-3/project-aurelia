"""
Configuration file for RAG Cache Demonstration DAG
Update these settings as needed for your environment
"""

# Backend API Configuration
BACKEND_API = "https://aurelia-backend-445621378597.us-central1.run.app" 

# Test Queries Configuration
TEST_QUERIES = [
    {
        "query": "What is term structure of interest rates?",
        "strategy": "rrf_fusion",
        "top_k": 10,
        "description": "Term structure of interest rates query"
    }
]

# DAG Configuration
DAG_CONFIG = {
    'owner': 'project-aurelia',
    'depends_on_past': False,
    'start_date': '2024-01-01',
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': 5,  # minutes
    'schedule_interval': None,  # Manual trigger only
    'catchup': False,
    'tags': ['rag', 'caching', 'performance', 'demo'],
}

# Request Configuration
REQUEST_CONFIG = {
    'timeout': 60,  # seconds
    'headers': {
        'Content-Type': 'application/json'
    }
}
