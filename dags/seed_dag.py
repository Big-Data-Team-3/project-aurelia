"""
Airflow DAG to demonstrate RAG API caching performance
This DAG runs queries on both /query and /query/cached endpoints to show caching advantages
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
import requests
import json
import time
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'project-aurelia',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'seed_dag',
    default_args=default_args,
    description='Demonstrate RAG API caching performance advantages',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['rag', 'caching', 'performance', 'demo'],
)

# Configuration
BACKEND_API = "https://aurelia-backend-445621378597.us-central1.run.app"  # Update this to your actual backend URL
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

def make_api_request(endpoint, query_data, query_name):
    """Make API request and return response with timing"""
    url = f"{BACKEND_API}{endpoint}"
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json=query_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"✅ {query_name} - {endpoint}: {response_time:.2f}ms")
            return {
                "success": True,
                "response_time_ms": response_time,
                "response": result,
                "query_name": query_name,
                "endpoint": endpoint
            }
        else:
            logging.error(f"❌ {query_name} - {endpoint}: HTTP {response.status_code}")
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "query_name": query_name,
                "endpoint": endpoint
            }
            
    except Exception as e:
        logging.error(f"❌ {query_name} - {endpoint}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "query_name": query_name,
            "endpoint": endpoint
        }

def run_normal_query(**context):
    """Run query on /query endpoint"""
    query_data = context['params']['query_data']
    query_name = context['params']['query_name']
    
    logging.info(f"🔄 Running NORMAL query: {query_name}")
    result = make_api_request("/rag/query", query_data, f"{query_name} (NORMAL)")
    
    # Store result in XCom for comparison
    context['task_instance'].xcom_push(key=f"{query_name}_normal", value=result)
    return result

def run_cached_query(**context):
    """Run query on /query/cached endpoint to show cache performance"""
    query_data = context['params']['query_data']
    query_name = context['params']['query_name']
    
    logging.info(f"⚡ Running CACHED query: {query_name}")
    result = make_api_request("/rag/query/cached", query_data, f"{query_name} (CACHED)")
    
    # Store result in XCom for comparison
    context['task_instance'].xcom_push(key=f"{query_name}_cached", value=result)
    return result

def compare_performance(**context):
    """Compare performance between normal and cached queries"""
    query_name = context['params']['query_name']
    
    # Get results from XCom
    cached_first = context['task_instance'].xcom_pull(key=f"{query_name}_cached_first")
    normal = context['task_instance'].xcom_pull(key=f"{query_name}_normal")
    cached_second = context['task_instance'].xcom_pull(key=f"{query_name}_cached_second")
    
    logging.info(f"\n📊 PERFORMANCE COMPARISON FOR: {query_name}")
    logging.info("=" * 60)
    
    if cached_first and cached_first.get('success'):
        logging.info(f"1️⃣ CACHED (First Run): {cached_first['response_time_ms']:.2f}ms")
        if 'response' in cached_first and 'metadata' in cached_first['response']:
            metadata = cached_first['response']['metadata']
            logging.info(f"   Cache Hit: {metadata.get('cache_hit', 'N/A')}")
            logging.info(f"   Processing Time: {metadata.get('processing_time_ms', 'N/A')}ms")
    
    if normal and normal.get('success'):
        logging.info(f"2️⃣ NORMAL Query: {normal['response_time_ms']:.2f}ms")
        if 'response' in normal and 'metadata' in normal['response']:
            metadata = normal['response']['metadata']
            logging.info(f"   Cache Hit: {metadata.get('cache_hit', 'N/A')}")
            logging.info(f"   Processing Time: {metadata.get('processing_time_ms', 'N/A')}ms")
    
    if cached_second and cached_second.get('success'):
        logging.info(f"3️⃣ CACHED (Second Run): {cached_second['response_time_ms']:.2f}ms")
        if 'response' in cached_second and 'metadata' in cached_second['response']:
            metadata = cached_second['response']['metadata']
            logging.info(f"   Cache Hit: {metadata.get('cache_hit', 'N/A')}")
            logging.info(f"   Speedup Factor: {metadata.get('speedup_factor', 'N/A')}")
            logging.info(f"   Cache Performance: {metadata.get('cache_performance', 'N/A')}")
    
    # Calculate speedup
    if (cached_first and cached_first.get('success') and 
        cached_second and cached_second.get('success')):
        
        first_time = cached_first['response_time_ms']
        second_time = cached_second['response_time_ms']
        speedup = first_time / second_time if second_time > 0 else 0
        
        logging.info(f"\n🚀 CACHING SPEEDUP: {speedup:.2f}x faster!")
        logging.info(f"   First run: {first_time:.2f}ms")
        logging.info(f"   Second run: {second_time:.2f}ms")
        logging.info(f"   Time saved: {first_time - second_time:.2f}ms")
    
    logging.info("=" * 60)
    
    return {
        "query_name": query_name,
        "cached_first": cached_first,
        "normal": normal,
        "cached_second": cached_second
    }

def generate_final_report(**context):
    """Generate final performance report comparing normal vs cached queries"""
    logging.info("\n🎯 FINAL CACHING PERFORMANCE REPORT")
    logging.info("=" * 80)
    
    all_results = []
    for query in TEST_QUERIES:
        query_name = query['description']
        
        # Get results from XCom for normal and cached queries
        normal_result = context['task_instance'].xcom_pull(key=f"{query_name}_normal")
        cached_result = context['task_instance'].xcom_pull(key=f"{query_name}_cached")
        
        if normal_result and cached_result:
            comparison = {
                "query_name": query_name,
                "normal": normal_result,
                "cached": cached_result
            }
            all_results.append(comparison)
    
    total_queries = len(all_results)
    successful_queries = sum(1 for r in all_results if r.get('normal', {}).get('success') and r.get('cached', {}).get('success'))
    
    logging.info(f"📈 SUMMARY:")
    logging.info(f"   Total Queries Tested: {total_queries}")
    logging.info(f"   Successful Comparisons: {successful_queries}")
    
    if total_queries > 0:
        logging.info(f"   Success Rate: {(successful_queries/total_queries)*100:.1f}%")
    else:
        logging.warning("   No queries were processed - cannot calculate success rate")
    
    # Calculate average speedup (cached vs normal)
    speedups = []
    for result in all_results:
        normal = result.get('normal', {})
        cached = result.get('cached', {})
        
        if (normal.get('success') and cached.get('success')):
            normal_time = normal['response_time_ms']
            cached_time = cached['response_time_ms']
            if cached_time > 0:
                speedup = normal_time / cached_time
                speedups.append(speedup)
                logging.info(f"   {result['query_name']}: Normal {normal_time:.2f}ms vs Cached {cached_time:.2f}ms (Speedup: {speedup:.2f}x)")
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        logging.info(f"\n🚀 PERFORMANCE SUMMARY:")
        logging.info(f"   Average Speedup: {avg_speedup:.2f}x")
        logging.info(f"   Best Speedup: {max(speedups):.2f}x")
        logging.info(f"   Worst Speedup: {min(speedups):.2f}x")
    else:
        logging.warning("   No successful speedup calculations available")
        avg_speedup = 0
    
    logging.info("\n✅ CACHING DEMONSTRATION COMPLETE!")
    logging.info("=" * 80)
    
    return {
        "total_queries": total_queries,
        "successful_demos": successful_queries,
        "average_speedup": avg_speedup if speedups else 0,
        "results": all_results
    }

# Create tasks for each query
tasks = []

for i, query in enumerate(TEST_QUERIES):
    query_name = query['description']
    query_data = {
        "query": query['query'],
        "strategy": query['strategy'],
        "top_k": query['top_k'],
        "user_id": "airflow_demo_user"
    }
    
    # Task 1: Run normal query
    normal_task = PythonOperator(
        task_id=f'normal_{i+1}',
        python_callable=run_normal_query,
        params={
            'query_data': query_data,
            'query_name': query_name
        },
        dag=dag,
    )
    
    # Task 2: Run cached query
    cached_task = PythonOperator(
        task_id=f'cached_{i+1}',
        python_callable=run_cached_query,
        params={
            'query_data': query_data,
            'query_name': query_name
        },
        dag=dag,
    )
    
    # Set task dependencies - normal first, then cached
    normal_task >> cached_task
    
    tasks.extend([normal_task, cached_task])

# Final report task
final_report_task = PythonOperator(
    task_id='final_report',
    python_callable=generate_final_report,
    dag=dag,
)

# Set final dependency - all cached tasks should complete before final report
for i in range(len(TEST_QUERIES)):
    dag.get_task(f'cached_{i+1}') >> final_report_task

# Add a start task
start_task = BashOperator(
    task_id='start_demo',
    bash_command='echo "🚀 Starting RAG Caching Performance Demonstration..."',
    dag=dag,
)

# Add an end task
end_task = BashOperator(
    task_id='end_demo',
    bash_command='echo "✅ RAG Caching Performance Demonstration Complete!"',
    dag=dag,
)

# Set start and end dependencies
start_task >> tasks[0]  # First task of first query
final_report_task >> end_task
