#!/usr/bin/env python3
"""
RAG System Automated Startup
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def run_dependency_check():
    """Run the dependency checker first"""
    print("🔍 Running dependency check...")
    
    check_script = Path(__file__).parent / "check_dependencies.py"
    if check_script.exists():
        result = subprocess.run([sys.executable, str(check_script)], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        return result.returncode == 0
    else:
        print("⚠️  Dependency checker not found, proceeding anyway...")
        return True

def start_backend():
    """Start FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return None
    
    original_dir = os.getcwd()
    os.chdir(backend_dir)
    
    cmd = [sys.executable, "-m", "uvicorn", "main:app", 
           "--reload", "--host", "0.0.0.0", "--port", "8000"]
    
    process = subprocess.Popen(cmd)
    os.chdir(original_dir)  # Return to original directory
    return process

def start_streamlit():
    """Start Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    
    streamlit_dir = Path(__file__).parent / "streamlit"
    if not streamlit_dir.exists():
        print("❌ Streamlit directory not found!")
        return None
    
    original_dir = os.getcwd()
    os.chdir(streamlit_dir)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
           "--server.port", "8501", "--server.address", "0.0.0.0"]
    
    process = subprocess.Popen(cmd)
    os.chdir(original_dir)  # Return to original directory
    return process

def wait_for_backend(timeout=60):
    """Wait for backend to be ready"""
    print("⏳ Waiting for backend to start...")
    
    for i in range(timeout):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            print(f"   Still waiting... ({i}/{timeout}s)")
    
    print("❌ Backend failed to start within timeout")
    return False

def check_rag_health():
    """Check RAG system health"""
    print("🔍 Checking RAG system health...")
    
    try:
        response = requests.get("http://localhost:8000/rag/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ RAG Status: {health.get('status', 'unknown')}")
            
            components = [
                ("Pinecone", health.get('pinecone_connected', False)),
                ("OpenAI", health.get('openai_connected', False)),
                ("Reranker", health.get('reranker_loaded', False)),
                ("Wikipedia", health.get('wikipedia_available', False))
            ]
            
            for name, status in components:
                icon = "✅" if status else "❌"
                print(f"   {icon} {name}")
            
            return health.get('status') == 'healthy'
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Main startup function"""
    print("🚀 Project Aurelia RAG System")
    print("=" * 50)
    
    # Check dependencies first
    if not run_dependency_check():
        print("❌ Dependency check failed. Please fix issues and try again.")
        sys.exit(1)
    
    try:
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            sys.exit(1)
        
        # Wait for backend
        if not wait_for_backend():
            backend_process.terminate()
            sys.exit(1)
        
        # Check RAG health
        rag_healthy = check_rag_health()
        if not rag_healthy:
            print("⚠️  RAG system not fully healthy, but continuing...")
        
        # Start Streamlit
        streamlit_process = start_streamlit()
        if not streamlit_process:
            backend_process.terminate()
            sys.exit(1)
        
        print("\n🎉 RAG System Started Successfully!")
        print("=" * 50)
        print("🔗 FastAPI Backend: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🎨 Streamlit App: http://localhost:8501")
        print("🤖 RAG Chat: http://localhost:8501 → 'RAG Chat' tab")
        print("\n💡 Press Ctrl+C to stop all services")
        
        # Wait for interrupt
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            backend_process.terminate()
            streamlit_process.terminate()
            
            # Wait for clean shutdown
            try:
                backend_process.wait(timeout=5)
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                streamlit_process.kill()
            
            print("✅ All services stopped!")
    
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
