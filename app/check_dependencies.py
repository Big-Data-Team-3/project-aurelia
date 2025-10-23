#!/usr/bin/env python3
"""
RAG System Dependency Checker
Verifies all requirements before starting the system
"""

import sys
import subprocess
import importlib
import os
import re
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'streamlit': 'Streamlit UI framework',
        'pinecone': 'Pinecone vector database client',
        'openai': 'OpenAI API client',
        'sentence_transformers': 'Sentence transformers for reranking',
        'wikipedia': 'Wikipedia API client',
        'rank_bm25': 'BM25 implementation',
        'pydantic': 'Data validation',
        'pydantic_settings': 'Pydantic settings',
        'requests': 'HTTP client',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'python-dotenv': 'Environment variable loader'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            # Handle special import names
            import_name = package.replace('-', '_')
            if package == 'python-dotenv':
                import_name = 'dotenv'
            elif package == 'rank_bm25':
                import_name = 'rank_bm25'
            
            importlib.import_module(import_name)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("📥 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def validate_openai_api_key(api_key):
    """Validate OpenAI API key format and test connectivity"""
    if not api_key:
        return False, "API key is empty"
    
    # Check format (OpenAI keys start with 'sk-' and are typically 51 characters)
    if not api_key.startswith('sk-'):
        return False, "Invalid format - should start with 'sk-'"
    
    if len(api_key) < 40:
        return False, "Invalid format - too short"
    
    # Test actual connectivity
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple test call
        response = client.models.list()
        if response and hasattr(response, 'data'):
            return True, "Valid and working"
        else:
            return False, "API key format valid but response unexpected"
            
    except openai.AuthenticationError:
        return False, "Authentication failed - invalid API key"
    except openai.RateLimitError:
        return True, "Valid (rate limited but authenticated)"
    except openai.APIConnectionError:
        return False, "Network connection failed"
    except Exception as e:
        return False, f"Test failed: {str(e)}"

def validate_pinecone_api_key(api_key):
    """Validate Pinecone API key format and test connectivity"""
    if not api_key:
        return False, "API key is empty"
    
    # Pinecone keys are typically UUIDs (36 characters with hyphens)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    if not uuid_pattern.match(api_key):
        return False, "Invalid format - should be UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
    
    # Test actual connectivity
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Try to list indexes to test connectivity
        indexes = pc.list_indexes()
        return True, f"Valid and working ({len(indexes)} indexes found)"
        
    except Exception as e:
        error_msg = str(e).lower()
        if "unauthorized" in error_msg or "invalid api key" in error_msg:
            return False, "Authentication failed - invalid API key"
        elif "forbidden" in error_msg:
            return True, "Valid (but access restricted)"
        else:
            return False, f"Test failed: {str(e)}"

def check_environment_file():
    """Check if .env file exists and validate API keys"""
    print("\n🔧 Checking environment configuration...")
    
    env_path = Path("app/.env")
    env_example_path = Path("app/.env.example")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        if env_example_path.exists():
            print("📝 Copy the example: cp app/.env.example app/.env")
        else:
            print("📝 Create app/.env with your API keys")
        print("\n📋 Required format:")
        print("OPENAI_API_KEY=sk-your-actual-openai-key")
        print("PINECONE_API_KEY=your-pinecone-uuid-key")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        print("❌ python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    
    print("✅ .env file found")
    
    # Validate API keys
    validation_passed = True
    
    # Check OpenAI API Key
    print("\n🔑 Validating OpenAI API Key...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        validation_passed = False
    elif openai_key.startswith("your_") or openai_key.strip() == "":
        print("❌ OPENAI_API_KEY is placeholder or empty")
        validation_passed = False
    else:
        is_valid, message = validate_openai_api_key(openai_key)
        if is_valid:
            print(f"✅ OpenAI API Key: {message}")
        else:
            print(f"❌ OpenAI API Key: {message}")
            validation_passed = False
    
    # Check Pinecone API Key
    print("\n🔑 Validating Pinecone API Key...")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        validation_passed = False
    elif pinecone_key.startswith("your_") or pinecone_key.strip() == "":
        print("❌ PINECONE_API_KEY is placeholder or empty")
        validation_passed = False
    else:
        is_valid, message = validate_pinecone_api_key(pinecone_key)
        if is_valid:
            print(f"✅ Pinecone API Key: {message}")
        else:
            print(f"❌ Pinecone API Key: {message}")
            validation_passed = False
    
    # Check optional configuration
    print("\n⚙️ Checking optional configuration...")
    optional_vars = {
        "PINECONE_INDEX_NAME": "fintbx-embeddings",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "GENERATION_MODEL": "gpt-4o-mini"
    }
    
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"✅ {var}: {value}")
    
    return validation_passed

def check_api_connectivity():
    """Test API connectivity with detailed diagnostics"""
    print("\n🌐 Testing API connectivity...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv("app/.env")
        
        # Test if we can import the RAG config
        try:
            sys.path.append(str(Path("app/backend").resolve()))
            from config.rag_config import RAGConfig
            config = RAGConfig()
            print("✅ RAG configuration loaded successfully")
            
            # Test if services can be initialized
            try:
                from services.vector_search import VectorSearchService
                vector_service = VectorSearchService(config)
                print("✅ Vector search service initialized")
            except Exception as e:
                print(f"⚠️  Vector search service failed: {e}")
            
            try:
                from services.generation import GenerationService
                generation_service = GenerationService(config)
                print("✅ Generation service initialized")
            except Exception as e:
                print(f"⚠️  Generation service failed: {e}")
                
        except Exception as e:
            print(f"❌ RAG configuration failed: {e}")
            
    except Exception as e:
        print(f"⚠️  API connectivity test failed: {e}")

def check_system_resources():
    """Check system resources"""
    print("\n💻 Checking system resources...")
    
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 4:
            print(f"✅ Memory: {memory_gb:.1f}GB available")
        else:
            print(f"⚠️  Memory: {memory_gb:.1f}GB (recommend 4GB+ for reranking models)")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        
        if disk_gb >= 2:
            print(f"✅ Disk space: {disk_gb:.1f}GB free")
        else:
            print(f"⚠️  Disk space: {disk_gb:.1f}GB free (recommend 2GB+ for models)")
            
    except ImportError:
        print("⚠️  Install psutil for detailed system checks: pip install psutil")

def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    data_paths = [
        "data/chunks/hybrid_intelligent_chunks_for_embedding.jsonl",
        "data/final/docling_blocks_cleaned.jsonl"
    ]
    
    missing_files = []
    for path in data_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} (missing)")
            missing_files.append(path)
    
    if missing_files:
        print("⚠️  Some data files are missing. RAG system will work but with limited content.")
        return False
    
    return True

def main():
    """Main dependency check function"""
    print("🔍 RAG System Dependency Check")
    print("=" * 60)
    
    critical_checks = [
        check_python_version(),
        check_required_packages(),
        check_environment_file(),
    ]
    
    # Optional checks
    check_api_connectivity()
    check_system_resources()
    check_data_files()
    
    print("\n" + "=" * 60)
    
    if all(critical_checks):
        print("🎉 All critical checks passed! System ready to start.")
        print("\n🚀 To start the RAG system:")
        print("1. Backend: cd app/backend && uvicorn main:app --reload --port 8000")
        print("2. Frontend: cd app/streamlit && streamlit run app.py --server.port 8501")
        print("3. Or use the automated script: python app/start_rag.py")
        return True
    else:
        print("❌ Some critical checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Create .env file with valid API keys")
        print("• Get OpenAI API key: https://platform.openai.com/api-keys")
        print("• Get Pinecone API key: https://app.pinecone.io/")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
