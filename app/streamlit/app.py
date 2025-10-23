import streamlit as st
import requests
import time
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Project Aurelia - RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Project Aurelia RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about financial documents and get AI-powered answers")
    
    # Main RAG Chat Interface
    show_rag_chat()

# Removed unused functions to keep the app simple and focused on RAG chat

def show_rag_chat():
    """Simple RAG Chat Interface"""
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API endpoint
        api_base_url = st.text_input(
            "API Base URL", 
            value="http://localhost:8000",
            help="FastAPI backend URL"
        )
        
        # Search strategy
        search_strategy = st.selectbox(
            "Search Strategy",
            ["rrf_fusion", "hybrid", "vector_only", "reranked"],
            index=0,
            help="How to search documents"
        )
        
        # Simple settings
        st.subheader("Settings")
        include_sources = st.checkbox("Show sources", value=True)
        enable_wikipedia = st.checkbox("Wikipedia fallback", value=True)
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced"):
            top_k = st.slider("Sources to find", 5, 20, 10)
            rerank_top_k = st.slider("Sources to use", 3, 10, 5)
            temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.slider("Max response length", 500, 2000, 1000, 100)
        
        # System status
        st.markdown("---")
        if st.button("🔍 Check System Health"):
            with st.spinner("Checking..."):
                health = check_rag_health(api_base_url)
                if health and health.get('status') == 'healthy':
                    st.success("✅ System is healthy!")
                else:
                    st.error("❌ System issues detected")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your financial document assistant. Ask me anything about the documents in the system!"
        })
    
    # Chat History Display
    st.markdown("### 💬 Chat")
    
    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and include_sources:
                if message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            if source.get('title'):
                                st.write(f"📄 {source['title']}")
                            if source.get('section'):
                                st.write(f"📍 Section: {source['section']}")
                            if source.get('pages'):
                                st.write(f"📖 Pages: {', '.join(map(str, source['pages']))}")
                            st.write(f"🎯 Relevance: {source.get('score', 0):.2f}")
                            st.write(f"📝 Preview: {source.get('text', '')[:150]}...")
                            if source.get('url'):
                                st.write(f"🔗 [View Source]({source['url']})")
                            st.markdown("---")
    
    # Chat Input
    if prompt := st.chat_input("Ask about financial documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating response..."):
                try:
                    # Prepare config
                    config = {
                        "api_base_url": api_base_url,
                        "search_strategy": search_strategy,
                        "top_k": top_k,
                        "rerank_top_k": rerank_top_k,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "enable_wikipedia": enable_wikipedia,
                        "include_sources": include_sources
                    }
                    
                    # Call RAG API
                    response = call_rag_api(prompt, config)
                    
                    if response and "answer" in response:
                        # Display response
                        st.write(response["answer"])
                        
                        # Show timing info
                        if "total_time_ms" in response:
                            st.caption(f"⏱️ Generated in {response['total_time_ms']:.0f}ms")
                        
                        # Add to chat history
                        assistant_msg = {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response.get("sources", [])
                        }
                        st.session_state.messages.append(assistant_msg)
                        
                    else:
                        error_msg = "❌ Sorry, I couldn't generate a response. Please check if the backend is running."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                        
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "👋 Chat cleared! Ask me anything about the financial documents."
            }]
            st.rerun()


def call_rag_api(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Call the RAG API endpoint"""
    try:
        url = f"{config['api_base_url']}/rag/query"
        
        payload = {
            "query": query,
            "strategy": config["search_strategy"],
            "top_k": config["top_k"],
            "rerank_top_k": config["rerank_top_k"],
            "include_sources": config["include_sources"],
            "enable_wikipedia_fallback": config["enable_wikipedia"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"]
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Make sure it's running on the configured URL.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The backend might be overloaded.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def check_rag_health(api_base_url: str) -> Dict[str, Any]:
    """Check RAG system health"""
    try:
        url = f"{api_base_url}/rag/health"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# Removed unused functions to keep the app simple and focused on RAG chat

if __name__ == "__main__":
    main()
