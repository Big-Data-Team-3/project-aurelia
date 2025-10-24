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
    .quick-chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    .chat-bubble-button .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1.5rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        white-space: pre-line;
    }
    .chat-bubble-button .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    .chat-bubble-button .stButton > button:active {
        transform: translateY(0px);
    }
    .quick-chat-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .centered-subtitle {
        text-align: center !important;
        margin: 1rem 0;
        width: 100%;
        display: block;
    }
    .centered-subtitle h3 {
        text-align: center !important;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Project Aurelia RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown('<div class="centered-subtitle">Ask questions about financial documents and get AI-powered answers</div>', unsafe_allow_html=True)
    
    # Quick Chat Bubbles
    show_quick_chat_bubbles()
    
    # Main RAG Chat Interface
    show_rag_chat()

def show_quick_chat_bubbles():
    """Display quick chat bubbles for common queries using cached API"""
    
    # Define the three quick chat options
    quick_queries = [
        {
            "title": "What is Sharpe Ratio?",
            "query": "What is Sharpe Ratio?",
            "description": "Understand the Sharpe Ratio"
        },
        {
            "title": "How to build a diversified investment portfolio?", 
            "query": "How to build a diversified investment portfolio?",
            "description": "Understand the diversification of an investment portfolio"
        },
        {
            "title": "What are the different types of financial risks?",
            "query": "What are the different types of financial risks?",
            "description": "Understand the different types of financial risks"
        }
    ]
    
    # Add title for quick chat section
    st.markdown('<div class="quick-chat-title">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create three columns for the bubbles
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="chat-bubble-button">', unsafe_allow_html=True)
        if st.button(
            f"**{quick_queries[0]['title']}**\n\n{quick_queries[0]['description']}",
            key="quick_1",
            help="Click to ask about financial ratios",
            use_container_width=True
        ):
            # Process the query using cached API
            process_quick_query(quick_queries[0]['query'], quick_queries[0]['title'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chat-bubble-button">', unsafe_allow_html=True)
        if st.button(
            f"**{quick_queries[1]['title']}**\n\n{quick_queries[1]['description']}",
            key="quick_2", 
            help="Click to ask about portfolio management",
            use_container_width=True
        ):
            # Process the query using cached API
            process_quick_query(quick_queries[1]['query'], quick_queries[1]['title'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chat-bubble-button">', unsafe_allow_html=True)
        if st.button(
            f"**{quick_queries[2]['title']}**\n\n{quick_queries[2]['description']}",
            key="quick_3",
            help="Click to ask about risk analysis", 
            use_container_width=True
        ):
            # Process the query using cached API
            process_quick_query(quick_queries[2]['query'], quick_queries[2]['title'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add separator
    st.markdown("---")

def process_quick_query(query: str, title: str):
    """Process a quick query using the cached API and add it to chat history"""
    
    # Add the query as a user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get API base URL from session state or use default
    api_base_url = "http://localhost:8000"  # Default value
    
    # Process the query using cached API (invisible to user)
    try:
        # Call the cached API endpoint
        response = call_cached_rag_api(query, api_base_url)
        
        if response and "answer" in response:
            # Add the response as an assistant message to chat history
            assistant_msg = {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "metadata": response.get("metadata", {})  # Store metadata for internal use
            }
            st.session_state.messages.append(assistant_msg)
            
            # Store session_id if provided
            if "session_id" in response and response["session_id"]:
                if hasattr(st.session_state, "session_id"):
                    st.session_state.session_id = response["session_id"]
            
            # Force a rerun to show the new messages
            st.rerun()
        else:
            # Add error message to chat history
            error_msg = "❌ Sorry, I couldn't generate a response. Please check if the backend is running."
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg
            })
            st.rerun()
            
    except Exception as e:
        # Add error message to chat history
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.messages.append({
            "role": "assistant", 
            "content": error_msg
        })
        st.rerun()

def call_cached_rag_api(query: str, api_base_url: str) -> Dict[str, Any]:
    """Call the cached RAG API endpoint"""
    try:
        url = f"{api_base_url}/rag/query/cached"
        
        payload = {
            "query": query,
            "strategy": "rrf_fusion",
            "top_k": 10,
            "rerank_top_k": 5,
            "include_sources": True,
            "enable_wikipedia_fallback": True,
            "temperature": 0.1,
            "max_tokens": 1000,
            "user_id": "streamlit_user"
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

def show_rag_chat():
    """Simple RAG Chat Interface"""
    
    # Initialize session state first (before any other code that might access it)
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = "streamlit_user"
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your financial document assistant. Ask me anything about the documents in the system!"
        })
    
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
        
        # Session Management
        st.markdown("---")
        st.subheader("🔗 Session Management")
        
        if st.session_state.session_id:
            st.success(f"✅ Active Session")
            
            if st.button("🔄 New Session"):
                st.session_state.session_id = None
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "👋 New session started! Ask me anything about the financial documents."
                }]
                st.rerun()
        else:
            st.info("🆕 No active session")
            st.caption("A new session will be created with your first query")
        
        # Cache Statistics
        st.markdown("---")
        st.subheader("📊 Cache Performance")
        if st.button("📈 View Cache Stats"):
            with st.spinner("Loading cache statistics..."):
                cache_stats = get_cache_stats(api_base_url)
                if cache_stats:
                    stats = cache_stats.get('cache_stats', {})
                    st.metric("Cached Responses", stats.get('total_cached_responses', 0))
                    st.metric("Avg Processing Time", f"{stats.get('average_processing_time_ms', 0):.0f}ms")
                    st.caption(f"💾 {stats.get('cache_hit_potential', 'No cached queries')}")
                else:
                    st.error("❌ Could not load cache stats")
        
        # System status
        st.markdown("---")
        if st.button("🔍 Check System Health"):
            with st.spinner("Checking..."):
                health = check_rag_health(api_base_url)
                if health and health.get('status') == 'healthy':
                    st.success("✅ System is healthy!")
                else:
                    st.error("❌ System issues detected")
        
        # Add source statistics if available
        if "messages" in st.session_state and st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if (last_message.get("role") == "assistant" and 
                "sources" in last_message and 
                last_message["sources"]):
                
                wikipedia_count = len([s for s in last_message["sources"] if s.get('source_type') == 'wikipedia'])
                document_count = len([s for s in last_message["sources"] if s.get('source_type') != 'wikipedia'])
                
    
    # Session state is already initialized at the beginning of the function
    
    # Chat History Display
    st.markdown("### 💬 Chat")
    
    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show subtle cache performance info for assistant messages
            if (message["role"] == "assistant" and 
                "metadata" in message and 
                message["metadata"]):
                metadata = message["metadata"]
                if metadata.get("cache_hit"):
                    st.caption(f"⚡ Cached response ({metadata.get('speedup_factor', 0):.0f}x faster)")
                elif metadata.get("cache_stored"):
                    st.caption("💾 Response cached for future use")
            
            # Show sources for assistant messages
            if (message["role"] == "assistant" and 
                include_sources and 
                "sources" in message and 
                message["sources"] and 
                len(message["sources"]) > 0):
                
                # Check source types for this message
                wikipedia_sources = [s for s in message["sources"] if s.get('source_type') == 'wikipedia']
                document_sources = [s for s in message["sources"] if s.get('source_type') != 'wikipedia']
                
                # Show source summary for chat history
                if wikipedia_sources and document_sources:
                    st.caption(f"📚 {len(document_sources)} documents + 🌐 {len(wikipedia_sources)} Wikipedia")
                elif wikipedia_sources:
                    st.caption(f"🌐 {len(wikipedia_sources)} Wikipedia sources")
                elif document_sources:
                    st.caption(f"📄 {len(document_sources)} document sources")
                
                with st.expander(f"📚 View {len(message['sources'])} sources"):
                    for i, source in enumerate(message["sources"], 1):
                        # Add source type indicator
                        source_type = source.get('source_type', 'document')
                        if source_type == 'wikipedia':
                            st.markdown(f"**Source {i}:** 🌐 **Wikipedia**")
                        else:
                            st.markdown(f"**Source {i}:** 📄 **Document**")
                        
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
                        
                        # CRITICAL FIX: Add to chat history BEFORE showing sources
                        assistant_msg = {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response.get("sources", [])
                        }
                        st.session_state.messages.append(assistant_msg)
                        
                        # Show sources immediately after adding to session state
                        if (include_sources and 
                            "sources" in response and 
                            response["sources"] and 
                            len(response["sources"]) > 0):
                            
                            # Check if Wikipedia was used
                            wikipedia_sources = [s for s in response["sources"] if s.get('source_type') == 'wikipedia']
                            document_sources = [s for s in response["sources"] if s.get('source_type') != 'wikipedia']
                            
                            # Show source summary
                            if wikipedia_sources:
                                st.info(f"📚 **Wikipedia Sources**: {len(wikipedia_sources)} results from Wikipedia")
                            if document_sources:
                                st.success(f"📄 **Document Sources**: {len(document_sources)} results from your documents")
                            
                            # Add Wikipedia warning if applicable
                            if response.get("metadata", {}).get("wikipedia_fallback_used"):
                                st.warning("⚠️ **Note**: This response includes information from Wikipedia as I couldn't find sufficient relevant information in your document corpus.")
                            
                            with st.expander(f"📚 View {len(response['sources'])} sources"):
                                for i, source in enumerate(response["sources"], 1):
                                    # Add source type indicator
                                    source_type = source.get('source_type', 'document')
                                    if source_type == 'wikipedia':
                                        st.markdown(f"**Source {i}:** 🌐 **Wikipedia**")
                                    else:
                                        st.markdown(f"**Source {i}:** 📄 **Document**")
                                    
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
    """Call the RAG API endpoint with session management"""
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
            "max_tokens": config["max_tokens"],
            "session_id": getattr(st.session_state, "session_id", None),  # Pass existing session
            "user_id": getattr(st.session_state, "user_id", "streamlit_user"),
            "create_session": True,  # Allow session creation
            "stream": True
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Store session_id for next request
        if "session_id" in data and data["session_id"]:
            if hasattr(st.session_state, "session_id"):
                st.session_state.session_id = data["session_id"]
        
        return data
        
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


def get_cache_stats(api_base_url: str) -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        url = f"{api_base_url}/rag/cache/stats"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
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
