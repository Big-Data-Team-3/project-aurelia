import streamlit as st
import streamlit.components.v1 as components
from utils.auth_utils import login_with_google, is_logged_in, get_user_info, logout


def show():
    """Show the home page"""
    st.title("🧠 Project Aurelia")
    st.subheader("Welcome to the Intelligent RAG-powered Knowledge System")

    st.write("""
    **Project Aurelia** is an advanced AI assistant designed to help users extract meaningful insights from complex financial documents. 
    It leverages **Retrieval-Augmented Generation (RAG)** to combine:

    - **Knowledge retrieval**: Efficiently searches large corpora of structured and unstructured documents to find relevant information.  
    - **Reasoning**: Uses AI reasoning to contextualize and synthesize answers tailored to your queries.  
    - **Contextual awareness**: Maintains conversation context to provide coherent, accurate, and actionable responses over multiple interactions.

    The system can answer financial questions, highlight key metrics, extract formulas, assess risk levels, and even suggest follow-up questions to deepen analysis. 
    It integrates real-time AI responses with document sourcing transparency, ensuring that all generated insights can be traced back to their source.

    For more details, source code, and implementation guidelines, refer to the [Project Aurelia GitHub repository](https://github.com/Big-Data-Team-3/project-aurelia).
    """)





    # Get the Google OAuth URL
    google_auth_url = login_with_google()

    components.html(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 50px;">
            <button onclick="window.open('{google_auth_url}', '_blank')" style="
                background-color:#4285F4;
                color:white;
                border:none;
                padding:10px 20px;
                font-size:16px;
                text-align:center;
                cursor:pointer;">
                Login with Google
            </button>
        </div>
    """, height=50)
