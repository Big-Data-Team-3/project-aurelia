import streamlit as st
from utils.auth_utils import login_with_google, is_logged_in, get_user_info, logout

st.set_page_config(page_title="Project Aurelia", layout="wide")

# Initialize session state with default values
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user_id = None
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.session_id = None

# Import your page modules
import home
import chat_ui

# Define pages based on login status
if not is_logged_in():
    # Run home page directly
    home.show()
else:
    # Run chat page directly for logged-in users
    chat_ui.show()