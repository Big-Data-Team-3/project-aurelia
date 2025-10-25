import streamlit as st
from utils.auth_utils import (
    login_with_google, is_logged_in, get_user_info, logout, 
    restore_user_from_backend, validate_backend_token
)

st.set_page_config(page_title="Project Aurelia", layout="wide")

# Initialize session state with default values
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user_id = None
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.auth_token = None

# 🚀 NEW: Check for restored user session from backend token
if not st.session_state.get("user_id") and not st.session_state.get("user"):
    # Check for auth token in URL parameters (from localStorage)
    restore_auth_token = st.query_params.get("restore_auth_token")
    if restore_auth_token:
        try:
            print(f"🔍 app.py: Restoring session with token: {restore_auth_token[:20]}...")
            # Validate token with backend
            user_data = validate_backend_token(restore_auth_token)
            if user_data:
                print(f"🔍 app.py: Token validation successful for user: {user_data['user'].get('email', 'Unknown')}")
                # Restore session state from backend
                st.session_state.auth_token = restore_auth_token
                st.session_state.user_id = user_data["user"]["id"]
                st.session_state.user = user_data["user"]
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.oauth_processed = True
                
                print(f"🔍 app.py: Session state restored - user_id: {st.session_state.user_id}, user: {st.session_state.user.get('email', 'Unknown')}")
                
                # Clear the URL parameter
                st.query_params.clear()
                # Don't call st.rerun() here as it resets session state
            else:
                print("🔍 app.py: Token validation failed")
                # Token is invalid, clear URL parameter
                st.query_params.clear()
        except Exception as e:
            print(f"🔍 app.py: Error restoring session: {e}")
            st.error(f"Error restoring user session from backend: {e}")
            # Clear the URL parameter
            st.query_params.clear()
    
    # localStorage restoration is now handled in is_logged_in() function

# Import your page modules
import home
import chat_ui

# Define pages based on login status
print(f"🔍 app.py: Checking login status...")
login_status = is_logged_in()
print(f"🔍 app.py: Login status: {login_status}")

if not login_status:
    print("🔍 app.py: User not logged in, showing home page")
    # Run home page directly
    home.show()
else:
    print("🔍 app.py: User logged in, showing chat UI")
    # Run chat page directly for logged-in users
    chat_ui.show()