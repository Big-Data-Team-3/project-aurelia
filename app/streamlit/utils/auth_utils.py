import streamlit as st
import os
import requests
import urllib.parse
from urllib.parse import urlparse, parse_qs
import dotenv

dotenv.load_dotenv()

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

def login_with_google():
    """Login with Google - returns auth URL or handles callback"""
    # Check if we're handling a callback
    query_params = st.query_params
    
    if "code" in query_params:
        # Handle OAuth callback
        code = query_params["code"]
        return handle_google_callback(code)
    else:
        # Generate auth URL
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={GOOGLE_REDIRECT_URI}&response_type=code&scope=openid%20email%20profile&access_type=offline"
        return auth_url

def handle_google_callback(code):
    """Handle Google OAuth callback"""
    try:
        # Clear URL parameters immediately to prevent re-processing on refresh
        st.query_params.clear()
        
        # Check if we've already processed this code (prevent duplicate processing)
        if "oauth_processed" in st.session_state and st.session_state.oauth_processed:
            st.info("✅ **OAuth already processed. You are logged in!**")
            return None
        
        # Check if credentials are set
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            st.error("❌ **Authentication Error:** Google OAuth credentials not configured!")
            st.write("Please set the following environment variables:")
            st.write("- `GOOGLE_CLIENT_ID`")
            st.write("- `GOOGLE_CLIENT_SECRET`")
            st.write("- `GOOGLE_REDIRECT_URI`")
            return None
        
        # Exchange code for token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": GOOGLE_REDIRECT_URI,
        }
        
        st.write("🔄 **Exchanging code for token...**")
        response = requests.post(token_url, data=token_data)
        
        if response.status_code != 200:
            st.error(f"❌ **Token Exchange Failed:** {response.status_code}")
            st.write("**Response:**", response.text)
            
            # Specific error handling for common issues
            if "invalid_grant" in response.text:
                st.error("🔍 **Common causes of 'invalid_grant' error:**")
                st.write("1. **Authorization code expired** - Codes expire in ~10 minutes")
                st.write("2. **Code already used** - Authorization codes can only be used once")
                st.write("3. **Redirect URI mismatch** - Check Google Cloud Console settings")
                st.write("4. **Client ID/Secret mismatch** - Verify your credentials")
                st.write("")
                st.write("**Try:** Refresh the page and try logging in again")
            
            return None
        
        token_response = response.json()
        st.success("✅ **Token exchange successful!**")
        
        if "access_token" in token_response:
            access_token = token_response["access_token"]
            
            # Get user info
            st.write("🔄 **Fetching user information...**")
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
            user_response = requests.get(user_info_url, headers=headers)
            
            if user_response.status_code != 200:
                st.error(f"❌ **User Info Failed:** {user_response.status_code}")
                st.write("**Response:**", user_response.text)
                return None
            
            user_info = user_response.json()
            st.success("✅ **User info retrieved successfully!**")
            st.write(f"**User:** {user_info.get('name', 'Unknown')} ({user_info.get('email', 'No email')})")
            
            # Store user info in session state
            user_data = {
                "id": user_info.get("id"),
                "name": user_info.get("name"),
                "email": user_info.get("email"),
                "profile_pic": user_info.get("picture"),
            }
            
            st.session_state.user_id = user_info.get("id")
            st.session_state.user = user_data
            st.session_state.messages = []
            
            # Mark OAuth as processed to prevent re-processing
            st.session_state.oauth_processed = True
            
            st.success("🎉 **Successfully logged in!**")
            st.rerun()
            
        else:
            st.error("Failed to authenticate with Google")
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
    
    return None

def logout():
    """Logout the user and redirect to home page"""
    if "user_id" in st.session_state:
        st.session_state.user_id = None
    if "session_id" in st.session_state:
        st.session_state.session_id = None
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "user" in st.session_state:
        st.session_state.user = None
    if "oauth_processed" in st.session_state:
        st.session_state.oauth_processed = None
    
    st.success("✅ **Successfully logged out! Redirecting to home page...**")
    st.rerun()

def is_logged_in():
    """Check if the user is logged in"""
    return st.session_state.get("user_id") is not None

def get_user_info():
    """Get the user information"""
    return st.session_state.get("user", None)