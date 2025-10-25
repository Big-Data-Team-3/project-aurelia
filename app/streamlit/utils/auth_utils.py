# region Imports
import streamlit as st
import os
import requests
import urllib.parse
from urllib.parse import urlparse, parse_qs
import dotenv
# endregion

# region Environment Configuration
dotenv.load_dotenv()

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
# endregion

# region Google OAuth Functions
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

# Update your handle_google_callback function
def handle_google_callback(code):
    """Handle Google OAuth callback and store user in backend"""
    try:
        # Clear URL parameters immediately
        st.query_params.clear()
        
        # Check if we've already processed this code
        if "oauth_processed" in st.session_state and st.session_state.oauth_processed:
            st.info("✅ **OAuth already processed. You are logged in!**")
            return None
        
        # Check if credentials are set
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            st.error("❌ **Authentication Error:** Google OAuth credentials not configured!")
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
            return None
        
        token_response = response.json()
        st.success("✅ **Token exchange successful!**")
        
        if "access_token" in token_response:
            access_token = token_response["access_token"]
            
            # Get user info from Google
            st.write("🔄 **Fetching user information...**")
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
            user_response = requests.get(user_info_url, headers=headers)
            
            if user_response.status_code != 200:
                st.error(f"❌ **User Info Failed:** {user_response.status_code}")
                return None
            
            user_info = user_response.json()
            st.success("✅ **User info retrieved successfully!**")
            
            # Call FastAPI backend to store user and get auth token
            st.write("🔄 **Storing user in backend database...**")
            backend_response = store_user_in_backend(access_token)
            
            if backend_response:
                # Store backend auth token in session
                auth_token = backend_response["auth_token"]
                st.session_state.auth_token = auth_token
                st.session_state.user = backend_response["user"]
                st.session_state.user_id = backend_response["user"]["id"]
                st.session_state.messages = []
                st.session_state.oauth_processed = True
                
                # 🚀 NEW: Store token in localStorage for persistence
                store_auth_token_in_localstorage(auth_token)
                
                st.success("🎉 **Successfully logged in and stored in database!**")
                st.rerun()
            else:
                st.error("❌ **Failed to store user in backend database**")
                
        else:
            st.error("Failed to authenticate with Google")
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
    
    return None

# endregion

# region Backend Integration Functions
def store_user_in_backend(google_access_token):
    """Store user in FastAPI backend and get auth token"""
    try:
        # Call FastAPI backend
        backend_url = "http://localhost:8000/auth/google-login"
        
        response = requests.post(
            backend_url,
            json={"google_token": google_access_token},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ **Backend Connection Error:** Cannot connect to FastAPI backend at http://localhost:8000")
        st.write("Make sure your FastAPI server is running!")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ **Backend Timeout:** FastAPI backend is not responding")
        return None
    except Exception as e:
        st.error(f"Backend error: {str(e)}")
        return None

def validate_token_with_backend(token: str) -> dict:
    """Validate token with FastAPI backend"""
    try:
        response = requests.post(
            "http://localhost:8000/auth/validate-token",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def store_token_in_backend(google_user_data: dict) -> str:
    """Store user in backend and get auth token"""
    try:
        google_access_token = google_user_data["access_token"]
        response = requests.post(
            "http://localhost:8000/auth/google-login",
            json={"google_token": google_access_token}
        )
        if response.status_code == 200:
            return response.json()["auth_token"]
        return None
    except:
        return None
# endregion

# region LocalStorage and Token Persistence Functions
def store_auth_token_in_localstorage(auth_token: str):
    """Store auth token in localStorage for persistence"""
    try:
        import streamlit.components.v1 as components
        
        components.html(f"""
        <script>
            // Store auth token in localStorage
            localStorage.setItem('auth_token', '{auth_token}');
            console.log('Auth token stored in localStorage');
        </script>
        """, height=0)
        
        return True
    except Exception as e:
        print(f"Error storing auth token: {e}")
        return False

def clear_auth_token_from_localstorage():
    """Clear auth token from localStorage"""
    try:
        import streamlit.components.v1 as components
        
        components.html("""
        <script>
            // Clear auth token from localStorage
            localStorage.removeItem('auth_token');
            console.log('Auth token cleared from localStorage');
        </script>
        """, height=0)
        
        return True
    except Exception as e:
        print(f"Error clearing auth token: {e}")
        return False

def restore_user_from_backend():
    """Restore user session from backend token stored in localStorage"""
    try:
        import streamlit.components.v1 as components
        
        components.html("""
        <script>
            // Check localStorage for auth token
            const authToken = localStorage.getItem('auth_token');
            
            if (authToken) {
                console.log('Found auth token in localStorage:', authToken.substring(0, 20) + '...');
                
                // Check if token is already in URL to avoid infinite loops
                const currentUrl = new URL(window.location);
                if (!currentUrl.searchParams.has('restore_auth_token')) {
                    // Send token to Streamlit via URL parameters
                    currentUrl.searchParams.set('restore_auth_token', authToken);
                    window.history.replaceState({}, '', currentUrl);
                    console.log('Token added to URL parameters, waiting for Streamlit to process...');
                    
                    // Don't reload - let Streamlit handle the URL parameter change naturally
                } else {
                    console.log('Token already in URL parameters, skipping...');
                }
            } else {
                console.log('No auth token found in localStorage');
            }
        </script>
        """, height=0)
        
        return True
    except Exception as e:
        print(f"Error restoring user from backend: {e}")
        return False

def validate_backend_token(auth_token: str):
    """Validate backend auth token and get user data"""
    try:
        response = requests.post(
            "http://localhost:8000/auth/validate-token",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Token validation failed: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Backend connection error")
        return None
    except requests.exceptions.Timeout:
        print("❌ Backend timeout")
        return None
    except Exception as e:
        print(f"Token validation error: {e}")
        return None
# endregion

# region Session Management Functions
# Update your logout function
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
    if "auth_token" in st.session_state:
        st.session_state.auth_token = None
    
    # 🚀 NEW: Clear localStorage
    clear_auth_token_from_localstorage()
    
    st.success("✅ **Successfully logged out! Redirecting to home page...**")
    st.rerun()

# Update your is_logged_in function
def is_logged_in():
    """Check if the user is logged in with valid auth token"""
    # Check if we have an auth token in session state
    if st.session_state.get("auth_token"):
        # Check if we have user data in session (avoid redundant backend calls)
        if st.session_state.get("user") and st.session_state.get("user_id"):
            print(f"🔍 is_logged_in: User session found - {st.session_state.get('user', {}).get('email', 'Unknown')}")
            return True
        
        # Only validate with backend if we don't have user data
        print("🔍 is_logged_in: Validating token with backend...")
        user_data = validate_backend_token(st.session_state.auth_token)
        if user_data:
            # Update session state with fresh user data
            st.session_state.user = user_data["user"]
            st.session_state.user_id = user_data["user"]["id"]
            print(f"🔍 is_logged_in: Token validated successfully - {user_data['user'].get('email', 'Unknown')}")
            return True
        else:
            # Token is invalid, clear session
            print("🔍 is_logged_in: Token validation failed, clearing session")
            st.session_state.auth_token = None
            st.session_state.user = None
            st.session_state.user_id = None
            return False
    
    # No auth token in session state
    print("🔍 is_logged_in: No auth token found in session state")
    return False



def get_user_info():
    """Get the user information"""
    return st.session_state.get("user", None)
# endregion