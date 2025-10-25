# routers/auth.py
# region Imports
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import requests
from models import User, AuthToken
from config.database import get_db
from config.database import DATABASE_URL
from pydantic import BaseModel
# endregion
#region Initialize Router
router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
#endregion
#region Google Login Request Model
class GoogleLoginRequest(BaseModel):
    google_token: str
#endregion
#region Database Connection Check
@router.get("/db/check")
async def check_db_connection():
    """Check if the database is connected"""
    try:
        print(f"Checking database connection to: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)
        print(f"Engine created: {engine}")
        engine.connect()
        print(f"Connection established")
        return {"status": "success", "message": "✅Database connected successfully! "}
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"❌ Database connection failed")

#endregion
#region Google Login
@router.post("/google-login")
async def google_login(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    """Validate Google token and create/update user + auth token"""
    try:
        print(f"🔍 Received Google token: {request.google_token[:20]}...")  # Debug log
        
        # Validate Google token
        user_info = validate_google_token(request.google_token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        
        print(f"✅ Google token validated for user: {user_info.get('email')}")
        
        # Check if user exists
        user = db.query(User).filter(User.google_id == user_info["id"]).first()
        
        if not user:
            # Create new user
            user = User(
                google_id=user_info["id"],
                email=user_info["email"],
                name=user_info["name"],
                profile_pic=user_info.get("picture")
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"🆕 New user created: {user.email}")
        else:
            # Update existing user
            user.email = user_info["email"]
            user.name = user_info["name"]
            user.profile_pic = user_info.get("picture")
            user.last_login = datetime.utcnow()
            db.commit()
            print(f"🔄 Existing user updated: {user.email}")
        
        # Check for existing valid token
        existing_token = db.query(AuthToken).filter(
            AuthToken.user_id == user.id,
            AuthToken.expires_at > datetime.utcnow()
        ).first()
        
        if existing_token:
            # Reuse existing token
            print(f"♻️ Reusing existing token for user: {user.email}")
            auth_token = existing_token.token
        else:
            # Create new token
            auth_token = create_auth_token(user.id, db)
            print(f"🆕 New token created for user: {user.email}")
        
        return {
            "auth_token": auth_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "profile_pic": user.profile_pic
            },
            "token_reused": existing_token is not None
        }
    except Exception as e:
        print(f"❌ Error in google_login: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
#endregion
#region Validate Google Token
def validate_google_token(token: str) -> dict:
    """Validate Google access token"""
    try:
        print(f"🔍 Validating Google token: {token[:20]}...")
        response = requests.get(
            f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token}",
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Google token validation successful for: {user_info.get('email')}")
            return user_info
        else:
            print(f"❌ Google token validation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Google token validation error: {str(e)}")
        return None
#endregion
#region Validate Token
@router.post("/validate-token")
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Validate stored auth token"""
    token = credentials.credentials
    
    # Check token in database (remove is_active check)
    auth_token = db.query(AuthToken).filter(
        AuthToken.token == token,
        # AuthToken.is_active == True,  # ❌ Remove this line
        AuthToken.expires_at > datetime.utcnow()
    ).first()
    
    if not auth_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Get user
    user = db.query(User).filter(User.id == auth_token.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "profile_pic": user.profile_pic
        }
    }
#endregion
#region Create Auth Token
def create_auth_token(user_id: int, db: Session) -> str:
    """Create a new auth token"""
    # Generate secure token
    token = secrets.token_urlsafe(32)
    
    # Create token record (remove is_active)
    auth_token = AuthToken(
        token=token,
        user_id=user_id,
        expires_at=datetime.utcnow() + timedelta(days=30)  # 30-day expiry
        # is_active=True  # ❌ Remove this line
    )
    
    db.add(auth_token)
    db.commit()
    
    return token
#endregion
#region Authentication Dependency
async def get_current_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Authentication dependency for protected endpoints"""
    token = credentials.credentials
    
    # Check token in database
    auth_token = db.query(AuthToken).filter(
        AuthToken.token == token,
        AuthToken.expires_at > datetime.utcnow()
    ).first()
    
    if not auth_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Get user
    user = db.query(User).filter(User.id == auth_token.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "profile_pic": user.profile_pic
        }
    }
#endregion

#