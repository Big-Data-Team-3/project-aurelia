# routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import requests
from models.user import User, AuthToken
from config.database import get_db
from config.database import DATABASE_URL
router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
@router.post("/google-login")
async def google_login(google_token: str, db: Session = Depends(get_db)):
    """Validate Google token and create/update user + auth token"""
    try:
        # Validate Google token
        user_info = validate_google_token(google_token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        
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
        else:
            # Update existing user
            user.email = user_info["email"]
            user.name = user_info["name"]
            user.profile_pic = user_info.get("picture")
            db.commit()
        
        # Create auth token
        auth_token = create_auth_token(user.id, db)
        
        return {
            "auth_token": auth_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "profile_pic": user.profile_pic
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-token")
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Validate stored auth token"""
    token = credentials.credentials
    
    # Check token in database
    auth_token = db.query(AuthToken).filter(
        AuthToken.token == token,
        AuthToken.is_active == True,
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

def validate_google_token(token: str) -> dict:
    """Validate Google access token"""
    try:
        response = requests.get(
            f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token}"
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_auth_token(user_id: int, db: Session) -> str:
    """Create a new auth token"""
    # Generate secure token
    token = secrets.token_urlsafe(32)
    
    # Create token record
    auth_token = AuthToken(
        token=token,
        user_id=user_id,
        expires_at=datetime.utcnow() + timedelta(days=30)  # 30-day expiry
    )
    
    db.add(auth_token)
    db.commit()
    
    return token

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