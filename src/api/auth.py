"""Authentication and authorization for the API."""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.database import User

settings = get_settings()
security = HTTPBearer()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.jwt_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def generate_api_key() -> str:
    """Generate a new API key."""
    return f"sk-{secrets.token_urlsafe(32)}"

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user

def get_user_by_token(db: Session, token: str) -> Optional[User]:
    """Get user by JWT token."""
    payload = verify_token(token)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    try:
        user_id = int(user_id)
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except (ValueError, TypeError):
        return None

def get_user_by_api_key(db: Session, api_key: str) -> Optional[User]:
    """Get user by API key."""
    user = db.query(User).filter(User.api_key == api_key).first()
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session)
) -> User:
    """Get the current authenticated user."""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    
    # Try JWT token first
    user = get_user_by_token(db, token)
    
    # If not JWT, try API key
    if not user:
        user = get_user_by_api_key(db, token)
    
    if not user:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def verify_api_key(request: Request, db: Session = Depends(get_db_session)) -> Optional[User]:
    """Verify API key from request headers."""
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
        user = get_user_by_api_key(db, api_key)
        if user and user.is_active:
            return user
    
    # Check X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = get_user_by_api_key(db, api_key)
        if user and user.is_active:
            return user
    
    return None

def create_user(db: Session, email: str, password: str, **kwargs) -> User:
    """Create a new user."""
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise ValueError("User with this email already exists")
    
    # Generate API key
    api_key = generate_api_key()
    
    # Create user
    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        api_key=api_key,
        api_key_created_at=datetime.utcnow(),
        **kwargs
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

def update_user_password(db: Session, user: User, new_password: str) -> User:
    """Update user password."""
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    db.refresh(user)
    return user

def regenerate_api_key(db: Session, user: User) -> str:
    """Regenerate API key for a user."""
    new_api_key = generate_api_key()
    user.api_key = new_api_key
    user.api_key_created_at = datetime.utcnow()
    db.commit()
    return new_api_key

def check_permissions(user: User, required_permission: str) -> bool:
    """Check if user has required permission."""
    # This is a simple role-based check
    # In a more complex system, you'd have a proper permissions system
    
    if required_permission == "admin":
        # Check if user is admin (you'd need to add admin field to User model)
        return getattr(user, 'is_admin', False)
    
    if required_permission == "analytics":
        # Allow analytics access for paid tiers
        return user.subscription_tier.value != "free"
    
    # Default: all authenticated users have basic permissions
    return True

class RequirePermission:
    """Dependency class to require specific permissions."""
    
    def __init__(self, permission: str):
        self.permission = permission
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if not check_permissions(current_user, self.permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {self.permission} required"
            )
        return current_user

# Pre-defined permission dependencies
require_admin = RequirePermission("admin")
require_analytics = RequirePermission("analytics")