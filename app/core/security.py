"""
Security utilities for JWT authentication, password hashing, and security measures.
"""

from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets
import hashlib
import hmac
import time
from .config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Token models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    scopes: list[str] = []

class RefreshTokenData(BaseModel):
    user_id: str
    token_id: str
    expires_at: datetime

# Security utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)

def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(user_id: str) -> tuple[str, str]:
    """Create refresh token and return token and token_id."""
    token_id = generate_secure_token(16)
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "user_id": user_id,
        "token_id": token_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt, token_id

def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Verify token type
        if payload.get("type") != token_type:
            return None
            
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            return None
            
        return payload
    except JWTError:
        return None

def validate_api_key(api_key: str, expected_hash: str) -> bool:
    """Validate API key using secure comparison."""
    if not api_key or not expected_hash:
        return False
    
    # Create hash of provided API key
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Secure comparison to prevent timing attacks
    return hmac.compare_digest(api_key_hash, expected_hash)

def generate_api_key() -> tuple[str, str]:
    """Generate API key and return key and hash."""
    api_key = f"sk-{generate_secure_token(32)}"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return api_key, api_key_hash

def create_csrf_token(user_id: str) -> str:
    """Create CSRF token for form protection."""
    timestamp = str(int(time.time()))
    message = f"{user_id}:{timestamp}"
    signature = hmac.new(
        settings.SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{message}:{signature}"

def verify_csrf_token(token: str, user_id: str, max_age: int = 3600) -> bool:
    """Verify CSRF token."""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False
            
        token_user_id, timestamp, signature = parts
        
        # Verify user ID
        if token_user_id != user_id:
            return False
            
        # Verify age
        token_time = int(timestamp)
        if time.time() - token_time > max_age:
            return False
            
        # Verify signature
        message = f"{token_user_id}:{timestamp}"
        expected_signature = hmac.new(
            settings.SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    except (ValueError, TypeError):
        return False

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

def sanitize_input(text: str) -> str:
    """Basic input sanitization."""
    if not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "\"", "'", "&", ";", "(", ")", "{", "}", "[", "]"]
    sanitized = text
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    return sanitized.strip()

def is_safe_url(url: str, allowed_hosts: list[str]) -> bool:
    """Check if URL is safe for redirects."""
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        
        # Only allow http and https
        if parsed.scheme not in ["http", "https"]:
            return False
            
        # Check if host is in allowed list
        if parsed.netloc and parsed.netloc not in allowed_hosts:
            return False
            
        return True
    except Exception:
        return False 