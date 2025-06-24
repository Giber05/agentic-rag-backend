"""
User models for authentication and authorization.
"""

from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from .base import BaseResponse

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"
    READONLY = "readonly"

class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class UserCreate(BaseModel):
    """User creation request model."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    """User update request model."""
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    metadata: Optional[Dict[str, Any]] = None

class PasswordChange(BaseModel):
    """Password change request model."""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str

class User(BaseModel):
    """User model."""
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole
    status: UserStatus
    is_verified: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None

class UserProfile(BaseModel):
    """User profile model (without sensitive data)."""
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole
    status: UserStatus
    is_verified: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None

class UserResponse(BaseResponse):
    """User response model."""
    data: User

class UserProfileResponse(BaseResponse):
    """User profile response model."""
    data: UserProfile

class UsersResponse(BaseResponse):
    """Multiple users response model."""
    data: List[User]
    total: int
    page: int
    per_page: int

# API Key models
class APIKeyCreate(BaseModel):
    """API key creation request model."""
    name: str
    description: Optional[str] = None
    scopes: List[str] = []
    expires_at: Optional[datetime] = None

class APIKey(BaseModel):
    """API key model."""
    id: str
    name: str
    description: Optional[str] = None
    key_hash: str
    scopes: List[str] = []
    user_id: str
    is_active: bool = True
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

class APIKeyResponse(BaseResponse):
    """API key response model."""
    data: APIKey
    key: Optional[str] = None  # Only returned on creation

class APIKeysResponse(BaseResponse):
    """Multiple API keys response model."""
    data: List[APIKey]

# Session models
class UserSession(BaseModel):
    """User session model."""
    id: str
    user_id: str
    refresh_token_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

class UserSessionResponse(BaseResponse):
    """User session response model."""
    data: UserSession

class UserSessionsResponse(BaseResponse):
    """Multiple user sessions response model."""
    data: List[UserSession]

# Email verification models
class EmailVerificationRequest(BaseModel):
    """Email verification request model."""
    email: EmailStr

class EmailVerification(BaseModel):
    """Email verification model."""
    token: str

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr

class PasswordReset(BaseModel):
    """Password reset model."""
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

# Audit log models
class AuditAction(str, Enum):
    """Audit action enumeration."""
    LOGIN = "login"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    PROFILE_UPDATE = "profile_update"
    API_KEY_CREATE = "api_key_create"
    API_KEY_DELETE = "api_key_delete"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"

class AuditLog(BaseModel):
    """Audit log model."""
    id: str
    user_id: Optional[str] = None
    action: AuditAction
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

class AuditLogResponse(BaseResponse):
    """Audit log response model."""
    data: AuditLog

class AuditLogsResponse(BaseResponse):
    """Multiple audit logs response model."""
    data: List[AuditLog]
    total: int
    page: int
    per_page: int 