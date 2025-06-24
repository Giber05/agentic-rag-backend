"""
Supabase Auth API endpoints
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, validator
from supabase import create_client, Client
import httpx
from datetime import datetime

from app.core.config import settings
from app.core.supabase_auth import get_current_user, get_current_admin_user, supabase_auth
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize Supabase client for auth operations
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY  # Use anon key for auth operations
)

# Request/Response Models
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: Dict[str, Any]
    expires_in: int

@router.post("/register", response_model=Dict[str, Any])
async def register(request: RegisterRequest):
    """Register a new user with Supabase Auth"""
    try:
        # Register user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "full_name": request.full_name or ""
                }
            }
        })
        
        if auth_response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed. Email might already be in use."
            )
        
        user_id = auth_response.user.id
        
        # The profile will be created automatically by the trigger
        # Skip manual profile update for now since the trigger handles it
        logger.info(f"User registered successfully: {user_id}")
        
        return {
            "success": True,
            "message": "User registered successfully. Please check your email for verification.",
            "data": {
                "user_id": user_id,
                "email": request.email,
                "email_confirmed": auth_response.user.email_confirmed_at is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Dict[str, Any])
async def login(request: LoginRequest):
    """Login user with Supabase Auth"""
    try:
        # Login with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if auth_response.user is None or auth_response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user_id = auth_response.user.id
        
        # Update last login time in profile
        try:
            await supabase_auth.update_user_profile(user_id, {
                "last_login_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Could not update last login time: {e}")
        
        # Get user profile
        profile = await supabase_auth.get_user_profile(user_id)
        
        return {
            "success": True,
            "message": "Login successful",
            "data": {
                "access_token": auth_response.session.access_token,
                "refresh_token": auth_response.session.refresh_token,
                "expires_in": auth_response.session.expires_in,
                "user": {
                    "id": user_id,
                    "email": auth_response.user.email,
                    "full_name": profile.get("full_name", "") if profile else "",
                    "role": profile.get("role", "user") if profile else "user",
                    "is_verified": auth_response.user.email_confirmed_at is not None,
                    "created_at": auth_response.user.created_at,
                    "last_login_at": profile.get("last_login_at") if profile else None
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout current user"""
    try:
        # Logout from Supabase Auth
        supabase.auth.sign_out()
        
        return {
            "success": True,
            "message": "Logged out successfully"
        }
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {
            "success": True,
            "message": "Logged out successfully"  # Always return success for logout
        }

@router.post("/refresh")
async def refresh_token(refresh_token: str = Form(...)):
    """Refresh access token"""
    try:
        # Refresh token with Supabase Auth
        auth_response = supabase.auth.refresh_session(refresh_token)
        
        if auth_response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "data": {
                "access_token": auth_response.session.access_token,
                "refresh_token": auth_response.session.refresh_token,
                "expires_in": auth_response.session.expires_in
            }
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not refresh token"
        )

@router.post("/forgot-password")
async def forgot_password(request: PasswordResetRequest):
    """Send password reset email"""
    try:
        # Send password reset email via Supabase Auth
        supabase.auth.reset_password_email(request.email, {
            "redirect_to": f"{settings.FRONTEND_URL}/reset-password"
        })
        
        return {
            "success": True,
            "message": "Password reset email sent if the email exists in our system"
        }
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        # Always return success to prevent email enumeration
        return {
            "success": True,
            "message": "Password reset email sent if the email exists in our system"
        }

@router.get("/me")
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user profile"""
    return {
        "success": True,
        "message": None,
        "data": current_user
    }

@router.put("/profile")
async def update_profile(
    request: UpdateProfileRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user profile"""
    try:
        update_data = {}
        if request.full_name is not None:
            update_data["full_name"] = request.full_name
        if request.metadata is not None:
            update_data["metadata"] = request.metadata
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Update profile
        updated_profile = await supabase_auth.update_user_profile(
            current_user["id"], 
            update_data
        )
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "data": updated_profile
        }
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not update profile: {str(e)}"
        )

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Change user password"""
    try:
        # First verify current password by attempting to sign in
        try:
            supabase.auth.sign_in_with_password({
                "email": current_user["email"],
                "password": request.current_password
            })
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        supabase.auth.update_user({
            "password": request.new_password
        })
        
        return {
            "success": True,
            "message": "Password changed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not change password"
        ) 