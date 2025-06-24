"""
Supabase Auth integration for the backend
"""

import os
import jwt
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

from app.core.config import settings

# Initialize Supabase client
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_ROLE_KEY
)

# HTTP Bearer for extracting tokens
security = HTTPBearer()

class SupabaseAuth:
    def __init__(self):
        self.supabase = supabase
        self.jwt_secret = settings.SUPABASE_JWT_SECRET
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify Supabase JWT token"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                audience="authenticated"
            )
            
            # Check if token is expired
            if payload.get('exp', 0) < datetime.utcnow().timestamp():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return payload
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from profiles table"""
        try:
            result = self.supabase.table('profiles').select('*').eq('id', user_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching user profile: {str(e)}"
            )
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user profile in profiles table"""
        try:
            result = self.supabase.table('profiles').insert(user_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating user profile: {str(e)}"
            )
    
    async def update_user_profile(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile in profiles table"""
        try:
            result = self.supabase.table('profiles').update(update_data).eq('id', user_id).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating user profile: {str(e)}"
            )

# Initialize auth instance
supabase_auth = SupabaseAuth()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        # Verify token
        payload = supabase_auth.verify_token(credentials.credentials)
        user_id = payload.get('sub')
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user profile
        profile = await supabase_auth.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return {
            'id': user_id,
            'email': payload.get('email'),  # Get email from JWT payload, not profile
            'full_name': profile.get('full_name'),
            'role': profile.get('role', 'user'),
            'status': profile.get('status', 'active'),
            'is_verified': payload.get('user_metadata', {}).get('email_verified', False),  # Get from JWT payload
            'metadata': profile.get('metadata', {}),
            'created_at': profile.get('created_at'),
            'last_login_at': profile.get('last_login_at')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )

async def get_current_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current user and verify admin role"""
    if current_user.get('role') != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
    """Get current user if token is provided, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None 