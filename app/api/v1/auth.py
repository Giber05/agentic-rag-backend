"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from typing import Optional
from datetime import datetime, timedelta
import secrets

from ...core.config import settings
from ...core.security import (
    create_access_token, 
    create_refresh_token, 
    verify_token, 
    get_password_hash, 
    verify_password,
    generate_api_key,
    create_csrf_token
)
from ...core.dependencies import (
    get_current_user,
    get_current_user_optional,
    security_dependencies,
    get_client_ip,
    AuthenticationError
)
from ...core.logging import get_logger
from ...models.user_models import (
    UserCreate,
    UserLogin,
    UserProfile,
    UserProfileResponse,
    PasswordChange,
    EmailVerificationRequest,
    PasswordResetRequest,
    PasswordReset,
    APIKeyCreate,
    APIKeyResponse,
    APIKeysResponse,
    UserRole,
    UserStatus
)
from ...models.base import BaseResponse
from ...services.user_service import UserService
from ...services.email_service import EmailService
from ...core.oauth import oauth_manager

logger = get_logger(__name__)
router = APIRouter()

# Dependency injection
def get_user_service() -> UserService:
    return UserService()

def get_email_service() -> EmailService:
    return EmailService()

@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_create: UserCreate,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_service: UserService = Depends(get_user_service),
    email_service: EmailService = Depends(get_email_service)
):
    """Register a new user."""
    try:
        # Check if user already exists
        existing_user = await user_service.get_user_by_email(user_create.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        user = await user_service.create_user(user_create)
        
        # Send verification email
        if settings.EMAIL_VERIFICATION_ENABLED:
            verification_token = await user_service.create_email_verification_token(user.id)
            await email_service.send_verification_email(user.email, verification_token)
        
        # Log registration
        logger.info(
            "User registered successfully",
            user_id=user.id,
            email=user.email,
            client_ip=client_ip
        )
        
        return UserProfileResponse(
            success=True,
            message="User registered successfully. Please check your email for verification.",
            data=UserProfile(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                status=user.status,
                is_verified=user.is_verified,
                created_at=user.created_at
            )
        )
        
    except Exception as e:
        logger.error(f"Registration failed: {e}", client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=BaseResponse)
async def login(
    user_login: UserLogin,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_service: UserService = Depends(get_user_service)
):
    """Authenticate user and return tokens."""
    try:
        # Verify credentials
        user = await user_service.authenticate_user(user_login.email, user_login.password)
        if not user:
            logger.warning(
                "Login attempt with invalid credentials",
                email=user_login.email,
                client_ip=client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check user status
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is {user.status.value}"
            )
        
        # Create tokens
        access_token = create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "scopes": [user.role.value]
            }
        )
        
        refresh_token, token_id = create_refresh_token(user.id)
        
        # Create session
        session = await user_service.create_session(
            user_id=user.id,
            refresh_token_id=token_id,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent")
        )
        
        # Update last login
        await user_service.update_last_login(user.id)
        
        # Create CSRF token
        csrf_token = create_csrf_token(user.id)
        
        logger.info(
            "User logged in successfully",
            user_id=user.id,
            email=user.email,
            client_ip=client_ip
        )
        
        return BaseResponse(
            success=True,
            message="Login successful",
            data={
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "csrf_token": csrf_token,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "is_verified": user.is_verified
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}", email=user_login.email, client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=BaseResponse)
async def refresh_token(
    refresh_token: str,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_service: UserService = Depends(get_user_service)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = verify_token(refresh_token, "refresh")
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("user_id")
        token_id = payload.get("token_id")
        
        # Verify session
        session = await user_service.get_session_by_token_id(token_id)
        if not session or not session.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        # Get user
        user = await user_service.get_user_by_id(user_id)
        if not user or user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token = create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "scopes": [user.role.value]
            }
        )
        
        # Update session
        await user_service.update_session_activity(session.id)
        
        logger.info(
            "Token refreshed successfully",
            user_id=user.id,
            client_ip=client_ip
        )
        
        return BaseResponse(
            success=True,
            message="Token refreshed successfully",
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}", client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout", response_model=BaseResponse)
async def logout(
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Logout user and invalidate session."""
    try:
        # Invalidate all user sessions
        await user_service.invalidate_user_sessions(current_user.user_id)
        
        logger.info(
            "User logged out successfully",
            user_id=current_user.user_id
        )
        
        return BaseResponse(
            success=True,
            message="Logged out successfully"
        )
        
    except Exception as e:
        logger.error(f"Logout failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Get current user profile."""
    try:
        user = await user_service.get_user_by_id(current_user.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserProfileResponse(
            success=True,
            data=UserProfile(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                status=user.status,
                is_verified=user.is_verified,
                metadata=user.metadata,
                created_at=user.created_at,
                last_login_at=user.last_login_at
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

@router.post("/change-password", response_model=BaseResponse)
async def change_password(
    password_change: PasswordChange,
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Change user password."""
    try:
        # Verify current password
        user = await user_service.get_user_by_id(current_user.user_id)
        if not user or not verify_password(password_change.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        await user_service.update_password(user.id, password_change.new_password)
        
        # Invalidate all sessions except current one
        await user_service.invalidate_other_sessions(user.id, exclude_current=True)
        
        logger.info(
            "Password changed successfully",
            user_id=user.id
        )
        
        return BaseResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.post("/verify-email", response_model=BaseResponse)
async def verify_email(
    token: str,
    user_service: UserService = Depends(get_user_service)
):
    """Verify user email with token."""
    try:
        success = await user_service.verify_email(token)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        return BaseResponse(
            success=True,
            message="Email verified successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed"
        )

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_create: APIKeyCreate,
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Create a new API key."""
    try:
        api_key, api_key_hash = generate_api_key()
        
        api_key_obj = await user_service.create_api_key(
            user_id=current_user.user_id,
            name=api_key_create.name,
            description=api_key_create.description,
            key_hash=api_key_hash,
            scopes=api_key_create.scopes,
            expires_at=api_key_create.expires_at
        )
        
        logger.info(
            "API key created",
            user_id=current_user.user_id,
            api_key_id=api_key_obj.id
        )
        
        return APIKeyResponse(
            success=True,
            message="API key created successfully",
            data=api_key_obj,
            key=api_key  # Only returned on creation
        )
        
    except Exception as e:
        logger.error(f"API key creation failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key creation failed"
        )

@router.get("/api-keys", response_model=APIKeysResponse)
async def list_api_keys(
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """List user's API keys."""
    try:
        api_keys = await user_service.get_user_api_keys(current_user.user_id)
        
        return APIKeysResponse(
            success=True,
            data=api_keys
        )
        
    except Exception as e:
        logger.error(f"List API keys failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )

@router.delete("/api-keys/{api_key_id}", response_model=BaseResponse)
async def delete_api_key(
    api_key_id: str,
    current_user = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    """Delete an API key."""
    try:
        success = await user_service.delete_api_key(current_user.user_id, api_key_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        logger.info(
            "API key deleted",
            user_id=current_user.user_id,
            api_key_id=api_key_id
        )
        
        return BaseResponse(
            success=True,
            message="API key deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key deletion failed: {e}", user_id=current_user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key deletion failed"
        )

# OAuth Endpoints

@router.get("/oauth/providers", response_model=BaseResponse)
async def get_oauth_providers():
    """Get available OAuth providers."""
    providers = oauth_manager.get_available_providers()
    return BaseResponse(
        success=True,
        message="OAuth providers retrieved successfully",
        data={"providers": providers}
    )

@router.get("/oauth/{provider}/authorize", response_model=BaseResponse)
async def oauth_authorize(provider: str):
    """Get OAuth authorization URL for the specified provider."""
    try:
        oauth_provider = oauth_manager.get_provider(provider)
        if not oauth_provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"OAuth provider '{provider}' not found or not configured"
            )
        
        state = oauth_manager.generate_state()
        authorization_url = oauth_provider.get_authorization_url(state)
        
        return BaseResponse(
            success=True,
            message="Authorization URL generated successfully",
            data={
                "authorization_url": authorization_url,
                "state": state,
                "provider": provider
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth authorization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth authorization failed"
        )

@router.post("/oauth/{provider}/callback", response_model=BaseResponse)
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_service: UserService = Depends(get_user_service)
):
    """Handle OAuth callback and create/login user."""
    try:
        oauth_provider = oauth_manager.get_provider(provider)
        if not oauth_provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"OAuth provider '{provider}' not found"
            )
        
        # Exchange code for token
        token_data = await oauth_provider.exchange_code_for_token(code, state)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange code for token"
            )
        
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No access token received"
            )
        
        # Get user information
        user_info = await oauth_provider.get_user_info(access_token)
        if not user_info or not user_info.get("email"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user information or email not available"
            )
        
        # Check if user exists
        existing_user = await user_service.get_user_by_email(user_info["email"])
        
        if existing_user:
            # Login existing user
            user = existing_user
            
            # Update user info if needed
            if user_info.get("name") and user_info["name"] != user.full_name:
                await user_service.update_user_profile(
                    user.id, 
                    {"full_name": user_info["name"]}
                )
                user.full_name = user_info["name"]
        else:
            # Create new user
            user_create = UserCreate(
                email=user_info["email"],
                full_name=user_info.get("name", ""),
                password="oauth_user",  # Placeholder password for OAuth users
                role=UserRole.USER
            )
            
            user = await user_service.create_user(user_create)
            
            # Mark as verified since OAuth provider verified the email
            await user_service.verify_email_token(user.id)
            user.is_verified = True
        
        # Check user status
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is {user.status.value}"
            )
        
        # Create tokens
        access_token_jwt = create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "scopes": [user.role.value]
            }
        )
        
        refresh_token, token_id = create_refresh_token(user.id)
        
        # Create session
        session = await user_service.create_session(
            user_id=user.id,
            refresh_token_id=token_id,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent")
        )
        
        # Update last login
        await user_service.update_last_login(user.id)
        
        # Create CSRF token
        csrf_token = create_csrf_token(user.id)
        
        logger.info(
            "OAuth login successful",
            user_id=user.id,
            email=user.email,
            provider=provider,
            client_ip=client_ip
        )
        
        return BaseResponse(
            success=True,
            message=f"OAuth login successful via {provider}",
            data={
                "access_token": access_token_jwt,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "csrf_token": csrf_token,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "is_verified": user.is_verified
                },
                "provider": provider
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}", provider=provider, client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth login failed"
        ) 