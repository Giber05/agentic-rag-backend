"""
User service for authentication and user management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import secrets
import hashlib

from ..core.config import settings
from ..core.security import get_password_hash, verify_password, TokenData
from ..core.logging import get_logger
from ..models.user_models import (
    User, UserCreate, UserUpdate, UserRole, UserStatus,
    APIKey, UserSession, AuditLog, AuditAction
)
from supabase import create_client, Client

logger = get_logger(__name__)

class UserService:
    """Service for user management and authentication."""
    
    def __init__(self):
        self.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    async def create_user(self, user_create: UserCreate) -> User:
        """Create a new user."""
        try:
            user_id = str(uuid.uuid4())
            password_hash = get_password_hash(user_create.password)
            now = datetime.utcnow()
            
            user_data = {
                "id": user_id,
                "email": user_create.email,
                "full_name": user_create.full_name,
                "password_hash": password_hash,
                "role": user_create.role.value,
                "status": UserStatus.PENDING.value if settings.EMAIL_VERIFICATION_ENABLED else UserStatus.ACTIVE.value,
                "is_verified": not settings.EMAIL_VERIFICATION_ENABLED,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat()
            }
            
            # Insert user into database
            result = self.supabase.table("users").insert(user_data).execute()
            
            if not result.data:
                raise Exception("Failed to create user")
            
            user_dict = result.data[0]
            
            # Create audit log
            await self._create_audit_log(
                user_id=user_id,
                action=AuditAction.USER_CREATE,
                metadata={"email": user_create.email}
            )
            
            logger.info(f"User created successfully: {user_create.email}")
            
            return User(
                id=user_dict["id"],
                email=user_dict["email"],
                full_name=user_dict.get("full_name"),
                role=UserRole(user_dict["role"]),
                status=UserStatus(user_dict["status"]),
                is_verified=user_dict["is_verified"],
                created_at=datetime.fromisoformat(user_dict["created_at"]),
                updated_at=datetime.fromisoformat(user_dict["updated_at"])
            )
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        try:
            result = self.supabase.table("users").select("*").eq("email", email).execute()
            
            if not result.data:
                return None
            
            user_dict = result.data[0]
            return self._dict_to_user(user_dict)
            
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            result = self.supabase.table("users").select("*").eq("id", user_id).execute()
            
            if not result.data:
                return None
            
            user_dict = result.data[0]
            return self._dict_to_user(user_dict)
            
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            user = await self.get_user_by_email(email)
            if not user:
                return None
            
            # Get password hash from database
            result = self.supabase.table("users").select("password_hash").eq("id", user.id).execute()
            if not result.data:
                return None
            
            password_hash = result.data[0]["password_hash"]
            
            if not verify_password(password, password_hash):
                await self._create_audit_log(
                    user_id=user.id,
                    action=AuditAction.LOGIN,
                    metadata={"success": False, "reason": "invalid_password"}
                )
                return None
            
            await self._create_audit_log(
                user_id=user.id,
                action=AuditAction.LOGIN,
                metadata={"success": True}
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return None
    
    async def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[User]:
        """Update user information."""
        try:
            update_data = {}
            if user_update.full_name is not None:
                update_data["full_name"] = user_update.full_name
            if user_update.role is not None:
                update_data["role"] = user_update.role.value
            if user_update.status is not None:
                update_data["status"] = user_update.status.value
            if user_update.metadata is not None:
                update_data["metadata"] = user_update.metadata
            
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.supabase.table("users").update(update_data).eq("id", user_id).execute()
            
            if not result.data:
                return None
            
            await self._create_audit_log(
                user_id=user_id,
                action=AuditAction.PROFILE_UPDATE,
                metadata={"fields_updated": list(update_data.keys())}
            )
            
            return self._dict_to_user(result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            return None
    
    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Update user password."""
        try:
            password_hash = get_password_hash(new_password)
            
            result = self.supabase.table("users").update({
                "password_hash": password_hash,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", user_id).execute()
            
            if result.data:
                await self._create_audit_log(
                    user_id=user_id,
                    action=AuditAction.PASSWORD_CHANGE,
                    metadata={"success": True}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update password: {e}")
            return False
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        try:
            result = self.supabase.table("users").update({
                "last_login_at": datetime.utcnow().isoformat()
            }).eq("id", user_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            return False
    
    async def create_session(
        self, 
        user_id: str, 
        refresh_token_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create a new user session."""
        try:
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            expires_at = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
            
            session_data = {
                "id": session_id,
                "user_id": user_id,
                "refresh_token_id": refresh_token_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "is_active": True
            }
            
            result = self.supabase.table("user_sessions").insert(session_data).execute()
            
            if not result.data:
                raise Exception("Failed to create session")
            
            session_dict = result.data[0]
            
            return UserSession(
                id=session_dict["id"],
                user_id=session_dict["user_id"],
                refresh_token_id=session_dict["refresh_token_id"],
                ip_address=session_dict.get("ip_address"),
                user_agent=session_dict.get("user_agent"),
                created_at=datetime.fromisoformat(session_dict["created_at"]),
                expires_at=datetime.fromisoformat(session_dict["expires_at"]),
                is_active=session_dict["is_active"]
            )
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session_by_token_id(self, token_id: str) -> Optional[UserSession]:
        """Get session by refresh token ID."""
        try:
            result = self.supabase.table("user_sessions").select("*").eq("refresh_token_id", token_id).execute()
            
            if not result.data:
                return None
            
            session_dict = result.data[0]
            return UserSession(
                id=session_dict["id"],
                user_id=session_dict["user_id"],
                refresh_token_id=session_dict["refresh_token_id"],
                ip_address=session_dict.get("ip_address"),
                user_agent=session_dict.get("user_agent"),
                created_at=datetime.fromisoformat(session_dict["created_at"]),
                expires_at=datetime.fromisoformat(session_dict["expires_at"]),
                is_active=session_dict["is_active"]
            )
            
        except Exception as e:
            logger.error(f"Failed to get session by token ID: {e}")
            return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        try:
            result = self.supabase.table("user_sessions").update({
                "last_used_at": datetime.utcnow().isoformat()
            }).eq("id", session_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            return False
    
    async def invalidate_user_sessions(self, user_id: str) -> bool:
        """Invalidate all user sessions."""
        try:
            result = self.supabase.table("user_sessions").update({
                "is_active": False
            }).eq("user_id", user_id).execute()
            
            await self._create_audit_log(
                user_id=user_id,
                action=AuditAction.LOGOUT,
                metadata={"all_sessions": True}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate user sessions: {e}")
            return False
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        key_hash: str = "",
        scopes: List[str] = None,
        expires_at: Optional[datetime] = None
    ) -> APIKey:
        """Create a new API key for user."""
        try:
            api_key_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            api_key_data = {
                "id": api_key_id,
                "name": name,
                "description": description,
                "key_hash": key_hash,
                "scopes": scopes or [],
                "user_id": user_id,
                "is_active": True,
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None
            }
            
            result = self.supabase.table("api_keys").insert(api_key_data).execute()
            
            if not result.data:
                raise Exception("Failed to create API key")
            
            api_key_dict = result.data[0]
            
            await self._create_audit_log(
                user_id=user_id,
                action=AuditAction.API_KEY_CREATE,
                metadata={"api_key_id": api_key_id, "name": name}
            )
            
            return APIKey(
                id=api_key_dict["id"],
                name=api_key_dict["name"],
                description=api_key_dict.get("description"),
                key_hash=api_key_dict["key_hash"],
                scopes=api_key_dict["scopes"],
                user_id=api_key_dict["user_id"],
                is_active=api_key_dict["is_active"],
                created_at=datetime.fromisoformat(api_key_dict["created_at"]),
                expires_at=datetime.fromisoformat(api_key_dict["expires_at"]) if api_key_dict.get("expires_at") else None
            )
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        try:
            result = self.supabase.table("api_keys").select("*").eq("user_id", user_id).execute()
            
            api_keys = []
            for api_key_dict in result.data:
                api_keys.append(APIKey(
                    id=api_key_dict["id"],
                    name=api_key_dict["name"],
                    description=api_key_dict.get("description"),
                    key_hash=api_key_dict["key_hash"],
                    scopes=api_key_dict["scopes"],
                    user_id=api_key_dict["user_id"],
                    is_active=api_key_dict["is_active"],
                    created_at=datetime.fromisoformat(api_key_dict["created_at"]),
                    expires_at=datetime.fromisoformat(api_key_dict["expires_at"]) if api_key_dict.get("expires_at") else None,
                    last_used_at=datetime.fromisoformat(api_key_dict["last_used_at"]) if api_key_dict.get("last_used_at") else None
                ))
            
            return api_keys
            
        except Exception as e:
            logger.error(f"Failed to get user API keys: {e}")
            return []
    
    async def delete_api_key(self, user_id: str, api_key_id: str) -> bool:
        """Delete an API key."""
        try:
            result = self.supabase.table("api_keys").delete().eq("id", api_key_id).eq("user_id", user_id).execute()
            
            if result.data:
                await self._create_audit_log(
                    user_id=user_id,
                    action=AuditAction.API_KEY_DELETE,
                    metadata={"api_key_id": api_key_id}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return False
    
    async def create_email_verification_token(self, user_id: str) -> str:
        """Create email verification token."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        # Store token in database
        token_data = {
            "token": token,
            "user_id": user_id,
            "type": "email_verification",
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.supabase.table("verification_tokens").insert(token_data).execute()
        return token
    
    async def verify_email(self, token: str) -> bool:
        """Verify email with token."""
        try:
            # Get token from database
            result = self.supabase.table("verification_tokens").select("*").eq("token", token).execute()
            
            if not result.data:
                return False
            
            token_data = result.data[0]
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            
            if datetime.utcnow() > expires_at:
                return False
            
            # Update user as verified
            user_result = self.supabase.table("users").update({
                "is_verified": True,
                "status": UserStatus.ACTIVE.value,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", token_data["user_id"]).execute()
            
            # Delete token
            self.supabase.table("verification_tokens").delete().eq("token", token).execute()
            
            if user_result.data:
                await self._create_audit_log(
                    user_id=token_data["user_id"],
                    action=AuditAction.EMAIL_VERIFICATION,
                    metadata={"success": True}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to verify email: {e}")
            return False
    
    def _dict_to_user(self, user_dict: Dict[str, Any]) -> User:
        """Convert dictionary to User model."""
        return User(
            id=user_dict["id"],
            email=user_dict["email"],
            full_name=user_dict.get("full_name"),
            role=UserRole(user_dict["role"]),
            status=UserStatus(user_dict["status"]),
            is_verified=user_dict["is_verified"],
            metadata=user_dict.get("metadata"),
            created_at=datetime.fromisoformat(user_dict["created_at"]),
            updated_at=datetime.fromisoformat(user_dict["updated_at"]),
            last_login_at=datetime.fromisoformat(user_dict["last_login_at"]) if user_dict.get("last_login_at") else None
        )
    
    async def _create_audit_log(
        self,
        user_id: Optional[str],
        action: AuditAction,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create audit log entry."""
        try:
            audit_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "action": action.value,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("audit_logs").insert(audit_data).execute()
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            # Don't raise exception for audit log failures 