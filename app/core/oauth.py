"""
OAuth authentication providers for Google and GitHub.
"""

import asyncio
import secrets
from typing import Optional, Dict, Any
import httpx
from urllib.parse import urlencode

from .config import settings
from .security import get_password_hash
from .logging import get_logger

logger = get_logger(__name__)

class OAuthProvider:
    """Base OAuth provider class."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    def get_authorization_url(self, state: str) -> str:
        """Get OAuth authorization URL."""
        raise NotImplementedError
    
    async def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        raise NotImplementedError
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth provider."""
        raise NotImplementedError

class GoogleOAuth(OAuthProvider):
    """Google OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.scope = "openid email profile"
    
    def get_authorization_url(self, state: str) -> str:
        """Get Google OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            "state": state,
            "access_type": "offline"
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": self.redirect_uri,
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from Google."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.user_info_url,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    return {
                        "id": user_data.get("id"),
                        "email": user_data.get("email"),
                        "name": user_data.get("name"),
                        "picture": user_data.get("picture"),
                        "verified_email": user_data.get("verified_email", False),
                        "provider": "google"
                    }
                else:
                    logger.error(f"User info fetch failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching user info: {e}")
            return None

class GitHubOAuth(OAuthProvider):
    """GitHub OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.user_info_url = "https://api.github.com/user"
        self.user_emails_url = "https://api.github.com/user/emails"
        self.scope = "user:email"
    
    def get_authorization_url(self, state: str) -> str:
        """Get GitHub OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "state": state,
            "allow_signup": "true"
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str, state: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                    },
                    headers={"Accept": "application/json"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {e}")
            return None
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from GitHub."""
        try:
            async with httpx.AsyncClient() as client:
                # Get user profile
                user_response = await client.get(
                    self.user_info_url,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if user_response.status_code != 200:
                    logger.error(f"User info fetch failed: {user_response.status_code}")
                    return None
                
                user_data = user_response.json()
                
                # Get user emails
                emails_response = await client.get(
                    self.user_emails_url,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                emails_data = []
                if emails_response.status_code == 200:
                    emails_data = emails_response.json()
                
                # Find primary/verified email
                primary_email = None
                for email in emails_data:
                    if email.get("primary") and email.get("verified"):
                        primary_email = email.get("email")
                        break
                
                if not primary_email and user_data.get("email"):
                    primary_email = user_data.get("email")
                
                return {
                    "id": str(user_data.get("id")),
                    "email": primary_email,
                    "name": user_data.get("name") or user_data.get("login"),
                    "username": user_data.get("login"),
                    "avatar_url": user_data.get("avatar_url"),
                    "provider": "github"
                }
                    
        except Exception as e:
            logger.error(f"Error fetching user info: {e}")
            return None

class OAuthManager:
    """OAuth manager for handling multiple providers."""
    
    def __init__(self):
        self.providers = {}
        self._init_providers()
    
    def _init_providers(self):
        """Initialize OAuth providers based on configuration."""
        # Google OAuth
        if hasattr(settings, 'GOOGLE_CLIENT_ID') and settings.GOOGLE_CLIENT_ID:
            self.providers['google'] = GoogleOAuth(
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET,
                redirect_uri=f"{settings.FRONTEND_URL}/auth/google/callback"
            )
        
        # GitHub OAuth
        if hasattr(settings, 'GITHUB_CLIENT_ID') and settings.GITHUB_CLIENT_ID:
            self.providers['github'] = GitHubOAuth(
                client_id=settings.GITHUB_CLIENT_ID,
                client_secret=settings.GITHUB_CLIENT_SECRET,
                redirect_uri=f"{settings.FRONTEND_URL}/auth/github/callback"
            )
    
    def get_provider(self, provider_name: str) -> Optional[OAuthProvider]:
        """Get OAuth provider by name."""
        return self.providers.get(provider_name)
    
    def get_available_providers(self) -> list[str]:
        """Get list of available OAuth providers."""
        return list(self.providers.keys())
    
    def generate_state(self) -> str:
        """Generate secure state parameter for OAuth."""
        return secrets.token_urlsafe(32)

# Global OAuth manager instance
oauth_manager = OAuthManager() 