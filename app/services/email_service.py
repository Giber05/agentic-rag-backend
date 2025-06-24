"""
Email service for sending verification and notification emails.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from pathlib import Path

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

class EmailService:
    """Service for sending emails."""
    
    def __init__(self):
        self.smtp_server = settings.EMAIL_HOST
        self.smtp_port = settings.EMAIL_PORT
        self.username = settings.EMAIL_USERNAME
        self.password = settings.EMAIL_PASSWORD
        self.use_tls = settings.EMAIL_USE_TLS
    
    async def send_verification_email(self, email: str, token: str) -> bool:
        """Send email verification email."""
        try:
            if not settings.EMAIL_VERIFICATION_ENABLED or not self.smtp_server:
                logger.info("Email verification disabled or not configured")
                return True
            
            subject = "Verify Your Email - Agentic RAG AI Agent"
            verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
            
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #333; margin-bottom: 10px;">Agentic RAG AI Agent</h1>
                        <h2 style="color: #666; font-weight: normal;">Email Verification</h2>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; color: #333; line-height: 1.6;">
                            Thank you for registering with Agentic RAG AI Agent! 
                            Please click the button below to verify your email address and activate your account.
                        </p>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{verification_url}" 
                           style="background: #007bff; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 6px; display: inline-block;
                                  font-weight: bold;">
                            Verify Email Address
                        </a>
                    </div>
                    
                    <div style="background: #e9ecef; padding: 15px; border-radius: 6px; margin-top: 20px;">
                        <p style="margin: 0; color: #666; font-size: 14px;">
                            If the button doesn't work, copy and paste this link into your browser:
                        </p>
                        <p style="margin: 10px 0 0 0; word-break: break-all;">
                            <a href="{verification_url}" style="color: #007bff;">{verification_url}</a>
                        </p>
                    </div>
                    
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                        <p style="margin: 0; color: #999; font-size: 12px; text-align: center;">
                            This verification link will expire in 24 hours.<br>
                            If you didn't create an account, you can safely ignore this email.
                        </p>
                    </div>
                </body>
            </html>
            """
            
            return await self._send_email(
                to_email=email,
                subject=subject,
                html_content=html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send verification email: {e}")
            return False
    
    async def send_password_reset_email(self, email: str, token: str) -> bool:
        """Send password reset email."""
        try:
            if not self.smtp_server:
                logger.info("Email not configured")
                return True
            
            subject = "Reset Your Password - Agentic RAG AI Agent"
            reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
            
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #333; margin-bottom: 10px;">Agentic RAG AI Agent</h1>
                        <h2 style="color: #666; font-weight: normal;">Password Reset</h2>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; color: #333; line-height: 1.6;">
                            We received a request to reset your password. 
                            Click the button below to create a new password for your account.
                        </p>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{reset_url}" 
                           style="background: #dc3545; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 6px; display: inline-block;
                                  font-weight: bold;">
                            Reset Password
                        </a>
                    </div>
                    
                    <div style="background: #e9ecef; padding: 15px; border-radius: 6px; margin-top: 20px;">
                        <p style="margin: 0; color: #666; font-size: 14px;">
                            If the button doesn't work, copy and paste this link into your browser:
                        </p>
                        <p style="margin: 10px 0 0 0; word-break: break-all;">
                            <a href="{reset_url}" style="color: #dc3545;">{reset_url}</a>
                        </p>
                    </div>
                    
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                        <p style="margin: 0; color: #999; font-size: 12px; text-align: center;">
                            This reset link will expire in 1 hour.<br>
                            If you didn't request a password reset, you can safely ignore this email.
                        </p>
                    </div>
                </body>
            </html>
            """
            
            return await self._send_email(
                to_email=email,
                subject=subject,
                html_content=html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send password reset email: {e}")
            return False
    
    async def send_security_alert_email(self, email: str, event: str, details: dict) -> bool:
        """Send security alert email."""
        try:
            if not self.smtp_server:
                return True
            
            subject = f"Security Alert - {event}"
            
            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #333; margin-bottom: 10px;">Agentic RAG AI Agent</h1>
                        <h2 style="color: #dc3545; font-weight: normal;">Security Alert</h2>
                    </div>
                    
                    <div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h3 style="color: #721c24; margin-top: 0;">Security Event Detected</h3>
                        <p style="margin: 0; color: #721c24; line-height: 1.6;">
                            We detected the following security event on your account:
                        </p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin-top: 0; color: #333;">Event Details:</h4>
                        <ul style="color: #333; line-height: 1.6;">
                            <li><strong>Event:</strong> {event}</li>
                            <li><strong>Time:</strong> {details.get('timestamp', 'Unknown')}</li>
                            <li><strong>IP Address:</strong> {details.get('ip_address', 'Unknown')}</li>
                            <li><strong>User Agent:</strong> {details.get('user_agent', 'Unknown')}</li>
                        </ul>
                    </div>
                    
                    <div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 6px; margin-top: 20px;">
                        <p style="margin: 0; color: #155724; font-size: 14px;">
                            <strong>What to do:</strong><br>
                            If this was you, no action is needed. If you don't recognize this activity, 
                            please change your password immediately and contact support.
                        </p>
                    </div>
                    
                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                        <p style="margin: 0; color: #999; font-size: 12px; text-align: center;">
                            This is an automated security notification from Agentic RAG AI Agent.
                        </p>
                    </div>
                </body>
            </html>
            """
            
            return await self._send_email(
                to_email=email,
                subject=subject,
                html_content=html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send security alert email: {e}")
            return False
    
    async def _send_email(
        self, 
        to_email: str, 
        subject: str, 
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """Send email via SMTP."""
        try:
            if not all([self.smtp_server, self.username, self.password]):
                logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.username
            message["To"] = to_email
            
            # Create text and HTML parts
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.username, to_email, message.as_string())
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False 