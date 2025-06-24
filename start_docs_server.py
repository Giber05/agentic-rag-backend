#!/usr/bin/env python3

"""
Quick script to start the FastAPI server and display documentation URLs.
"""

import uvicorn
import webbrowser
import time
from threading import Timer

def open_docs():
    """Open documentation URLs in browser after server starts."""
    time.sleep(2)  # Wait for server to start
    print("\n🌐 Opening API documentation in browser...")
    webbrowser.open("http://localhost:8000/api/v1/docs")

def main():
    print("🚀 Starting Agentic RAG AI Agent API Server...")
    print("📚 Enhanced API Documentation Available:")
    print("   - Swagger UI: http://localhost:8000/api/v1/docs")
    print("   - ReDoc: http://localhost:8000/api/v1/redoc")
    print("   - OpenAPI JSON: http://localhost:8000/api/v1/openapi.json")
    print("   - Postman Collection: backend/docs/postman_collection.json")
    print("   - Documentation Guide: backend/docs/API_DOCUMENTATION.md")
    print("\n⚡ Server starting on http://localhost:8000")
    print("🔄 Press Ctrl+C to stop the server\n")
    
    # Open docs in browser after 2 seconds
    Timer(2.0, open_docs).start()
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 