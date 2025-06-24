#!/usr/bin/env python3
"""
Simple test script to check document endpoints
"""

import requests
import json

def test_endpoints():
    """Test various document endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/health",
        "/api/v1/status", 
        "/api/v1/documents/formats/supported",
        "/api/v1/documents/stats/overview"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\n🔍 Testing: {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   Response: {json.dumps(data, indent=2)}")
                except:
                    print(f"   Response: {response.text}")
            else:
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   Connection Error: {e}")
        except Exception as e:
            print(f"   Unexpected Error: {e}")

if __name__ == "__main__":
    test_endpoints() 