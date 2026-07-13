"""
LLM Client Module
Handles communication with the LLM API for prescription text extraction.
Adapted from chatbot-app for use in prescription OCR pipeline.
"""

import os
import time
import requests
from typing import Optional
from pathlib import Path
import sys

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLM


class LLMClient:
    """
    Client for interacting with the LLM API.
    Uses the same API as the chatbot-app.
    """
    
    def __init__(self):
        """Initialize the LLM client with config settings."""
        self.api_base_url = LLM.get("api_base_url", "https://llm.jetstream-cloud.org/api/")
        self.api_key = LLM.get("api_key") or os.getenv("API_KEY")
        self.model = LLM.get("model", "gpt-oss-120b")
        self.max_tokens = LLM.get("max_tokens", 500)
        self.temperature = LLM.get("temperature", 0.3)
        self.available = False
        
        self._check_availability()
    
    def _check_availability(self):
        """Check if the LLM API is accessible."""
        if not self.api_key:
            print("[LLM] Warning: No API_KEY found. Set API_KEY environment variable.")
            return
            
        try:
            response = requests.get(
                f"{self.api_base_url}models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            if response.status_code == 200:
                self.available = True
                print("[LLM] ✓ API is accessible")
            else:
                print(f"[LLM] ✗ API error: {response.status_code}")
        except Exception as e:
            print(f"[LLM] ✗ Could not connect: {e}")
    
    def _make_request_with_retry(
        self, 
        messages: list, 
        max_retries: int = 3,
        initial_wait: float = 1.0
    ) -> Optional[requests.Response]:
        """Make API request with exponential backoff retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base_url}chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        if message.get("content"):
                            return response
                    
                    # Empty response, retry
                    if attempt < max_retries - 1:
                        wait_time = initial_wait * (2 ** attempt)
                        print(f"[LLM] Empty response, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                return response
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (2 ** attempt)
                    print(f"[LLM] Request failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        return None
    
    def get_completion(
        self, 
        prompt: str, 
        system_message: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a completion from the LLM.
        
        Args:
            prompt: User prompt
            system_message: Optional system message for context
            
        Returns:
            LLM response text or None if failed
        """
        if not self.available:
            return None
        
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._make_request_with_retry(messages)
            
            if response and response.status_code == 200:
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    return content.strip()
            
            return None
            
        except Exception as e:
            print(f"[LLM] Error: {e}")
            return None


# Global instance for lazy initialization
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def is_llm_available() -> bool:
    """Check if LLM is available."""
    return get_llm_client().available


if __name__ == "__main__":
    # Test the LLM client
    client = LLMClient()
    
    if client.available:
        response = client.get_completion(
            "What is Panadol used for? Answer in one sentence.",
            "You are a helpful medical assistant."
        )
        print(f"Response: {response}")
    else:
        print("LLM not available for testing")
