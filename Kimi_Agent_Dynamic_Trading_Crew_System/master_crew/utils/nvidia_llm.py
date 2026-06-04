"""
NVIDIA NIM API LLM Wrapper for CrewAI
Uses NVIDIA's free tier API for powerful inference
"""
import os
import requests
import json
from typing import Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential


class NVIDIA_LLM:
    """
    Custom LLM wrapper for NVIDIA NIM API.
    Compatible with CrewAI's expected LLM interface.
    """
    
    def __init__(
        self,
        model: str = "meta/llama-3.3-70b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("NVIDIA API Key is required. Set NVIDIA_API_KEY environment variable.")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call(self, prompt: str, **kwargs) -> str:
        """Make API call to NVIDIA NIM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst. Provide detailed, data-driven analysis with specific numbers and actionable insights. Always be precise and quantitative in your recommendations."},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            print(f"NVIDIA API Error: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow the LLM to be called directly"""
        return self.call(prompt, **kwargs)


class NVIDIA_CrewAI_LLM:
    """
    CrewAI-compatible LLM wrapper.
    CrewAI expects an object with a 'call' method.
    """
    
    def __init__(
        self,
        model: str = "meta/llama-3.3-70b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.llm = NVIDIA_LLM(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def call(self, prompt: str, **kwargs) -> str:
        """CrewAI calls this method"""
        return self.llm.call(prompt, **kwargs)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.call(prompt, **kwargs)


def get_llm(temperature: float = 0.2, max_tokens: int = 4096) -> NVIDIA_CrewAI_LLM:
    """Factory function to create NVIDIA LLM instance"""
    # Try different models in order of preference
    models = [
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
    ]
    
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not found. "
            "Get your free API key from: https://build.nvidia.com/explore/discover"
        )
    
    return NVIDIA_CrewAI_LLM(
        model=models[0],
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )