import requests
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    """
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def generate(self, model: str, prompt: str, format: str = "json", stream: bool = False) -> Optional[str]:
        """
        Sends a prompt to the specified Ollama model and returns the response text.
        
        Args:
            model: The name of the model to use (e.g., 'sciphi-triplex', 'gemma').
            prompt: The input text prompt.
            format: Response format, typically 'json' for structured extraction.
            stream: Whether to stream the response (not supported in this simple client).
            
        Returns:
            The response content as a string, or None if the request failed.
        """
        if stream:
            raise NotImplementedError("Streaming is not supported in OllamaClient.generate")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": format
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return None
