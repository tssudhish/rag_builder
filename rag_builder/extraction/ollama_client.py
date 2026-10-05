import requests
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with a local Ollama instance.
    """
    KNOWN_MODEL_ALIASES = {
        "sciphi-triplex": "sciphi/triplex:latest",
        "sciphi/triplex": "sciphi/triplex:latest",
        "triplex": "sciphi/triplex:latest",
        "gemma2": "gemma2:latest",
        "qwen": "qwen2.5-coder:1.5b",
    }

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        # Normalize base_url: strip trailing slashes and redundant /api/generate or /api
        cleaned = base_url.rstrip('/')
        if cleaned.endswith('/api/generate'):
            cleaned = cleaned[:-len('/api/generate')].rstrip('/')
        elif cleaned.endswith('/api'):
            cleaned = cleaned[:-len('/api')].rstrip('/')
        self.base_url = cleaned
        self.timeout = timeout

    def get_available_models(self) -> List[str]:
        """Queries Ollama for list of installed model names."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            logger.debug(f"Could not retrieve available Ollama models: {e}")
        return []

    def resolve_model(self, model: str) -> str:
        """
        Resolves model name handling aliases and checking available models if needed.
        """
        if not model:
            return model

        # Check known alias dictionary first
        lowered = model.strip().lower()
        if lowered in self.KNOWN_MODEL_ALIASES:
            return self.KNOWN_MODEL_ALIASES[lowered]

        return model

    def generate(self, model: str, prompt: str, format: str = "json", stream: bool = False) -> Optional[str]:
        """
        Sends a prompt to the specified Ollama model and returns the response text.
        
        Args:
            model: The name of the model to use (e.g., 'sciphi/triplex:latest', 'sciphi-triplex').
            prompt: The input text prompt.
            format: Response format, typically 'json' for structured extraction.
            stream: Whether to stream the response (not supported in this simple client).
            
        Returns:
            The response content as a string, or None if the request failed.
        """
        if stream:
            raise NotImplementedError("Streaming is not supported in OllamaClient.generate")

        resolved_model = self.resolve_model(model)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "format": format
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            # If 404, check if model was the issue and attempt fallback or provide clear message
            if response.status_code == 404:
                error_body = response.text
                available = self.get_available_models()
                
                # Try fallback matching if model not found
                matched_model = None
                normalized_target = resolved_model.replace('-', '/').replace(':latest', '')
                for avail in available:
                    avail_norm = avail.replace('-', '/').replace(':latest', '')
                    if normalized_target in avail_norm or avail_norm in normalized_target:
                        matched_model = avail
                        break

                if matched_model and matched_model != resolved_model:
                    logger.info(f"Model '{resolved_model}' not found; retrying with matching installed model '{matched_model}'")
                    payload["model"] = matched_model
                    retry_resp = requests.post(url, json=payload, timeout=self.timeout)
                    if retry_resp.status_code == 200:
                        return retry_resp.json().get("response")

                logger.error(
                    f"Ollama API request failed: 404 Client Error for url: {url} - "
                    f"Details: {error_body}. Installed models: {available}. "
                    f"Run 'ollama pull {resolved_model}' to install."
                )
                return None

            response.raise_for_status()
            data = response.json()
            return data.get("response")
        except requests.exceptions.RequestException as e:
            error_details = ""
            if hasattr(e, "response") and e.response is not None:
                error_details = f" - Response: {e.response.text}"
            logger.error(f"Ollama API request failed: {e}{error_details}")
            return None
