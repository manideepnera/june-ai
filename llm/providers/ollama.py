import json
import requests
from typing import Iterator, List
from llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Talks to the local Ollama API.
    Runs on http://localhost:11434 by default.
    """

    def __init__(self, model: str,  base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        """Send prompt, wait for full response."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
                "Is Ollama running? Try: ollama serve"
            )
        
    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Send prompt, yield response chunks as they arrive."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True
        }
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            )
        
    
    def embed(self, text: str) -> List[float]:
        """Generate emeddings vector for text."""

        payload = {
            "model": self.model,
            "prompt": text
        }
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def is_available(self) -> bool:
        """Check if Ollama is running and reachable"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False
