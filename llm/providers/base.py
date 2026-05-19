from abc import ABC, abstractmethod
from typing import Iterator, List

class BaseLLMProvider(ABC):
    """
    Every LLM provider must implement this interface.
    The orchestrator only knows about this class.
    It never imports OllamaProvider or OpenAIProvider directly.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """
        Single-shot generation.
        Send prompt, get full response string back.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """
        Streaming generation.
        Yields response chunks as they arrive from the model.
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a piece of text.
        Used by RAG for similarity search.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is reachable right now.
        Used by manager to decide which provider to use.
        """
        pass