from llm.providers.base import BaseLLMProvider
from llm.providers.ollama import OllamaProvider

class LLMManager:
    """
    Reads config and returns the right provider.
    Handles fallback automatically.
    Everything else in the system imports from here.
    Never imports a specific provider directly.
    """

    def __init__(self):
        self._provider: BaseLLMProvider = self._load_provider()

    def _load_provider(self) -> BaseLLMProvider:
        # Try Ollama first (local)
        ollama = OllamaProvider(model="qwen2.5:3b")

        if ollama.is_available():
            print("✓ Using Ollama (local)")
            return ollama
        
        # Fallback to OpenAI API 
        print("⚠ Ollama not available. Falling back to OpenAI API")
        try:
            from llm.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(model="gpt-4o")
        except Exception as e:
            raise RuntimeError(
                "No LLM provider available."
                "Start Ollama with: ollama serve"
            ) from e
        
    def generate(self, prompt: str, system: str = "") -> str:
        return self._provider.generate(prompt, system)
    
    def stream(self, prompt: str, system: str = ""):
        return self._provider.stream(prompt, system)
    
    def embed(self, text: str):
        return self._provider.embed(text)
    
    @property
    def provider_name(self) -> str:
        return self._provider.__class__.__name__
    

# Globally accessable

llm = LLMManager()