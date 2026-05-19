import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers.ollama import OllamaProvider


def test_ollama_available():
    provider = OllamaProvider(model="qwen2.5:3b")
    assert provider.is_available(), (
        "Ollama is not running. Start it with: ollama serve"
    )
    print("✓ Ollama is available")


def test_generate():
    provider = OllamaProvider(model="qwen2.5:3b")
    response = provider.generate(
        prompt="Reply with exactly: JUNE_TEST_OK",
        system="You are a test assistant. Follow instructions exactly."
    )
    assert len(response) > 0, "Response was empty"
    print(f"✓ Generate works. Response: {response.strip()}")


def test_stream():
    provider = OllamaProvider(model="qwen2.5:3b")
    chunks = list(provider.stream(
        prompt="Count from 1 to 5, one number per word.",
        system=""
    ))
    full_response = "".join(chunks)
    assert len(chunks) > 1, "Streaming returned only one chunk"
    assert len(full_response) > 0, "Streamed response was empty"
    print(f"✓ Stream works. Chunks: {len(chunks)}, Response: {full_response.strip()}")


def test_embed():
    provider = OllamaProvider(model="qwen2.5:3b")
    vector = provider.embed("Hello world")
    assert isinstance(vector, list), "Embedding should be a list"
    assert len(vector) > 0, "Embedding vector was empty"
    print(f"✓ Embed works. Vector dimensions: {len(vector)}")


if __name__ == "__main__":
    print("\n--- Running LLM Tests ---\n")
    test_ollama_available()
    test_generate()
    test_stream()
    test_embed()
    print("\n--- All tests passed ---\n")