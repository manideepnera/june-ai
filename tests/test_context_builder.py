import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.context_builder import ContextBuilder


def test_basic_prompt():
    builder = ContextBuilder()
    prompt = builder.build(user_input="What is machine learning?")

    assert "User: What is machine learning?" in prompt
    assert "June:" in prompt
    print("✓ Basic prompt structure correct")


def test_prompt_with_rag():
    builder = ContextBuilder()
    rag_context = (
        "Relevant context from your knowledge base:\n\n"
        "[Source 1: ml_notes.txt]\n"
        "Machine learning is a subset of AI that learns from data."
    )
    prompt = builder.build(
        user_input="What is machine learning?",
        rag_context=rag_context
    )

    assert "ml_notes.txt" in prompt
    assert "Machine learning" in prompt
    assert "User: What is machine learning?" in prompt
    print("✓ RAG context injected correctly")


def test_prompt_with_history():
    builder = ContextBuilder()
    history = [
        {"user": "Hello June", "assistant": "Hello! How can I help?"},
        {"user": "What is AI?", "assistant": "AI is artificial intelligence."}
    ]
    prompt = builder.build(
        user_input="Tell me more",
        history=history
    )

    assert "Hello June" in prompt
    assert "What is AI?" in prompt
    assert "Tell me more" in prompt
    print("✓ History included correctly")


def test_history_trimming():
    """History should be trimmed to max_history_turns."""
    builder = ContextBuilder(max_history_turns=2)
    history = [
        {"user": f"Message {i}", "assistant": f"Response {i}"}
        for i in range(10)
    ]
    prompt = builder.build(user_input="Latest message", history=history)

    # Only last 2 turns should appear
    assert "Message 9" in prompt
    assert "Message 8" in prompt
    assert "Message 0" not in prompt
    print("✓ History correctly trimmed to last 2 turns")


if __name__ == "__main__":
    print("\n--- Testing ContextBuilder ---\n")
    test_basic_prompt()
    test_prompt_with_rag()
    test_prompt_with_history()
    test_history_trimming()
    print("\n--- ContextBuilder tests passed ---\n")