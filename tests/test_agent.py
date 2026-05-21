import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.agent import Agent


def test_basic_chat():
    agent = Agent()
    response = agent.chat("Hello June")

    assert len(response) > 0
    print(f"✓ Basic chat works.")
    print(f"  Response: {response[:80]}...")


def test_rag_chat():
    agent = Agent()
    response = agent.chat("What is RAG and how does it work?")

    assert len(response) > 50
    print(f"✓ RAG chat works.")
    print(f"  Response: {response[:120]}...")


def test_streaming():
    agent = Agent()
    chunks = list(agent.chat_stream("Explain neural networks briefly"))

    assert len(chunks) > 1, "Should stream multiple chunks"
    full = "".join(chunks)
    assert len(full) > 50
    print(f"✓ Streaming works. Chunks received: {len(chunks)}")


def test_history_maintained():
    agent = Agent()

    agent.chat("My name is Alex")
    response = agent.chat("What is my name?")

    assert agent.history_length == 2
    print(f"✓ History maintained. Turns: {agent.history_length}")
    print(f"  Response: {response[:80]}...")


def test_history_reset():
    agent = Agent()
    agent.chat("Hello")
    agent.reset_history()
    assert agent.history_length == 0
    print("✓ History reset works")


if __name__ == "__main__":
    print("\n--- Testing Agent ---\n")
    test_basic_chat()
    test_rag_chat()
    test_streaming()
    test_history_maintained()
    test_history_reset()
    print("\n--- Agent tests passed ---\n")