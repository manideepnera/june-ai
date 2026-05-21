import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.router import Router


def test_router():
    router = Router()

    cases = [
        # (input, expected_rag, description)
        ("What is RAG?",                    True,  "direct question"),
        ("How does deep learning work?",    True,  "how question"),
        ("Explain neural networks to me",   True,  "explain keyword"),
        ("What did I write about Python?",  True,  "personal notes query"),
        ("hello",                           False, "greeting"),
        ("thanks",                          False, "casual thanks"),
        ("hi there",                        False, "casual hi"),
        ("ok got it",                       False, "casual acknowledgement"),
    ]

    print()
    all_passed = True
    for text, expected_rag, desc in cases:
        decision = router.analyze(text, has_knowledge_base=True)
        status = "✓" if decision.needs_rag == expected_rag else "✗"
        if decision.needs_rag != expected_rag:
            all_passed = False
        print(
            f"  {status} RAG={decision.needs_rag} | "
            f"'{text}' ({desc})"
        )
        print(f"      Reason: {decision.reasoning}")

    assert all_passed, "Some router decisions were wrong"
    print("\n✓ All router tests passed")


if __name__ == "__main__":
    print("\n--- Testing Router ---")
    test_router()
    print("\n--- Router tests passed ---\n")