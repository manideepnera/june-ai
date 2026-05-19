import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loaders.router import LoaderRouter


def test_txt_loader():
    router = LoaderRouter()
    text = router.load("storage/files/notes/neural_networks.txt")
    assert len(text) > 0
    assert "neural" in text.lower()
    print(f"✓ TXT loader works. Characters loaded: {len(text)}")


if __name__ == "__main__":
    print("\n--- Testing Loaders ---\n")
    test_txt_loader()
    print("\n--- Loader tests passed ---\n")