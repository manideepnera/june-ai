import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_websocket_streaming():
    """
    Test that the WebSocket endpoint streams chunks and ends with [DONE].
    Server must be running: uvicorn api.server:app --port 8000
    """
    try:
        import websockets
    except ImportError:
        print("Installing websockets...")
        os.system("pip install websockets")
        import websockets

    uri = "ws://localhost:8000/api/ws"
    print(f"\nConnecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected")

            message = "What is RAG?"
            await ws.send(message)
            print(f"✓ Sent: '{message}'")
            print("\nJune: ", end="", flush=True)

            chunks = []
            while True:
                chunk = await ws.recv()
                if chunk == "[DONE]":
                    break
                print(chunk, end="", flush=True)
                chunks.append(chunk)

            print("\n")
            full_response = "".join(chunks)
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_response) > 50
            print(f"✓ Streaming works.")
            print(f"  Chunks received : {len(chunks)}")
            print(f"  Response length : {len(full_response)} chars")

    except ConnectionRefusedError:
        print("\n✗ Could not connect. Is the server running?")
        print("  Start it with: uvicorn api.server:app --reload --port 8000")


if __name__ == "__main__":
    print("\n--- Testing WebSocket ---")
    print("(Server must be running on port 8000)")
    asyncio.run(test_websocket_streaming())
    print("\n--- WebSocket test done ---\n")