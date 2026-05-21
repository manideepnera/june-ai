"""
June AI Server Entry Point

This module starts the FastAPI server with proper uvicorn configuration:
- Disabled reload flag to prevent WebSocket disconnects during development
- WebSocket ping timeout tuned for long LLM inference times
- Proper lifespan management for singleton services

To run:
    python main.py

Or with explicit config:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --ws-ping-timeout 60
"""

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("June AI Server Starting")
    print("="*60)
    print("API: http://localhost:8000")
    print("WebSocket: ws://localhost:8000/api/ws")
    print("="*60 + "\n")

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # IMPORTANT: No reload during WebSocket testing
        ws_ping_interval=20,   # Send ping every 20s (standard)
        ws_ping_timeout=60,    # Wait 60s for pong (increased for Ollama)
        log_level="info",
    )
