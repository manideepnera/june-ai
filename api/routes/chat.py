from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from services import services

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class Chatresponse(BaseModel):
    response: str


@router.post("/chat", response_model=Chatresponse)
async def chat_http(request: ChatRequest):
    """
    Simple HTTP endpoint.
    Sends full message, waits for full response.
    Good for testing. Use WebSocket for real-time streaming.
    """
    agent = services.agent
    if agent is None:
        return Chatresponse(response="Error: Agent not initialized. Server may not have started properly.")
    
    response = agent.chat(request.message)
    return Chatresponse(response=response)


@router.post("/reset")
async def reset_history():
    """Clear the agent's conversation history."""
    agent = services.agent
    if agent is None:
        return {"status": "error", "message": "Agent not initialized"}
    
    agent.reset_history()
    return {"status": "history cleared"}


@router.get("/status")
async def status():
    """Get current agent status."""
    agent = services.agent
    if agent is None:
        return {"status": "error", "message": "Agent not initialized"}
    
    return {
        "history_turns": agent.history_length,
        "knowledge_available": agent.retriever.has_knowledge(),
        "knowledge_chunks": agent.retriever.store.count()
    }


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    Client sends a message, server streams chunks back in real time.
    
    Uses async streaming with event loop yielding after each chunk
    to keep WebSocket ping/pong heartbeats active during long inference.

    Message format (client → server):  plain text string
    Message format (server → client):
        chunks of text during streaming
        then "[DONE]" signal when complete
    """
    await websocket.accept()
    agent = services.agent
    
    if agent is None:
        await websocket.send_text("[ERROR: Agent not initialized. Server may not have started properly.]")
        await websocket.close()
        return

    try:
        while True:
            # Wait for message from client
            user_input = await websocket.receive_text()

            if not user_input.strip():
                continue

            try:
                # Stream response chunks back using async method
                # This properly yields control to event loop after each chunk
                async for chunk in agent.chat_stream_async(user_input):
                    if chunk:
                        await websocket.send_text(chunk)

                # Signal end of response
                await websocket.send_text("[DONE]")
                
            except Exception as stream_error:
                print(f"Streaming error: {stream_error}")
                await websocket.send_text(f"[ERROR: {str(stream_error)}]")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass