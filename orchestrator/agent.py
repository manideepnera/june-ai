import time
import asyncio
from typing import Optional
from sentence_transformers import SentenceTransformer
from llm.manager import LLMManager
from rag.retriever import RAGRetriever
from orchestrator.router import Router
from orchestrator.context_builder import ContextBuilder


class Agent:
    """
    Main orchestration agent.
    this is what the api and voice layers talk to.
    Nothing else in the system imports from llm/, rag/ directly.
    Everything goes through here.
    
    Accepts injected dependencies (embed_model, collection) to avoid
    reinitializing them on every request. This is required for proper
    lifespan management with FastAPI.
    """

    def __init__(
        self,
        embed_model: Optional[SentenceTransformer] = None,
        collection = None
    ):
        print("Initializing June AI agent...")
        self.embed_model = embed_model  # Injected from services
        self.collection = collection    # Injected from services
        self.llm = LLMManager()
        self.retriever = RAGRetriever(
            embed_model=embed_model,
            collection=collection
        )
        self.router = Router()
        self.context_builder = ContextBuilder()
        self.history: list[dict] = []
        print("✓ Agent ready\n")


    def chat(self, user_input: str) -> str:
        """
        Single-shot chat. returns full response string.
        use for testing and simple integrations.
        """
        response = ""
        for chunk in self.chat_stream(user_input):
            response += chunk
        return response
    
    def chat_stream(self, user_input: str):
        """
        Streaming chat. Yields response chunks as they arrive.
        Use this for the API and UI - gives real-time feel.
        """

        user_input = user_input.strip()
        if not user_input:
            return
        
        start_time = time.time()
        timing = {}  # Track timing for each phase

        # 1. Decide what is needed
        decision = self.router.analyze(
            user_input,
            has_knowledge_base=self.retriever.has_knowledge()
        )

        # 2. Gather RAG context if needed
        rag_context = ""
        rag_sources = []
        rag_timings = {"embed_ms": 0, "retrieval_ms": 0}
        
        if decision.needs_rag:
            rag_start = time.time()
            results = self.retriever.search(user_input, top_k=3)  # Optimized from 4
            if results:
                rag_context = self.retriever.format_context(results)
                rag_sources = [r["source"] for r in results]
                # Extract timing info from retriever
                if results and "_embed_ms" in results[0]:
                    rag_timings["embed_ms"] = results[0].get("_embed_ms", 0)
                    rag_timings["retrieval_ms"] = results[0].get("_retrieval_ms", 0)
            timing["rag_total_ms"] = round((time.time() - rag_start) * 1000, 1)

        # 3. Build final prompt
        prompt_start = time.time()
        prompt = self.context_builder.build(
            user_input=user_input,
            rag_context=rag_context,
            history=self.history
        )
        timing["prompt_ms"] = round((time.time() - prompt_start) * 1000, 1)

        system = self.context_builder.get_system_prompt()

        # 4. Stream response from LLM
        llm_start = time.time()
        response_chunks = []
        chunk_count = 0
        
        for chunk in self.llm.stream(prompt, system=system):
            response_chunks.append(chunk)
            yield chunk  # Stream immediately without buffering
            chunk_count += 1

        timing["llm_ms"] = round((time.time() - llm_start) * 1000, 1)

        # 5. Save to history
        full_response = "".join(response_chunks)
        self.history.append({
            "user": user_input,
            "june": full_response
        })

        # 6. Log detailed timing breakdown
        elapsed_total = round((time.time() - start_time) * 1000)
        
        rag_info = ""
        if decision.needs_rag and rag_sources:
            # Calculate context format time (avoid floating point artifacts)
            context_format_ms = max(0, timing.get('rag_total_ms', 0) - rag_timings['embed_ms'] - rag_timings['retrieval_ms'])
            rag_info = (
                f"\n  ├─ Embedding: {rag_timings['embed_ms']}ms\n"
                f"  ├─ Retrieval: {rag_timings['retrieval_ms']}ms\n"
                f"  ├─ Context Format: {context_format_ms}ms\n"
                f"  └─ Sources: {len(rag_sources)}"
            )
        else:
            rag_info = "\n  └─ Skipped"
        
        print(
            f"\n{'='*60}\n"
            f"Response: {len(full_response)} chars | {chunk_count} chunks\n"
            f"Total Time: {elapsed_total}ms\n"
            f"├─ RAG: {timing.get('rag_total_ms', 0)}ms{rag_info}\n"
            f"├─ Prompt Build: {timing.get('prompt_ms', 0)}ms\n"
            f"├─ LLM Generation: {timing.get('llm_ms', 0)}ms\n"
            f"└─ History: {len(self.history)} turns\n"
            f"{'='*60}"
        )

    async def chat_stream_async(self, user_input: str):
        """
        Async streaming chat with event loop yielding.
        Use this for WebSocket handlers to allow ping/pong heartbeats.
        Yields control to event loop after each chunk to keep connection alive.
        """
        user_input = user_input.strip()
        if not user_input:
            return
        
        start_time = time.time()
        timing = {}  # Track timing for each phase

        # 1. Decide what is needed
        decision = self.router.analyze(
            user_input,
            has_knowledge_base=self.retriever.has_knowledge()
        )

        # 2. Gather RAG context if needed (offload to executor to avoid blocking)
        rag_context = ""
        rag_sources = []
        rag_timings = {"embed_ms": 0, "retrieval_ms": 0}
        
        if decision.needs_rag:
            rag_start = time.time()
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.retriever.search,
                user_input,
                3  # Optimized from 4
            )
            if results:
                rag_context = self.retriever.format_context(results)
                rag_sources = [r["source"] for r in results]
                # Extract timing info from retriever
                if results and "_embed_ms" in results[0]:
                    rag_timings["embed_ms"] = results[0].get("_embed_ms", 0)
                    rag_timings["retrieval_ms"] = results[0].get("_retrieval_ms", 0)
            timing["rag_total_ms"] = round((time.time() - rag_start) * 1000, 1)

        # 3. Build final prompt
        prompt_start = time.time()
        prompt = self.context_builder.build(
            user_input=user_input,
            rag_context=rag_context,
            history=self.history
        )
        timing["prompt_ms"] = round((time.time() - prompt_start) * 1000, 1)

        system = self.context_builder.get_system_prompt()

        # 4. Stream response from LLM
        llm_start = time.time()
        response_chunks = []
        chunk_count = 0
        
        for chunk in self.llm.stream(prompt, system=system):
            response_chunks.append(chunk)
            yield chunk  # Stream immediately without buffering
            chunk_count += 1
            # Yield control to event loop after each chunk
            # This allows WebSocket ping/pong to process
            await asyncio.sleep(0)

        timing["llm_ms"] = round((time.time() - llm_start) * 1000, 1)

        # 5. Save to history
        full_response = "".join(response_chunks)
        self.history.append({
            "user": user_input,
            "june": full_response
        })

        # 6. Log detailed timing breakdown
        elapsed_total = round((time.time() - start_time) * 1000)
        
        rag_info = ""
        if decision.needs_rag and rag_sources:
            # Calculate context format time (avoid floating point artifacts)
            context_format_ms = max(0, timing.get('rag_total_ms', 0) - rag_timings['embed_ms'] - rag_timings['retrieval_ms'])
            rag_info = (
                f"\n  ├─ Embedding: {rag_timings['embed_ms']}ms\n"
                f"  ├─ Retrieval: {rag_timings['retrieval_ms']}ms\n"
                f"  ├─ Context Format: {context_format_ms}ms\n"
                f"  └─ Sources: {len(rag_sources)}"
            )
        else:
            rag_info = "\n  └─ Skipped"
        
        print(
            f"\n{'='*60}\n"
            f"Response: {len(full_response)} chars | {chunk_count} chunks\n"
            f"Total Time: {elapsed_total}ms\n"
            f"├─ RAG: {timing.get('rag_total_ms', 0)}ms{rag_info}\n"
            f"├─ Prompt Build: {timing.get('prompt_ms', 0)}ms\n"
            f"├─ LLM Generation: {timing.get('llm_ms', 0)}ms\n"
            f"└─ History: {len(self.history)} turns\n"
            f"{'='*60}"
        )

    def reset_history(self):
        """Clear conversation history. Start fresh."""
        self.history = []
        print("✓ Conversation history cleared")

    @property
    def history_length(self) -> int:
        return len(self.history)