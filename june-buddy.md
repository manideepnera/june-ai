# June AI — Full Project Flow & Architecture Guide

> A modular, local-first personal AI system.  
> Built in phases. Replaceable internals. Stable interfaces.

---

## Table of Contents

1. [Vision & Philosophy](#1-vision--philosophy)
2. [System Overview](#2-system-overview)
3. [Hardware Reality](#3-hardware-reality)
4. [Full Project Structure](#4-full-project-structure)
5. [Phase 1 — Core Intelligence](#5-phase-1--core-intelligence)
   - [5.1 LLM Layer](#51-llm-layer)
   - [5.2 RAG Engine](#52-rag-engine)
   - [5.3 Orchestrator](#53-orchestrator)
   - [5.4 API Layer](#54-api-layer)
   - [5.5 Voice Layer](#55-voice-layer)
   - [5.6 Chat UI](#56-chat-ui)
6. [Phase 2 — Memory & Personalization](#6-phase-2--memory--personalization)
7. [Phase 3 — Internet & Tools](#7-phase-3--internet--tools)
8. [Core Module — Shared Utilities](#8-core-module--shared-utilities)
9. [Config System](#9-config-system)
10. [Storage Layout](#10-storage-layout)
11. [Request Flow — End to End](#11-request-flow--end-to-end)
12. [Interface Contracts](#12-interface-contracts)
13. [Tech Stack Summary](#13-tech-stack-summary)
14. [Build Order](#14-build-order)
15. [Key Principles](#15-key-principles)

---

## 1. Vision & Philosophy

You are not building a chatbot.  
You are building **AI infrastructure** — a personal operating companion.

### What it does

- Talks naturally via voice
- Understands your personal knowledge (RAG)
- Remembers your interactions (memory)
- Accesses live information (internet)
- Executes actions (tools)
- Evolves over time without redesigning internals

### Core philosophy

```
system > model
```

The architecture must survive model changes.  
Today it runs Qwen2.5. Tomorrow it runs something better.  
Nothing outside `llm/` should care which model is running.

### Design principles

| Principle | What it means |
|---|---|
| Modular | Every layer is independent |
| Replaceable | Swap internals without breaking outer contracts |
| Scalable | Add capability without restructuring |
| Debuggable | Logs at every layer, clear error boundaries |
| Model-independent | `system > model` — LLM is a component, not the foundation |
| Local-first | Ollama primary, API as fallback only |

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────┐
│                     June AI                         │
│                                                     │
│  ┌─────────┐    ┌──────────────────────────────┐   │
│  │  Voice  │    │         Orchestrator         │   │
│  │  Layer  │───▶│  router → context_builder    │   │
│  └─────────┘    │       → agent                │   │
│                 └──────────┬───────────────────┘   │
│  ┌─────────┐               │                       │
│  │  Chat   │───────────────┤                       │
│  │   UI    │               │                       │
│  └─────────┘               ▼                       │
│                 ┌──────────────────────────────┐   │
│                 │         LLM Manager          │   │
│                 │   ollama / openai / qwen     │   │
│                 └──────────────────────────────┘   │
│                          ▲   ▲                     │
│              ┌───────────┘   └──────────┐          │
│         ┌────┴────┐             ┌───────┴──────┐   │
│         │   RAG   │             │    Memory    │   │
│         │ Engine  │             │    Layer     │   │
│         └─────────┘             └──────────────┘   │
│                                                     │
│         ┌─────────┐             ┌──────────────┐   │
│         │Internet │             │    Tools     │   │
│         │  Layer  │             │   Registry   │   │
│         └─────────┘             └──────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Independence rule

Every module exposes clean interfaces and knows nothing about its siblings.

```
memory.retrieve()    ← memory doesn't know about RAG
rag.search()         ← RAG doesn't know about voice  
llm.generate()       ← LLM doesn't know about memory
voice.listen()       ← voice doesn't know about LLM
internet.search()    ← internet doesn't know about tools
```

Only the **orchestrator** knows about everyone.

---

## 3. Hardware Reality

**Machine:** Lenovo Yoga Slim 7i Pro 14"  
**CPU:** Intel Core i7-11370H (4 cores / 8 threads)  
**RAM:** 8GB or 16GB LPDDR4X  
**GPU:** Intel Iris Xe (integrated) + NVIDIA MX450 (2GB GDDR6)

### Implication

MX450 has only 2GB VRAM — not usable for AI inference.  
Everything runs on **CPU only**.

### Practical model choices

| Model | RAM usage | CPU speed | Recommendation |
|---|---|---|---|
| Qwen2.5 3B (Q4) | ~3.5GB | Fast | ✅ Best daily driver |
| Phi-3 Mini 3.8B | ~4GB | Fast | ✅ Strong reasoning |
| Llama 3.2 3B | ~4GB | Fast | ✅ Good alternative |
| Qwen2.5 7B (Q4) | ~6GB | Moderate | ⚠️ Workable on 16GB |
| 14B+ models | 10GB+ | Slow | ❌ Avoid for now |

### Provider strategy

```yaml
# config/models.yaml
active_provider: ollama     # local — always try first
model: qwen2.5:3b

fallback_provider: openai   # API — when heavy reasoning needed
fallback_model: gpt-4o
```

---

## 4. Full Project Structure

```
june-ai/
│
├── core/                    # shared utilities, used everywhere
│   ├── logger.py
│   ├── settings.py
│   ├── events.py
│   ├── utils.py
│   ├── constants.py
│   └── exceptions.py
│
├── llm/                     # model communication ONLY
│   ├── providers/
│   │   ├── base.py          # BaseLLMProvider — the contract
│   │   ├── ollama.py        # local inference
│   │   ├── openai.py        # API fallback
│   │   └── qwen.py          # optional direct integration
│   ├── prompts/
│   │   ├── system.py
│   │   └── templates.py
│   ├── tokenizer/
│   ├── inference.py
│   └── manager.py           # reads config, returns active provider
│
├── rag/                     # knowledge engine
│   ├── loaders/
│   │   ├── pdf_loader.py
│   │   ├── txt_loader.py
│   │   ├── md_loader.py
│   │   └── web_loader.py
│   ├── chunking/
│   │   └── splitter.py
│   ├── embeddings/
│   │   └── embedder.py      # sentence-transformers, CPU
│   ├── vectordb/
│   │   └── chroma.py        # ChromaDB wrapper
│   ├── retriever.py
│   ├── indexing.py
│   └── pipeline.py          # load → chunk → embed → store
│
├── memory/                  # phase 2
│   ├── short_term/
│   │   └── buffer.py        # in-context conversation window
│   ├── long_term/
│   │   └── store.py         # post-session summaries
│   ├── semantic/
│   │   └── search.py        # vector search over memories
│   ├── storage/
│   │   └── sqlite.py
│   ├── retrieval.py
│   └── manager.py
│
├── voice/                   # voice layer, added last in phase 1
│   ├── wakeword/
│   │   └── detector.py      # OpenWakeWord
│   ├── stt/
│   │   └── whisper.py       # Whisper tiny/base
│   ├── tts/
│   │   └── piper.py         # Piper TTS
│   ├── streaming/
│   └── manager.py
│
├── internet/                # phase 3
│   ├── search/
│   │   └── searcher.py
│   ├── scraping/
│   │   └── scraper.py
│   ├── apis/
│   ├── summarizer/
│   └── manager.py
│
├── tools/                   # phase 3
│   ├── system/
│   ├── productivity/
│   ├── browser/
│   ├── coding/
│   └── registry.py          # tool discovery and invocation
│
├── orchestrator/            # the brain — only layer that knows everything
│   ├── planner.py
│   ├── router.py            # decides: RAG? memory? internet? tools?
│   ├── context_builder.py   # assembles final prompt
│   ├── workflow.py
│   ├── decision_engine.py
│   └── agent.py             # main entry point
│
├── api/                     # backend server
│   ├── server.py            # FastAPI app
│   ├── routes/
│   │   ├── chat.py
│   │   └── knowledge.py
│   ├── websocket/
│   │   └── handler.py       # streaming chat
│   └── auth/
│
├── ui/                      # frontend
│   └── (chat interface — phase 1 can be terminal)
│
├── config/
│   ├── models.yaml
│   ├── voice.yaml
│   ├── memory.yaml
│   └── rag.yaml
│
├── storage/
│   ├── vectors/             # ChromaDB files
│   ├── memory/              # SQLite memory db
│   ├── conversations/       # chat history
│   ├── cache/
│   └── files/               # uploaded documents
│
├── logs/                    # never skip this
│   ├── prompts/
│   ├── retrievals/
│   ├── errors/
│   └── latency/
│
├── tests/
│   ├── test_llm.py
│   ├── test_rag.py
│   ├── test_memory.py
│   └── test_voice.py
│
├── main.py
└── requirements.txt
```

---

## 5. Phase 1 — Core Intelligence

**Goal:** A working assistant that listens, knows your knowledge, and responds.

### Build order within Phase 1

```
Step 1 → LLM layer         (reliable generation)
Step 2 → RAG engine        (personal knowledge)
Step 3 → Orchestrator      (wires them together)
Step 4 → API + Chat UI     (interact without voice friction)
Step 5 → Voice layer       (add last, once brain is stable)
```

---

### 5.1 LLM Layer

**Location:** `llm/`  
**Responsibility:** Input → model → output. Nothing else.  
**Does NOT know about:** RAG, memory, voice, tools.

#### Base interface — `llm/providers/base.py`

```python
from abc import ABC, abstractmethod
from typing import Iterator, List

class BaseLLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Single-shot generation. Returns full response string."""
        pass

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Streaming generation. Yields chunks as they arrive."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for a piece of text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is reachable."""
        pass
```

#### Ollama provider — `llm/providers/ollama.py`

```python
import requests
from .base import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        return response.json()["response"]

    def stream(self, prompt: str, system: str = ""):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True
        }
        with requests.post(f"{self.base_url}/api/generate",
                           json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    yield chunk.get("response", "")

    def embed(self, text: str):
        payload = {"model": self.model, "prompt": text}
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
        return response.json()["embedding"]

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except:
            return False
```

#### Manager — `llm/manager.py`

```python
from core.settings import settings
from llm.providers.ollama import OllamaProvider
from llm.providers.openai import OpenAIProvider

class LLMManager:

    def __init__(self):
        self._provider = None
        self._load_provider()

    def _load_provider(self):
        if settings.active_provider == "ollama":
            provider = OllamaProvider(model=settings.model)
            if provider.is_available():
                self._provider = provider
                return
        # Fallback to API
        self._provider = OpenAIProvider(model=settings.fallback_model)

    def generate(self, prompt: str, system: str = "") -> str:
        return self._provider.generate(prompt, system)

    def stream(self, prompt: str, system: str = ""):
        return self._provider.stream(prompt, system)

    def embed(self, text: str):
        return self._provider.embed(text)

# Singleton
llm = LLMManager()
```

---

### 5.2 RAG Engine

**Location:** `rag/`  
**Responsibility:** Load your documents → chunk → embed → store → retrieve.  
**This is the assistant's external brain.**

#### Pipeline — `rag/pipeline.py`

```
Document files
     ↓
  Loader         (pdf / txt / md / web)
     ↓
  Chunker        (split into ~512 token chunks with overlap)
     ↓
  Embedder       (sentence-transformers → float vectors)
     ↓
  ChromaDB       (store vectors + original text)
     ↓
  Retriever      (query → top-k relevant chunks)
     ↓
  Context        (inject into orchestrator prompt)
```

#### Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | ChromaDB | Lightweight, local, no server needed |
| Embeddings | `all-MiniLM-L6-v2` | Fast on CPU, 384 dimensions, good quality |
| Chunk size | 512 tokens | Balances context and precision |
| Chunk overlap | 50 tokens | Prevents losing context at boundaries |
| Top-k retrieval | 3–5 chunks | Enough context, avoids prompt bloat |

#### Retriever — `rag/retriever.py`

```python
from rag.vectordb.chroma import ChromaStore
from llm.manager import llm

class RAGRetriever:

    def __init__(self):
        self.store = ChromaStore()

    def search(self, query: str, top_k: int = 4) -> list[str]:
        """Return top-k relevant text chunks for a query."""
        query_embedding = llm.embed(query)
        results = self.store.query(query_embedding, top_k=top_k)
        return results

    def format_context(self, chunks: list[str]) -> str:
        """Format chunks into a clean context block for the prompt."""
        if not chunks:
            return ""
        context = "\n\n---\n\n".join(chunks)
        return f"Relevant knowledge:\n\n{context}"
```

---

### 5.3 Orchestrator

**Location:** `orchestrator/`  
**Responsibility:** Decide what to call, build context, send to LLM.  
**This is the most important layer.**

#### Decision flow

```
User input arrives
       ↓
   Router checks:
   ┌─────────────────────────────┐
   │ Is RAG needed?      yes/no  │
   │ Is memory needed?   yes/no  │  ← phase 2
   │ Is internet needed? yes/no  │  ← phase 3
   │ Are tools needed?   yes/no  │  ← phase 3
   └─────────────────────────────┘
       ↓
   Context builder assembles:
   - system prompt
   - retrieved RAG chunks  (if needed)
   - conversation history
   - user message
       ↓
   Send to LLM → stream response
```

#### Agent — `orchestrator/agent.py`

```python
from llm.manager import llm
from rag.retriever import RAGRetriever
from orchestrator.router import Router
from orchestrator.context_builder import ContextBuilder

class Agent:

    def __init__(self):
        self.retriever = RAGRetriever()
        self.router = Router()
        self.context_builder = ContextBuilder()
        self.history = []

    def chat(self, user_input: str):
        # 1. Decide what's needed
        needs = self.router.analyze(user_input)

        # 2. Gather context
        rag_context = ""
        if needs.rag:
            chunks = self.retriever.search(user_input)
            rag_context = self.retriever.format_context(chunks)

        # 3. Build final prompt
        prompt = self.context_builder.build(
            user_input=user_input,
            rag_context=rag_context,
            history=self.history
        )

        # 4. Generate and stream response
        response = ""
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
            response += chunk

        # 5. Update history
        self.history.append({"user": user_input, "assistant": response})
        return response
```

---

### 5.4 API Layer

**Location:** `api/`  
**Responsibility:** Expose the agent over HTTP and WebSocket.

#### Server — `api/server.py`

```python
from fastapi import FastAPI, WebSocket
from orchestrator.agent import Agent

app = FastAPI(title="June AI")
agent = Agent()

@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        user_input = await websocket.receive_text()
        for chunk in agent.chat_stream(user_input):
            await websocket.send_text(chunk)

@app.post("/chat")
async def chat_http(body: dict):
    response = agent.chat(body["message"])
    return {"response": response}
```

**Run with:**

```bash
uvicorn api.server:app --reload --port 8000
```

---

### 5.5 Voice Layer

**Location:** `voice/`  
**Responsibility:** Wake word → listen → STT → send to agent → TTS → speak.  
**Add this last in Phase 1.**

#### Voice pipeline

```
"Hey June" detected (OpenWakeWord)
        ↓
Record audio until silence
        ↓
Whisper STT → text
        ↓
Send text to orchestrator
        ↓
Get response text back
        ↓
Piper TTS → audio
        ↓
Play audio
```

#### Model choices for your hardware

| Task | Model | RAM | Speed |
|---|---|---|---|
| Wake word | OpenWakeWord | ~50MB | Real-time |
| STT | Whisper tiny | ~150MB | Fast |
| STT | Whisper base | ~290MB | Good quality |
| TTS | Piper (low) | ~50MB | Real-time |

#### Manager — `voice/manager.py`

```python
from voice.wakeword.detector import WakeWordDetector
from voice.stt.whisper import WhisperSTT
from voice.tts.piper import PiperTTS

class VoiceManager:

    def __init__(self):
        self.wakeword = WakeWordDetector()
        self.stt = WhisperSTT(model="base")
        self.tts = PiperTTS()

    def listen(self) -> str:
        """Block until wake word, then return transcribed speech."""
        self.wakeword.wait_for_activation()
        audio = self.record_until_silence()
        return self.stt.transcribe(audio)

    def speak(self, text: str):
        """Convert text to speech and play it."""
        audio = self.tts.synthesize(text)
        self.play_audio(audio)
```

---

### 5.6 Chat UI

For Phase 1, start with terminal. Move to web UI once voice is stable.

#### Terminal interface — `main.py`

```python
from orchestrator.agent import Agent

def main():
    agent = Agent()
    print("June AI ready. Type your message.")
    print("─" * 40)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        print("\nJune: ", end="")
        agent.chat(user_input)
        print()

if __name__ == "__main__":
    main()
```

---

## 6. Phase 2 — Memory & Personalization

**Goal:** The assistant starts remembering you.

### Memory architecture

```
Short-term memory     ← conversation window (in-context, current session)
Long-term memory      ← post-session summaries stored in SQLite
Semantic memory       ← vector search over past memories (ChromaDB)
```

### Memory flow

```
Session ends
     ↓
Summarizer compresses conversation
     ↓
Summary stored in SQLite (long-term)
Summary embedded → stored in ChromaDB (semantic)
     ↓
Next session starts
     ↓
Router checks: is memory relevant?
     ↓
Semantic search over past memories
     ↓
Top memories injected into context
```

### What gets remembered

- User preferences ("I like concise answers")
- Past decisions and context
- Ongoing goals and projects
- Relationship context ("My team uses Python")
- Behavioral patterns

### Memory manager — `memory/manager.py`

```python
class MemoryManager:

    def store(self, session_summary: str):
        """Store a session summary in long-term + semantic store."""
        pass

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve semantically relevant memories for a query."""
        pass

    def get_recent(self, n: int = 5) -> list[str]:
        """Get n most recent session summaries."""
        pass
```

### Orchestrator update for Phase 2

```python
# orchestrator/agent.py — updated
needs = self.router.analyze(user_input)

memory_context = ""
if needs.memory:
    memories = self.memory.retrieve(user_input)
    memory_context = self.memory.format_context(memories)

prompt = self.context_builder.build(
    user_input=user_input,
    rag_context=rag_context,
    memory_context=memory_context,   # ← added
    history=self.history
)
```

---

## 7. Phase 3 — Internet & Tools

**Goal:** The assistant becomes aware of the world and can take actions.

### Internet layer

```
internet/
├── search/         ← DuckDuckGo or SearXNG (local, no API key)
├── scraping/       ← BeautifulSoup, extract page content
├── apis/           ← weather, news, etc.
├── summarizer/     ← compress long web pages before injecting
└── manager.py
```

```python
class InternetManager:
    def search(self, query: str) -> list[str]: pass
    def fetch_page(self, url: str) -> str: pass
    def summarize(self, content: str) -> str: pass
```

### Tools layer

```
tools/
├── system/         ← file ops, terminal commands
├── productivity/   ← calendar, notes, reminders
├── browser/        ← open URLs, interact with browser
├── coding/         ← run code, lint, format
└── registry.py     ← register and discover tools
```

#### Tool interface

```python
class BaseTool:
    name: str
    description: str

    def execute(self, **kwargs) -> str:
        """Run the tool. Returns result as string."""
        pass
```

#### Registry — `tools/registry.py`

```python
class ToolRegistry:
    _tools = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def list_available(self) -> list[str]:
        return list(self._tools.keys())
```

### Orchestrator update for Phase 3

```python
# Decision tree becomes:
if needs.internet:
    web_results = self.internet.search(user_input)
    web_context = self.internet.summarize(web_results)

if needs.tools:
    tool_name = self.router.select_tool(user_input)
    tool_result = self.tools.execute(tool_name, input=user_input)
```

---

## 8. Core Module — Shared Utilities

**Location:** `core/`  
**Rule:** Used by every module. Has no dependencies on other modules.

### Settings — `core/settings.py`

```python
from pydantic_settings import BaseSettings
import yaml

class Settings(BaseSettings):
    active_provider: str = "ollama"
    model: str = "qwen2.5:3b"
    fallback_provider: str = "openai"
    fallback_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

settings = Settings.from_yaml("config/models.yaml")
```

### Logger — `core/logger.py`

```python
import logging
import json
from datetime import datetime
from pathlib import Path

class JuneLogger:

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        Path("logs").mkdir(exist_ok=True)

    def log_prompt(self, prompt: str, response: str, latency_ms: float):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:500],
            "response": response[:500],
            "latency_ms": latency_ms
        }
        with open("logs/prompts.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_retrieval(self, query: str, chunks: list, scores: list):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_chunks": len(chunks),
            "scores": scores
        }
        with open("logs/retrievals.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
```

**Why logging matters:** AI systems are nearly impossible to debug without it.  
Log prompts, retrievals, latency, and errors from day one.

---

## 9. Config System

**Location:** `config/`  
**Rule:** Swap models, providers, parameters — with zero code changes.

### `config/models.yaml`

```yaml
active_provider: ollama
model: qwen2.5:3b

fallback_provider: openai
fallback_model: gpt-4o
openai_api_key: ${OPENAI_API_KEY}

ollama_base_url: http://localhost:11434
max_tokens: 2048
temperature: 0.7
```

### `config/rag.yaml`

```yaml
chunk_size: 512
chunk_overlap: 50
top_k: 4
embedding_model: all-MiniLM-L6-v2
vector_store: chroma
chroma_path: storage/vectors
```

### `config/voice.yaml`

```yaml
wakeword_model: hey_june
stt_model: whisper_base
tts_model: piper_low
sample_rate: 16000
silence_threshold: 0.01
silence_duration: 1.5
```

### `config/memory.yaml`

```yaml
short_term_window: 10
long_term_summary_threshold: 20
semantic_top_k: 3
memory_db_path: storage/memory/june.db
```

---

## 10. Storage Layout

```
storage/
├── vectors/             ← ChromaDB persistence
│   └── chroma.sqlite3
│
├── memory/              ← phase 2
│   └── june.db          ← SQLite (sessions, summaries, preferences)
│
├── conversations/       ← raw chat logs
│   └── 2024-01-15.jsonl
│
├── cache/               ← embedding cache, web page cache
│
└── files/               ← documents you feed to RAG
    ├── notes/
    ├── pdfs/
    └── research/
```

---

## 11. Request Flow — End to End

### Phase 1 — Voice request

```
1.  "Hey June" spoken
          ↓
2.  OpenWakeWord detects activation
          ↓
3.  Microphone records audio until silence
          ↓
4.  Whisper STT transcribes → "What did I write in my notes about RAG?"
          ↓
5.  voice/manager.py sends text to orchestrator/agent.py
          ↓
6.  router.py analyzes → needs_rag = True
          ↓
7.  rag/retriever.py: embed query → ChromaDB → top 4 chunks returned
          ↓
8.  context_builder.py assembles:
     [system prompt]
     [rag chunks]
     [conversation history]
     [user message]
          ↓
9.  llm/manager.py: checks Ollama available → yes
          ↓
10. OllamaProvider streams response token by token
          ↓
11. Piper TTS synthesizes response audio
          ↓
12. Audio plays back to user
          ↓
13. Prompt + response logged to logs/prompts.jsonl
```

### Phase 2 — adds memory injection at step 7

```
7a. memory/manager.py: semantic search over past memories
7b. Top 3 relevant memories injected into context alongside RAG chunks
```

### Phase 3 — adds internet and tools

```
6a. router detects: needs_internet = True (query about current events)
6b. internet/manager.py searches web → summarizes results
6c. Tool invocation if action needed (open file, run code, etc.)
```

---

## 12. Interface Contracts

These are the stable contracts. Internals can change completely.  
Everything outside a module depends only on these.

```python
# LLM
llm.generate(prompt: str, system: str) -> str
llm.stream(prompt: str, system: str) -> Iterator[str]
llm.embed(text: str) -> List[float]

# RAG
rag.search(query: str, top_k: int) -> List[str]
rag.index(file_path: str) -> None

# Memory (phase 2)
memory.retrieve(query: str) -> List[str]
memory.store(summary: str) -> None

# Voice
voice.listen() -> str
voice.speak(text: str) -> None

# Internet (phase 3)
internet.search(query: str) -> List[str]
internet.fetch(url: str) -> str

# Tools (phase 3)
tools.execute(name: str, **kwargs) -> str
tools.list() -> List[str]

# Orchestrator
agent.chat(user_input: str) -> str
agent.chat_stream(user_input: str) -> Iterator[str]
```

---

## 13. Tech Stack Summary

### Phase 1

| Component | Library | Why |
|---|---|---|
| Local inference | Ollama | Easy model management, REST API |
| LLM models | Qwen2.5 3B / Phi-3 mini | Fast on CPU, capable |
| Embeddings | sentence-transformers | CPU-friendly, good quality |
| Vector DB | ChromaDB | Lightweight, no server, local |
| API framework | FastAPI | Async, WebSocket, fast |
| Config validation | Pydantic | Type-safe, catches errors early |
| Config files | PyYAML | Human-readable |
| STT | OpenAI Whisper (tiny/base) | Runs on CPU, good accuracy |
| TTS | Piper | Fast, local, natural voice |
| Wake word | OpenWakeWord | Lightweight, customizable |

### Phase 2 additions

| Component | Library | Why |
|---|---|---|
| Memory store | SQLite | Simple, local, reliable |
| Memory search | ChromaDB (second collection) | Same stack, reused |

### Phase 3 additions

| Component | Library | Why |
|---|---|---|
| Web search | DuckDuckGo API / SearXNG | No API key needed |
| Scraping | BeautifulSoup + httpx | Lightweight |
| Browser control | Playwright | Reliable automation |

---

## 14. Build Order

### Week 1 — LLM spine

- [ ] Set up repo structure
- [ ] Write `BaseLLMProvider` interface
- [ ] Implement `OllamaProvider`
- [ ] Implement `LLMManager` with config
- [ ] Test: `llm.generate("hello")` works
- [ ] Write `OpenAIProvider` as fallback
- [ ] Test provider switching via config

### Week 2 — RAG engine

- [ ] Write document loaders (pdf, txt, md)
- [ ] Write chunker with overlap
- [ ] Set up sentence-transformers embedder
- [ ] Set up ChromaDB store
- [ ] Write retriever
- [ ] Index some of your real documents
- [ ] Test: `rag.search("your topic")` returns relevant chunks

### Week 3 — Orchestrator + terminal chat

- [ ] Write `Router` — basic RAG decision logic
- [ ] Write `ContextBuilder` — assembles prompt
- [ ] Write `Agent` — wires everything together
- [ ] Test end-to-end: question about your docs → answer
- [ ] Build terminal chat interface in `main.py`
- [ ] Add prompt + retrieval logging

### Week 4 — API + basic UI

- [ ] FastAPI server with `/chat` endpoint
- [ ] WebSocket handler for streaming
- [ ] Simple web chat UI (can be basic HTML)
- [ ] Test full stack: browser → API → agent → response

### Week 5 — Voice

- [ ] Install and test Whisper STT
- [ ] Install and test Piper TTS
- [ ] Set up OpenWakeWord
- [ ] Wire `VoiceManager` to agent
- [ ] Test full voice loop end to end

---

## 15. Key Principles

### Never break these rules

**1. LLM layer knows nothing**  
`llm/` never imports from `rag/`, `memory/`, `voice/`, or `tools/`.  
It only does: input → model → output.

**2. Only orchestrator knows everyone**  
`orchestrator/` is the only layer that imports from multiple modules.  
Everything else is a standalone service.

**3. All config in YAML**  
No hardcoded model names, URLs, or parameters.  
Swap anything without touching code.

**4. Every module has a manager**  
External code always calls `module/manager.py`.  
Never import internals directly from outside a module.

**5. Log everything from day one**  
Prompts, retrievals, latency, errors.  
AI systems become impossible to debug without logs.

**6. Fallback gracefully**  
Ollama down → fallback to API.  
RAG returns nothing → continue without context.  
Voice fails → fall back to text.  
Never crash. Always degrade.

**7. Test the interfaces, not the internals**  
`test_llm.py` tests `llm.generate()`, not `OllamaProvider._call_api()`.  
Internals can change. Tests must survive that.

---

## Closing Note

You are building:

```
Persistent Personalized AI Infrastructure
```

Not a chatbot script.

The distinction is everything.  
Phase 1 proves the concept.  
Phase 2 makes it yours.  
Phase 3 makes it capable.

The architecture you start with determines how far you can go.  
Start clean. Stay modular. Never couple what doesn't need to be coupled.

---

*Document version: Phase 1 design — June AI*