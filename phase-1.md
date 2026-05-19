# June AI — Start Steps
> Ollama setup, model install, Python env, project skeleton.  
> Follow in order. Don't skip steps.

---

## What you will have at the end

- Ollama installed and running locally
- Qwen2.5 3B model downloaded and responding
- Python environment ready
- Project folder created
- First file written and tested

---

## Step 1 — Check your system first

Open terminal and run these one by one.

```bash
python --version
```

Expected: `Python 3.10.x` or higher.  
If not installed → download from [python.org](https://python.org) and install.

```bash
pip --version
```

Expected: `pip 23.x` or higher.

```bash
git --version
```

Expected: `git version 2.x`.  
If not installed → download from [git-scm.com](https://git-scm.com).

Check available RAM:

```bash
# Linux
free -h

# Windows PowerShell
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory
```

You need at least **4GB free RAM** to run Qwen2.5 3B comfortably.

---

## Step 2 — Install Ollama

Ollama is the tool that runs AI models locally.  
It downloads models, manages them, and runs a local REST API on port 11434.  
Think of it like Docker, but for AI models.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This script installs Ollama as a system service.  
It will start automatically on boot after this.

### Windows

1. Go to [https://ollama.com/download](https://ollama.com/download)
2. Click **Download for Windows**
3. Run the `.exe` installer
4. Follow the installation wizard (next, next, install)
5. Ollama will appear in your system tray after install

### Verify Ollama installed correctly

```bash
ollama --version
```

Expected output:

```
ollama version 0.3.x
```

### Check Ollama service is running

```bash
# Linux
systemctl status ollama

# Windows — check system tray for Ollama icon
# Or open browser and go to:
# http://localhost:11434
```

Browser should show:

```
Ollama is running
```

If not running, start it manually:

```bash
# Linux
systemctl start ollama

# Windows
# Open Ollama from Start Menu
```

---

## Step 3 — Download Qwen2.5 3B model

This downloads the model to your machine.  
File size is about **2.0 GB** — make sure you have space and stable internet.

```bash
ollama pull qwen2.5:3b
```

You will see a live progress bar:

```
pulling manifest
pulling qwen2.5:3b-instruct-q4_K_M... ████████░░ 68% 1.3 GB/2.0 GB
```

Wait until you see:

```
success
```

### Verify the model downloaded

```bash
ollama list
```

Expected output:

```
NAME              ID            SIZE    MODIFIED
qwen2.5:3b        abc123def456  2.0 GB  1 minute ago
```

---

## Step 4 — Test the model works

### Quick test in terminal

```bash
ollama run qwen2.5:3b "Hello. Tell me what you are in one sentence."
```

You should get a response in a few seconds:

```
I am Qwen2.5, a large language model created by Alibaba Cloud...
```

If you see a response — your local LLM is working. 

### Exit the interactive mode

```
/bye
```

### Test via API (this is how your code will talk to it)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "Say hello",
  "stream": false
}'
```

Expected response (JSON):

```json
{
  "model": "qwen2.5:3b",
  "response": "Hello! How can I assist you today?",
  "done": true
}
```

This confirms the API works. Your Python code will use this exact endpoint.

---

## Step 5 — Understand what Ollama just set up

```
Your Machine
│
├── Ollama service (running in background on port 11434)
│   │
│   ├── REST API → http://localhost:11434
│   │   ├── POST /api/generate   ← send prompt, get response
│   │   ├── POST /api/embeddings ← send text, get vector
│   │   └── GET  /api/tags       ← list downloaded models
│   │
│   └── Models stored at:
│       Linux   → ~/.ollama/models/
│       Windows → C:\Users\you\.ollama\models\
```

Your June AI code will call `http://localhost:11434` to talk to the model.  
No internet needed once the model is downloaded.

---

## Step 6 — Set up Python project

### Create project folder

```bash
# Go to where you keep projects
cd ~/Documents        # Linux/Mac
cd C:\Users\you\      # Windows

# Create folder
mkdir june-ai
cd june-ai
```

### Create virtual environment

```bash
python -m venv venv
```

This creates a `venv/` folder with an isolated Python installation.

### Activate virtual environment

```bash
# Linux / Mac
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

Your prompt should now show `(venv)`:

```
(venv) user@machine:~/Documents/june-ai$
```

**Every time you open a new terminal, activate venv first.**  
Otherwise Python won't find your installed packages.

### Upgrade pip

```bash
pip install --upgrade pip
```

---

## Step 7 — Create requirements.txt

Create a file called `requirements.txt` in your `june-ai/` folder.

```txt
# Web framework
fastapi==0.111.0
uvicorn==0.29.0

# Data validation
pydantic==2.7.0
pydantic-settings==2.2.1

# Config
pyyaml==6.0.1
python-dotenv==1.0.1

# HTTP
requests==2.31.0
httpx==0.27.0

# RAG — vector database
chromadb==0.5.0

# RAG — embeddings
sentence-transformers==3.0.0

# LLM API fallback
openai==1.30.0

# Terminal UI
rich==13.7.1

# Voice — uncomment in week 5
# openai-whisper==20231117
```

### Install all packages

```bash
pip install -r requirements.txt
```

This takes **3–5 minutes**.  
`sentence-transformers` downloads PyTorch internally — that's the big one.

### Verify key packages installed

```bash
pip show fastapi
pip show chromadb
pip show sentence-transformers
```

All three should print version info without errors.

---

## Step 8 — Create project folder structure

Run this entire block in your terminal from inside `june-ai/`:

### Linux / Mac

```bash
mkdir -p \
  llm/providers \
  llm/prompts \
  llm/tokenizer \
  rag/loaders \
  rag/chunking \
  rag/embeddings \
  rag/vectordb \
  orchestrator \
  voice/stt \
  voice/tts \
  voice/wakeword \
  voice/streaming \
  memory/short_term \
  memory/long_term \
  memory/semantic \
  memory/storage \
  internet/search \
  internet/scraping \
  internet/apis \
  tools/system \
  tools/productivity \
  tools/browser \
  tools/coding \
  api/routes \
  api/websocket \
  api/auth \
  ui \
  core \
  config \
  storage/vectors \
  storage/conversations \
  storage/cache \
  storage/files/notes \
  storage/files/pdfs \
  logs \
  tests
```

### Windows PowerShell

```powershell
$dirs = @(
  "llm/providers","llm/prompts","llm/tokenizer",
  "rag/loaders","rag/chunking","rag/embeddings","rag/vectordb",
  "orchestrator",
  "voice/stt","voice/tts","voice/wakeword","voice/streaming",
  "memory/short_term","memory/long_term","memory/semantic","memory/storage",
  "internet/search","internet/scraping","internet/apis",
  "tools/system","tools/productivity","tools/browser","tools/coding",
  "api/routes","api/websocket","api/auth",
  "ui","core","config",
  "storage/vectors","storage/conversations","storage/cache",
  "storage/files/notes","storage/files/pdfs",
  "logs","tests"
)
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "All folders created."
```

### Create __init__.py in every module folder

Python needs this file in each folder to treat it as a module.

**Linux / Mac:**

```bash
find . -type d \
  -not -path './venv*' \
  -not -path './.git*' \
  -not -path './storage*' \
  -not -path './logs*' \
  | xargs -I{} touch {}/__init__.py
```

**Windows PowerShell:**

```powershell
Get-ChildItem -Recurse -Directory |
  Where-Object {
    $_.FullName -notmatch 'venv' -and
    $_.FullName -notmatch '\.git' -and
    $_.FullName -notmatch 'storage' -and
    $_.FullName -notmatch 'logs'
  } |
  ForEach-Object {
    New-Item -Force -Path "$($_.FullName)/__init__.py" -ItemType File | Out-Null
  }
Write-Host "All __init__.py files created."
```

### Verify structure looks right

```bash
ls
```

Expected:

```
api/         core/        llm/          orchestrator/   tests/
config/      internet/    logs/         rag/            tools/
memory/      storage/     ui/           voice/
requirements.txt    venv/
```

---

## Step 9 — Create config files

### config/models.yaml

```yaml
# Which provider to use first
active_provider: ollama
model: qwen2.5:3b
ollama_base_url: http://localhost:11434

# Fallback if Ollama is down
fallback_provider: openai
fallback_model: gpt-4o

# Generation settings
max_tokens: 2048
temperature: 0.7
```

### config/rag.yaml

```yaml
chunk_size: 512
chunk_overlap: 50
top_k: 4
embedding_model: all-MiniLM-L6-v2
chroma_path: storage/vectors
chroma_collection: june_knowledge
```

### .env (API keys — never commit this file)

```bash
OPENAI_API_KEY=your_openai_key_here
```

### .gitignore

```
venv/
.env
__pycache__/
*.pyc
storage/vectors/
logs/
.DS_Store
```

---

## Step 10 — Write your first real file

This is `llm/providers/base.py` — the most important file in the project.  
Everything else depends on this interface.

```python
# llm/providers/base.py

from abc import ABC, abstractmethod
from typing import Iterator, List


class BaseLLMProvider(ABC):
    """
    Every LLM provider must implement this interface.
    The orchestrator only knows about this class.
    It never imports OllamaProvider or OpenAIProvider directly.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """
        Single-shot generation.
        Send prompt, get full response string back.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """
        Streaming generation.
        Yields response chunks as they arrive from the model.
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a piece of text.
        Used by RAG for similarity search.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is reachable right now.
        Used by manager to decide which provider to use.
        """
        pass
```

---

## Step 11 — Write OllamaProvider

```python
# llm/providers/ollama.py

import json
import requests
from typing import Iterator, List
from llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Talks to the local Ollama API.
    Runs on http://localhost:11434 by default.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        """Send prompt, wait for full response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Send prompt, yield response chunks as they arrive."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True
        }
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}."
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        payload = {
            "model": self.model,
            "prompt": text
        }
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3
            )
            return response.status_code == 200
        except Exception:
            return False
```

---

## Step 12 — Write LLMManager

```python
# llm/manager.py

from llm.providers.base import BaseLLMProvider
from llm.providers.ollama import OllamaProvider


class LLMManager:
    """
    Reads config and returns the right provider.
    Handles fallback automatically.
    Everything else in the system imports from here.
    Never imports a specific provider directly.
    """

    def __init__(self):
        self._provider: BaseLLMProvider = self._load_provider()

    def _load_provider(self) -> BaseLLMProvider:
        # Try Ollama first (local)
        ollama = OllamaProvider(model="qwen2.5:3b")

        if ollama.is_available():
            print("✓ Using Ollama (local)")
            return ollama

        # Fallback to OpenAI API
        print("⚠ Ollama not available. Falling back to OpenAI API.")
        try:
            from llm.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(model="gpt-4o")
        except Exception as e:
            raise RuntimeError(
                "No LLM provider available. "
                "Start Ollama with: ollama serve"
            ) from e

    def generate(self, prompt: str, system: str = "") -> str:
        return self._provider.generate(prompt, system)

    def stream(self, prompt: str, system: str = ""):
        return self._provider.stream(prompt, system)

    def embed(self, text: str):
        return self._provider.embed(text)

    @property
    def provider_name(self) -> str:
        return self._provider.__class__.__name__


# Global singleton — import this everywhere
llm = LLMManager()
```

---

## Step 13 — Write your first test

```python
# tests/test_llm.py

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
```

### Run the test

```bash
python tests/test_llm.py
```

Expected output:

```
--- Running LLM Tests ---

✓ Using Ollama (local)
✓ Ollama is available
✓ Generate works. Response: JUNE_TEST_OK
✓ Stream works. Chunks: 12, Response: 1 2 3 4 5
✓ Embed works. Vector dimensions: 2048

--- All tests passed ---
```

**If all 4 tests pass — Phase 1 Step 1 is complete.**  
Your LLM layer is working. The foundation is solid.

---

## Common Errors and Fixes

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
curl http://localhost:11434

# If not running, start it
ollama serve
```

### "model not found"

```bash
# Check what models you have
ollama list

# Pull the model again
ollama pull qwen2.5:3b
```

### "ModuleNotFoundError"

```bash
# Make sure venv is activated
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install packages again
pip install -r requirements.txt
```

### "pip: command not found"

```bash
python -m pip install -r requirements.txt
```

### Ollama running slow on CPU

This is expected on your hardware.  
Qwen2.5 3B on CPU takes about **5–15 seconds** per response.  
That is normal. It will feel faster once you stream responses chunk by chunk.

---

## What you have now

```
✅ Ollama installed and running
✅ Qwen2.5 3B downloaded and responding
✅ Python venv set up
✅ All packages installed
✅ Project folder structure created
✅ Config files written
✅ BaseLLMProvider interface written
✅ OllamaProvider implemented
✅ LLMManager with fallback written
✅ All 4 LLM tests passing
```

---

## What comes next

Next session we build **RAG** — the knowledge engine.

```
Step 14 → Write document loaders (pdf, txt, md)
Step 15 → Write chunker
Step 16 → Set up sentence-transformers embedder
Step 17 → Set up ChromaDB
Step 18 → Write retriever
Step 19 → Feed it your actual documents
Step 20 → Test: ask a question, get answer from your own notes
```

---

*June AI — Start Steps v1.0*