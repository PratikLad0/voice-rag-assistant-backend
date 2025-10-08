# 🎙️ Voice RAG Customer-Care — Backend

FastAPI backend for a **voice-enabled RAG** assistant.
Features:

* **RAG**: FAISS + sentence-transformers + LLM (Ollama / Hugging Face / OpenAI).
* **Intents**: Returns (RAG) and **Order Status** (stub API).
* **Citations** appended to answers, e.g. `(from Returns Policy)`.

> Frontend is separate. This README is **backend-only**.

---

## 📁 Project structure (backend)

```
backend/
├─ app.py               # FastAPI app (routes, intent router)
├─ rag.py               # RAG core with provider switch (ollama/hf/openai)
├─ ingest.py            # Build FAISS index from ./data
├─ order_status.py      # Mock order status API (optional; app has fallback)
├─ data/                # Your source docs (md/txt/json)
├─ models/              # Generated: faiss.index + chunks.json
├─ requirements.txt
└─ .env.example
```

---

## 🔧 Requirements

* Python 3.10+ (tested on 3.10/3.11)
* pip / venv
* For local LLM: **Ollama** running on your machine
* If using HF in hosted env: Hugging Face **Access Token**

---

## ⚙️ Environment variables

Copy and edit:

```bash
cp .env.example .env
```

`.env.example`

```env
# Choose one provider: ollama | hf | openai
LLM_PROVIDER=ollama

# ---- Ollama (local/dev) ----
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b

# ---- Hugging Face Inference API (hosted/free) ----
HF_API_TOKEN=   # e.g. hf_xxxxxxxxx...
HF_MODEL=microsoft/Phi-3-mini-4k-instruct
# other good options:
# HF_MODEL=Qwen/Qwen2.5-3B-Instruct
# HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# ---- OpenAI (optional) ----
OPENAI_API_KEY=
LLM_MODEL=gpt-4o-mini

# ---- Retrieval settings ----
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4

# ---- Generation knobs ----
LLM_TEMPERATURE=0.2
LLM_MAX_NEW_TOKENS=256
```

---

## 📚 Prepare data & index

Put ~10–15 docs in `./data/` (Markdown `.md`, plain `.txt`, or FAQ `.json` as either a list or `{ "faqs": [...] }`).
Then build the index:

```bash
# Windows
python ingest.py

# macOS/Linux
python3 ingest.py
```

This creates `./models/faiss.index` and `./models/chunks.json`.

---

## 🖥️ Run locally

### 1) Create venv & install deps

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Choose a provider

#### Option A — **Ollama** (recommended for local / free)

Install & run Ollama, pull a small model:

* **Windows (PowerShell):**

```powershell
winget install Ollama.Ollama
ollama serve
ollama pull llama3:8b
```

* **macOS:** `brew install ollama && ollama serve && ollama pull llama3:8b`
* **Linux:**

```bash
curl -fsSL https://ollama.com/download/OllamaInstall.sh | sh
ollama serve
ollama pull llama3:8b
```

Set in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b
```

#### Option B — **Hugging Face Inference API** (good for hosting/free tier)

Create a token at **Settings → Access Tokens** on huggingface.co.

Set in `.env`:

```env
LLM_PROVIDER=hf
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
HF_MODEL=Qwen/Qwen2.5-3B-Instruct
```

#### Option C — **OpenAI** (if you have credits)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### 3) Start the server

```bash
uvicorn app:app --reload --port 8000
```

**Health check:** open `http://localhost:8000/health`

---

## 🌐 API Endpoints

* `GET /health` → `{"ok": true, "provider": "ollama|hf|openai"}`
* `GET /version` → app version and provider
* `GET /order-status?orderId=AB12345` → mock JSON status
* `POST /ask`
  **Body**: `{"text":"What's your returns policy?"}`
  **Returns**: `{"answer":"... (from Returns Policy)"}`
  Router:

  * if text contains order keywords (e.g., “track order”, “order status”), returns stubbed status
  * else RAG answer with citation

---

## 🚀 Deploy (backend only)

### Render (free) with **Hugging Face** provider

* Create a **Web Service** → Runtime: **Python**
* **Build Command:**

  ```bash
  pip install -r backend/requirements.txt
  python backend/ingest.py
  ```
* **Start Command:**

  ```bash
  uvicorn backend.app:app --host 0.0.0.0 --port $PORT
  ```
* **Environment:**

  ```
  LLM_PROVIDER=hf
  HF_API_TOKEN=hf_xxx
  HF_MODEL=Qwen/Qwen2.5-3B-Instruct
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  TOP_K=4
  ```

> If you prefer running Ollama in the cloud, you’ll need a **paid** private service with a persistent disk. For free tier, use `hf`.

---

## 🧪 Quick tests

```bash
# Returns FAQ via RAG:
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"What is your returns policy?\"}"

# Order status intent:
curl -s "http://localhost:8000/order-status?orderId=AB12345"
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"What is my order status for #AB12345?\"}"
```

---

## 🛠️ Troubleshooting

* **`RuntimeError: FAISS index or chunks.json missing`**
  Run `python ingest.py` after adding files to `./data`.

* **`ollama: command not found` / not recognized**
  Install Ollama, then open a **new** terminal. Verify: `ollama --version`.
  Ensure `OLLAMA_BASE_URL=http://localhost:11434` and `ollama serve` is running.

* **Hugging Face 500 / StopIteration / deprecation warnings**

  * Ensure `.env` has `LLM_PROVIDER=hf`, valid `HF_API_TOKEN`, valid `HF_MODEL`.
  * We use `stop=` (not deprecated `stop_sequences`).
  * First request may be slow (cold start); try again.

* **OpenAI 429 insufficient_quota**
  Add billing or switch provider to `ollama`/`hf`.

* **CORS issues from browser**
  `app.py` allows `origins="*"` by default. For production, restrict domains.

---

## 🧩 Implementation details

* **Chunking/Embeddings**: managed in `ingest.py` (simple fixed-size chunks; metadata includes `title` for citations).
* **Retrieval**: cosine/IP with FAISS, top-K set by `TOP_K`.
* **Prompting**: system + user with explicit instruction to **always cite**.
* **Providers**:

  * **Ollama**: REST call to `/api/chat` with `temperature` control.
  * **HF**: `InferenceClient.text_generation()` bound to `HF_MODEL`, uses `stop=` and `return_full_text=False`.
  * **OpenAI**: chat completions (optional).

---

## 📄 License

MIT 

---

