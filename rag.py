import os
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import requests
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover
    InferenceClient = None

MODELS_DIR = Path(__file__).parent / "models"


# ------------ env helpers ------------
def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


# ------------ RAG ------------
class RAG:
    """
    Retrieval-Augmented Generation core with provider switch:
      - Ollama (local)
      - Hugging Face Inference API (serverless, free tier)
      - OpenAI (optional)
    """

    def __init__(self):
        # Retrieval setup
        self.embed_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.top_k = env_int("TOP_K", 4)

        self.enc = SentenceTransformer(self.embed_model_name)

        # Load FAISS + chunks
        index_path = MODELS_DIR / "faiss.index"
        chunks_path = MODELS_DIR / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise RuntimeError(
                "FAISS index or chunks.json missing. Run `python ingest.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        # Generation/common settings
        self.temperature = env_float("LLM_TEMPERATURE", 0.2)
        self.max_new_tokens = env_int("LLM_MAX_NEW_TOKENS", 256)

        # Provider switch
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()

        if self.provider == "ollama":
            self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")

        elif self.provider == "hf":
            if InferenceClient is None:
                raise RuntimeError("huggingface-hub not installed. Add it to requirements.txt")
            self.hf_token = os.getenv("HF_API_TOKEN")
            if not self.hf_token:
                raise RuntimeError("HF_API_TOKEN is not set. Get one at https://huggingface.co/settings/tokens")
            self.hf_model = os.getenv("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")
            # Bind model to client once (avoids provider resolution errors)
            self.hf_client = InferenceClient(model=self.hf_model, token=self.hf_token)

        else:  # openai (optional)
            try:
                from openai import OpenAI
            except Exception as e:  # pragma: no cover
                raise RuntimeError("OpenAI client not installed. `pip install openai`") from e
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=api_key)
            self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ---------- Retrieval ----------
    def retrieve(self, query: str) -> List[dict]:
        q = self.enc.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, self.top_k)
        hits = [self.chunks[int(i)] for i in I[0] if i != -1]
        return hits

    def build_prompt(self, query: str, hits: List[dict]) -> Tuple[str, str, str]:
        """
        Returns (system, user, merged_for_hf)
        """
        # Join retrieved context
        ctx = "\n\n".join([f"[{h['title']}] {h['text']}" for h in hits]) if hits else ""
        cite_titles = "; ".join(sorted({h["title"] for h in hits})) if hits else "the knowledge base"

        system = (
            "You are a helpful customer-care assistant. "
            "Use ONLY the provided context to answer. "
            "If the answer isn't in the context, say you don't know. "
            "End your answer with a short citation like (from TITLE)."
        )
        user = f"Question: {query}\n\nContext:\n{ctx}\n\nRemember to end with: (from {cite_titles})"
        merged = f"{system}\n\n{user}"
        return system, user, merged

    # ---------- Providers ----------
    def _ollama_chat(self, system: str, user: str) -> str:
        try:
            url = f"{self.ollama_base}/api/chat"
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "").strip()
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    def _hf_generate(self, prompt_merged: str) -> str:
        try:
            # Use `stop` (not deprecated stop_sequences)
            stop = ["\nUser:", "\nAssistant:", "\nSystem:"]
            text = self.hf_client.text_generation(
                prompt_merged,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop=stop,
                details=False,
                return_full_text=False,
            )
            return (text or "").strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"HuggingFace inference error: {type(e).__name__}: {e}")

    def _openai_chat(self, system: str, user: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    # ---------- Utils ----------
    @staticmethod
    def _ensure_citation(text: str, hits: List[dict]) -> str:
        if "(from" not in text:
            cite = "; ".join(sorted({h["title"] for h in hits})) if hits else "the knowledge base"
            text += f" (from {cite})"
        return text

    # ---------- Public ----------
    def answer(self, query: str) -> Tuple[str, List[dict]]:
        hits = self.retrieve(query)
        system, user, merged = self.build_prompt(query, hits)

        if self.provider == "ollama":
            text = self._ollama_chat(system, user)
        elif self.provider == "hf":
            text = self._hf_generate(merged)
        else:
            text = self._openai_chat(system, user)

        text = text or "Sorry, I couldn't generate a response."
        text = self._ensure_citation(text, hits)
        return text, hits
