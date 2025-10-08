import os, json
import faiss, numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

class RAG:
    def __init__(self):
        self.embed_model_name = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
        self.top_k = int(os.getenv("TOP_K","4"))
        self.enc = SentenceTransformer(self.embed_model_name)
        self.index = faiss.read_index(str(MODELS_DIR/"faiss.index"))
        self.chunks = json.loads((MODELS_DIR/"chunks.json").read_text(encoding="utf-8"))

        # LLM provider switch
        self.provider = os.getenv("LLM_PROVIDER","ollama").lower()
        if self.provider == "ollama":
            self.ollama_base = os.getenv("OLLAMA_BASE_URL","http://localhost:11434").rstrip("/")
            self.ollama_model = os.getenv("OLLAMA_MODEL","llama3:8b")
        else:
            # Fallback to OpenAI if you want both paths available
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm_model = os.getenv("LLM_MODEL","gpt-4o-mini")

    def retrieve(self, query: str):
        q = self.enc.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, self.top_k)
        hits = [self.chunks[int(i)] for i in I[0] if i != -1]
        return hits

    def build_prompt(self, query: str, hits):
        ctx = "\n\n".join([f"[{h['title']}] {h['text']}" for h in hits])
        cite = "; ".join(sorted({h['title'] for h in hits})) or "the knowledge base"
        system = ("You are a helpful customer-care assistant. "
                  "Answer using ONLY the provided context. "
                  "If the answer isn't in context, say you don't know. "
                  "End your answer with a short citation like (from TITLE).")
        user = f"Question: {query}\n\nContext:\n{ctx}\n\nRemember to end with: (from {cite})"
        return system, user

    def _ollama_chat(self, system: str, user: str) -> str:
        """
        Calls Ollama's /api/chat endpoint (non-OpenAI API).
        """
        url = f"{self.ollama_base}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # You can tune these:
            "options": {"temperature": 0.2}
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Non-streaming response: final message is in data["message"]["content"]
        return data.get("message", {}).get("content", "").strip()

    def _openai_chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()

    def _ensure_citation(self, text: str, hits) -> str:
        if "(from" not in text:
            cite = "; ".join(sorted({h['title'] for h in hits})) or "the knowledge base"
            text += f" (from {cite})"
        return text

    def answer(self, query: str):
        hits = self.retrieve(query)
        system, user = self.build_prompt(query, hits)
        if self.provider == "ollama":
            text = self._ollama_chat(system, user)
        else:
            text = self._openai_chat(system, user)
        text = self._ensure_citation(text, hits)
        return text, hits
