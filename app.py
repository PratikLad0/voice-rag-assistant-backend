import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv  # add this

from rag import RAG

# ---- optional: import order status stub from separate file ----
try:
    from order_status import get_status as get_order_status
except Exception:
    # Fallback stub if order_status.py is missing
    def get_order_status(order_id: str):
        import random
        return {
            "orderId": order_id,
            "status": random.choice(
                ["processing", "packed", "shipped", "out for delivery", "delivered", "returned"]
            ),
            "etaDays": random.randint(0, 5),
            "lastUpdated": int(time.time()),
        }


app = FastAPI(title="Voice RAG Assistant", version="1.0.0")

# CORS for local dev & hosted frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG (load index, models, provider)
rag = RAG()


class AskIn(BaseModel):
    text: str
    sessionId: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True, "provider": os.getenv("LLM_PROVIDER", "ollama")}


@app.get("/version")
def version():
    return {
        "app": "Voice RAG Assistant",
        "version": "1.0.0",
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
    }


@app.get("/order-status")
def order_status(orderId: str):
    """
    Mock Order Status endpoint.
    You can call this from the UI, or we auto-use it via the intent router.
    """
    if not orderId or len(orderId) < 3:
        raise HTTPException(status_code=400, detail="orderId is required")
    return get_order_status(orderId)


# ---- Simple intent router: "order status" vs RAG ----
ORDER_KEYWORDS = [
    "order status",
    "track order",
    "where is my order",
    "status of order",
    "order#",
    "order id",
    "order no",
]


def _maybe_extract_order_id(text: str) -> Optional[str]:
    # Naive extraction: take the first alphanumeric token length>=5
    for tok in text.replace("#", " ").split():
        if tok.isalnum() and len(tok) >= 5:
            return tok
    return None


@app.post("/ask")
def ask(inp: AskIn):
    text = (inp.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query")

    lower = text.lower()

    try:
        # Intent: order status (stub)
        if any(k in lower for k in ORDER_KEYWORDS):
            oid = _maybe_extract_order_id(text) or "ORD12345"
            st = get_order_status(oid)
            msg = f"Order {st['orderId']} is **{st['status']}**. ETA {st['etaDays']} day(s)."
            return {"answer": msg + " (from Order Status API)"}

        # Default: RAG answer
        t0 = time.time()
        ans, hits = rag.answer(text)
        latency_ms = int((time.time() - t0) * 1000)
        # (Optional) log to console
        print({"q": text, "provider": os.getenv("LLM_PROVIDER", "ollama"), "ms": latency_ms})
        return {"answer": ans}

    except HTTPException:
        raise
    except Exception as e:
        # Defensive: don't crash the server
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")
