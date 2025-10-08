import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import RAG
from order_status import get_status

load_dotenv()
app = FastAPI(title="Voice RAG Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
rag = RAG()

class AskIn(BaseModel):
    text: str
    sessionId: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/order-status")
def order_status(orderId: str):
    return get_status(orderId)

@app.post("/ask")
def ask(inp: AskIn):
    text = inp.text.strip()
    # Simple intent router: "order status" vs RAG fallback
    if any(k in text.lower() for k in ["order status","track order","where is my order","status of order"]):
        # Extract a naive order id
        toks = text.replace("#"," ").split()
        oid = next((t for t in toks if t.isalnum() and len(t) >= 5), "ORD12345")
        st = get_status(oid)
        msg = f"Order {st['orderId']} is **{st['status']}**. ETA {st['etaDays']} day(s)."
        return {"answer": msg + " (from Order Status API)"}
    else:
        ans, _ = rag.answer(text)
        return {"answer": ans}
