import os, json, glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def load_docs():
    docs = []
    for p in DATA_DIR.glob("**/*"):
        if p.suffix.lower() in {".md", ".txt"}:
            text = p.read_text(encoding="utf-8")
            docs.append({"id": str(p), "title": p.stem, "text": text})
        elif p.suffix.lower() == ".json":
            j = json.loads(p.read_text(encoding="utf-8"))
            # Expect either [{"title":..., "text":...}, ...] or {"faqs":[...]}
            items = j.get("faqs", j if isinstance(j, list) else [])
            for i, it in enumerate(items):
                docs.append({
                    "id": f"{p}#{i}",
                    "title": it.get("title") or it.get("question","FAQ"),
                    "text": it.get("text") or it.get("answer","")
                })
    return docs

def chunk(text, max_tokens=500, overlap=100):
    # Simple char-based chunking (good enough for a small FAQ)
    max_chars = max_tokens*4
    ov_chars = overlap*4
    out = []
    i=0
    while i < len(text):
        out.append(text[i:i+max_chars])
        i += max(1, (max_chars - ov_chars))
    return out

def main():
    model_name = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    enc = SentenceTransformer(model_name)
    raw_docs = load_docs()

    chunks = []
    for d in raw_docs:
        for idx, c in enumerate(chunk(d["text"])):
            chunks.append({
                "doc_id": d["id"],
                "title": d["title"],
                "chunk_id": f'{d["id"]}:::{idx}',
                "text": c
            })

    X = enc.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(MODELS_DIR/"faiss.index"))
    with open(MODELS_DIR/"chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Ingested {len(raw_docs)} docs into {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
