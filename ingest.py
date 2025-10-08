# backend/ingest.py
import os, json
import numpy as np
import faiss
from pathlib import Path
from fastembed import TextEmbedding  # <— lightweight, no torch

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

def load_docs():
    docs = []
    for p in DATA_DIR.glob("**/*"):
        if p.suffix.lower() in {".md", ".txt"}:
            text = p.read_text(encoding="utf-8")
            docs.append({"id": str(p), "title": p.stem, "text": text})
        elif p.suffix.lower() == ".json":
            j = json.loads(p.read_text(encoding="utf-8"))
            items = j.get("faqs", j if isinstance(j, list) else [])
            for i, it in enumerate(items):
                docs.append({
                    "id": f"{p}#{i}",
                    "title": it.get("title") or it.get("question","FAQ"),
                    "text": it.get("text") or it.get("answer","")
                })
    return docs

def chunk(text, max_chars=2000, overlap=200):
    out, i = [], 0
    step = max_chars - overlap
    while i < len(text):
        out.append(text[i:i+max_chars])
        i += max(1, step)
    return out

def main():
    docs = load_docs()

    chunks = []
    for d in docs:
        for idx, c in enumerate(chunk(d["text"])):
            chunks.append({
                "doc_id": d["id"],
                "title": d["title"],
                "chunk_id": f'{d["id"]}:::{idx}',
                "text": c
            })

    # FastEmbed — small RAM, CPU-only, downloads ONNX on first run
    embedder = TextEmbedding(model_name=EMBED_MODEL)  # e.g., BAAI/bge-small-en-v1.5
    # embed returns a generator; collect to list
    vectors = list(embedder.embed([c["text"] for c in chunks]))
    X = np.array(vectors, dtype="float32")

    # Normalize for inner-product similarity
    faiss.normalize_L2(X)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    faiss.write_index(index, str(MODELS_DIR/"faiss.index"))
    (MODELS_DIR/"chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ingested {len(docs)} docs into {len(chunks)} chunks using {EMBED_MODEL}.")

if __name__ == "__main__":
    main()
