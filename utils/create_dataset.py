# utils/create_dataset.py
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
QA_ROOT = ROOT / "data" / "qa"
CHUNKS_ROOT = ROOT / "data" / "chunks"
SUMM_ROOT = ROOT / "data" / "summaries"
OUT = ROOT / "data" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)

def load_chunks(doc):
    base = CHUNKS_ROOT / doc
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    id2text = {}
    for ch in meta["chunks"]:
        t = (base / ch["filename"]).read_text(encoding="utf-8", errors="ignore")
        id2text[ch["id"]] = t
    return id2text

def load_summary(doc):
    p = SUMM_ROOT / f"{doc}_summary.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            # accept either "abstract" or "summary"
            return d.get("abstract") or d.get("summary") or ""
        except: pass
    return ""

def iter_qa():
    for doc_dir in QA_ROOT.iterdir():
        if not doc_dir.is_dir(): continue
        qaj = doc_dir / "qa.jsonl"
        if not qaj.exists(): continue
        id2text = load_chunks(doc_dir.name)
        summary = load_summary(doc_dir.name)
        with qaj.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                ex["_doc"] = doc_dir.name
                ex["_chunk_text"] = id2text.get(ex["chunk_id"], "")
                ex["_summary"] = summary
                yield ex

rag = OUT / "axess_rag.jsonl"
inst = OUT / "axess_instruct.jsonl"
chat = OUT / "axess_chat.jsonl"
rag.unlink(missing_ok=True); inst.unlink(missing_ok=True); chat.unlink(missing_ok=True)

with rag.open("w", encoding="utf-8") as fr, \
     inst.open("w", encoding="utf-8") as fi, \
     chat.open("w", encoding="utf-8") as fc:
    for ex in iter_qa():
        # RAG: (question + chunk context) -> answer
        fr.write(json.dumps({
            "input": {"question": ex["question"], "context": ex["_chunk_text"]},
            "output": ex["answer"],
            "meta": {"doc": ex["_doc"], "chunk_id": ex["chunk_id"],
                     "category": ex.get("category"), "difficulty": ex.get("difficulty")}
        }, ensure_ascii=False) + "\n")

        # Instruct: single-turn instruction style (uses doc summary as system hint)
        fi.write(json.dumps({
            "messages": [
                {"role":"system","content": f"Axess knowledge base (summary): {ex['_summary'][:1200]}"},
                {"role":"user","content": ex["question"]}
            ],
            "response": ex["answer"],
            "meta": {"doc": ex["_doc"], "chunk_id": ex["chunk_id"],
                     "category": ex.get("category"), "difficulty": ex.get("difficulty")}
        }, ensure_ascii=False) + "\n")

        # Chat: simple two-turn
        fc.write(json.dumps({
            "messages": [
                {"role":"user","content": ex["question"]},
                {"role":"assistant","content": ex["answer"]}
            ],
            "meta": {"doc": ex["_doc"], "chunk_id": ex["chunk_id"],
                     "category": ex.get("category"), "difficulty": ex.get("difficulty")}
        }, ensure_ascii=False) + "\n")

print("Wrote:", rag, inst, chat)
