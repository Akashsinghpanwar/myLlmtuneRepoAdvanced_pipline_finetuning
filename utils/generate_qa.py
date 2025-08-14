# utils/generate_qa.py
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import concurrent.futures
from tqdm import tqdm

# ===========================
# Pydantic model (validation)
# ===========================
class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The question text")
    evaluation_criteria: str = Field(..., description="Criteria for evaluating the answer")
    answer: str = Field(..., description="The answer to the question")
    difficulty: int = Field(..., ge=1, le=10, description="Difficulty rating from 1-10")
    category: str = Field(..., description="Category of the question (factual|reasoning)")

# ===========================
# JSON Schema to force arrays
# ===========================
QA_JSON_SCHEMA = {
  "type": "json_schema",
  "json_schema": {
    "name": "qa_array",
    "schema": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "question": {"type": "string"},
          "evaluation_criteria": {"type": "string"},
          "answer": {"type": "string"},
          "difficulty": {"type": "integer", "minimum": 1, "maximum": 10},
          "category": {"type": "string", "enum": ["factual", "reasoning"]}
        },
        "required": ["question", "evaluation_criteria", "answer", "difficulty", "category"]
      }
    }
  }
}

# ===========================
# Config & deps
# ===========================
def ensure_dependencies():
    try:
        import requests  # noqa: F401
        import dotenv    # noqa: F401
        from pydantic import BaseModel  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError:
        print("Required dependencies not installed. Installing now...")
        os.system("uv add requests python-dotenv pydantic tqdm")
        print("Please restart the script after installation.")
        sys.exit(1)

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent.parent / "config" / "config.json"
    if not config_path.exists() or not config_path.is_file():
        print(f"Config file not found: {config_path}")
        print("Please ensure a config file exists at config/config.json or pass --config.")
        sys.exit(1)
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

# ===========================
# IO helpers
# ===========================
def get_document_summary(doc_name: str) -> Optional[Dict[str, Any]]:
    path = Path(__file__).parent.parent / "data" / "summaries" / f"{doc_name}_summary.json"
    if not path.exists():
        print(f"Summary not found for {doc_name}. Please run summarize.py first.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error parsing summary file: {e}")
        return None

def get_chunks_for_document(doc_name: str) -> List[Dict[str, Any]]:
    chunks_dir = Path(__file__).parent.parent / "data" / "chunks" / doc_name
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        print(f"Chunks not found for {doc_name}. Please run chunk.py first.")
        return []

    meta_path = chunks_dir / "metadata.json"
    if not meta_path.exists():
        print(f"Metadata file not found for {doc_name}.")
        return []

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata file: {e}")
        return []

    chunks = []
    for info in metadata.get("chunks", []):
        p = chunks_dir / info["filename"]
        if not p.exists():
            print(f"Chunk file not found: {p}")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            chunks.append({
                "id": info["id"],
                "filename": info["filename"],
                "text": text,
                "length": info.get("length", len(text)),
                "is_table": info.get("is_table", info.get("is_tableish", False))
            })
        except Exception as e:
            print(f"Error reading chunk {p}: {e}")
    return chunks

# ===========================
# LLM call
# ===========================
STRICT_SYSTEM_MESSAGE = """
You generate high-quality QA pairs for experts.
Return STRICT JSON ONLY: an array of objects with keys:
question (string), evaluation_criteria (string), answer (string),
difficulty (int 1-10), category ("factual" | "reasoning").
No chain-of-thought, no <think>, no prose, no markdown, no code fences.
All answers must be supported by the provided chunk. Copy exact numbers/units/part codes.
Questions must be self-contained (no “in the document”).
"""

def _post_openrouter(model: str, system: str, user: str, temperature: float, max_tokens: int,
                     top_p: Optional[float], top_k: Optional[int], min_p: Optional[float],
                     api_key: str, response_format: Optional[dict] = None) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if top_p is not None: base_payload["top_p"] = top_p
    if top_k is not None: base_payload["top_k"] = top_k
    if min_p is not None: base_payload["min_p"] = min_p

    # Try with response_format first
    if response_format:
        payload = dict(base_payload)
        payload["response_format"] = response_format
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=180)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"]
        # fall through if it failed

    # Fallback without response_format
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      headers=headers, json=base_payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ===========================
# Parsing & validation
# ===========================
def _json_sanitize(s: str) -> str:
    # strip code fences and whitespace
    s = re.sub(r"^```json", "", s.strip(), flags=re.IGNORECASE|re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    # remove trailing commas before } or ]
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    return s.strip()

def _try_parse_qa_array(s: str) -> List[Dict[str, Any]]:
    s = s.strip()

    # 1) direct array
    try:
        if s.startswith("[") and s.endswith("]"):
            return json.loads(_json_sanitize(s))
    except Exception:
        pass

    # 2) fenced block
    m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL|re.IGNORECASE)
    if m:
        txt = _json_sanitize(m.group(1))
        try:
            return json.loads(txt)
        except Exception:
            pass

    # 3) any array substring
    m2 = re.search(r"\[\s*{.*}\s*\]", s, flags=re.DOTALL)
    if m2:
        txt = _json_sanitize(m2.group(0))
        try:
            return json.loads(txt)
        except Exception:
            pass

    # 4) single object → wrap
    m3 = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m3:
        txt = _json_sanitize(m3.group(0))
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            pass

    # Fallback: empty
    return []

def _normalize_qa_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        ev = (item.get("evaluation_criteria") or "").strip()
        cat = (item.get("category") or "").strip().lower()
        if cat not in {"factual","reasoning"}:
            cat = "factual"
        try:
            diff = int(item.get("difficulty", 5))
        except Exception:
            diff = 5
        diff = max(1, min(10, diff))

        qa = QuestionAnswer(
            question=q, evaluation_criteria=ev, answer=a,
            difficulty=diff, category=cat
        )
        return qa.model_dump()
    except ValidationError:
        return None

def _dedup_by_question(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for qa in pairs:
        key = qa["question"].strip().lower()
        if key in seen: continue
        seen.add(key)
        out.append(qa)
    return out

# ===========================
# Core generation
# ===========================
def generate_qa_for_chunk(summary: Dict[str, Any], chunk: Dict[str, Any], doc_name: str,
                          config: Dict[str, Any], api_key: str,
                          previous_qa_pairs: Optional[List[Dict[str, Any]]] = None,
                          iteration: int = 0) -> Dict[str, Any]:

    model = config["qa"]["model"]
    temperature = config["qa"]["temperature"]
    max_tokens  = config["qa"]["max_tokens"]
    top_p = config["qa"].get("top_p")
    top_k = config["qa"].get("top_k")
    min_p = config["qa"].get("min_p")
    context = config.get("qa", {}).get("context", "")

    user_message = f"""# Context
{context}

# Document Summary
{summary.get('summary', 'No summary available.')}

# Chunk Content (Chunk {chunk['id']})
{chunk['text']}

# Instructions
Create as many QA pairs as needed to cover ALL key info in the chunk.
Balance categories: both factual and reasoning as appropriate.
If tables/units/part-numbers exist, include at least one question requiring exact value/units lookup.
No references like "in the document". Self-contained questions using explicit product/component names.
Return STRICT JSON array only.
"""

    if iteration > 0 and previous_qa_pairs:
        asked = "\n".join([f"- {qa.get('question','')}" for qa in previous_qa_pairs if qa.get("question")])
        user_message += f"""
# Already generated questions (avoid duplicates)
{asked}
"""

    try:
        raw = _post_openrouter(
            model, STRICT_SYSTEM_MESSAGE, user_message,
            temperature, max_tokens, top_p, top_k, min_p,
            api_key, response_format=QA_JSON_SCHEMA
        )
    except Exception as e:
        print(f"Error calling model for chunk {chunk['id']}: {e}")
        return {"chunk_id": chunk["id"], "chunk_filename": chunk["filename"], "qa_pairs": [], "error": str(e)}

    pairs_raw = _try_parse_qa_array(raw)
    pairs_norm = []
    for itm in pairs_raw:
        norm = _normalize_qa_item(itm or {})
        if norm:
            pairs_norm.append(norm)

    # Dedup against previous
    if previous_qa_pairs:
        prev_set = { (qa["question"].strip().lower()) for qa in previous_qa_pairs if qa.get("question") }
        pairs_norm = [qa for qa in pairs_norm if qa["question"].strip().lower() not in prev_set]

    pairs_norm = _dedup_by_question(pairs_norm)

    return {
        "chunk_id": chunk["id"],
        "chunk_filename": chunk["filename"],
        "model": model,
        "qa_pairs": pairs_norm,
        "raw_response": raw,
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p, "top_k": top_k, "min_p": min_p
        }
    }

# ===========================
# Per-document runner
# ===========================
def process_document(doc_name: str, config: Dict[str, Any], api_key: str,
                     test: bool = False, force: bool = False, iterate: bool = False) -> bool:
    summary = get_document_summary(doc_name)
    if not summary: return False

    chunks = get_chunks_for_document(doc_name)
    if not chunks: return False

    qa_base_dir = Path(config.get("qa", {}).get("output_dir", Path(__file__).parent.parent / "data" / "qa"))
    qa_dir = qa_base_dir / doc_name
    qa_dir.mkdir(parents=True, exist_ok=True)

    qa_meta_path = qa_dir / "qa_metadata.json"
    if qa_meta_path.exists() and not force:
        print(f"Q&A already generated for {doc_name}. Use --force to regenerate.")
        return True

    if force and qa_dir.exists():
        print(f"Deleting existing Q&A files for {doc_name}...")
        for f in qa_dir.glob("chunk_*_qa*.json"): f.unlink(missing_ok=True)
        qa_meta_path.unlink(missing_ok=True)
        (qa_dir / "qa.jsonl").unlink(missing_ok=True)

    if test:
        chunks = chunks[:2]
        print(f"Test mode: Processing only the first two chunks for {doc_name}")

    use_batching = config.get("qa", {}).get("use_batching", True)
    batch_size   = config.get("qa", {}).get("batch_size", 16)
    if config.get("qa", {}).get("skip_tables", False):
        before = len(chunks)
        chunks = [c for c in chunks if not c.get("is_table", False)]
        print(f"Skipping table chunks: {before - len(chunks)} skipped")

    max_iterations = config.get("qa", {}).get("max_iterations", 3) if iterate else 1

    qa_results: List[Dict[str, Any]] = []

    def _process_single(chunk, iteration=0, prev=None):
        print(f"Generating Q&A for {doc_name}, chunk {chunk['id']}/{len(chunks)}"
              f"{f', iteration {iteration+1}/{max_iterations}' if iterate else ''}")
        res = generate_qa_for_chunk(summary, chunk, doc_name, config, api_key, prev, iteration)
        out_path = qa_dir / (f"chunk_{chunk['id']:03d}_qa_iter_{iteration+1}.json" if iterate else f"chunk_{chunk['id']:03d}_qa.json")
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Generated {len(res.get('qa_pairs', []))} Q&A pairs for chunk {chunk['id']}")
        return res

    if iterate:
        print(f"Iterative mode enabled: up to {max_iterations} iterations per chunk")
        for ch in chunks:
            prev_pairs: List[Dict[str, Any]] = []
            all_iters: List[Dict[str, Any]] = []
            total = 0
            for it in range(max_iterations):
                r = _process_single(ch, iteration=it, prev=prev_pairs)
                all_iters.append(r)
                new_pairs = r.get("qa_pairs", [])
                total += len(new_pairs)
                prev_pairs.extend(new_pairs)
                print(f"Iteration {it+1}/{max_iterations} for chunk {ch['id']}: +{len(new_pairs)} (Total: {total})")
                if it > 0 and len(new_pairs) == 0:
                    print(f"No new QAs in iteration {it+1}; stopping for chunk {ch['id']}.")
                    break
            combined = {
                "chunk_id": ch["id"],
                "chunk_filename": ch["filename"],
                "model": config["qa"]["model"],
                "qa_pairs": [qa for r in all_iters for qa in r.get("qa_pairs", [])],
                "iterations": len(all_iters),
                "parameters": all_iters[0].get("parameters", {}) if all_iters else {}
            }
            (qa_dir / f"chunk_{ch['id']:03d}_qa_combined.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
            qa_results.append(combined)
    else:
        if use_batching and len(chunks) > 1:
            print(f"Processing {len(chunks)} chunks in parallel (batch size={batch_size})")
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
                fut = {ex.submit(_process_single, ch): ch for ch in chunks}
                for f in tqdm(concurrent.futures.as_completed(fut), total=len(chunks), desc="Processing chunks"):
                    try:
                        qa_results.append(f.result())
                    except Exception as e:
                        ch = fut[f]
                        print(f"Error processing chunk {ch['id']}: {e}")
                        qa_results.append({"chunk_id": ch["id"], "chunk_filename": ch["filename"], "qa_pairs": [], "error": str(e)})
        else:
            print(f"Processing {len(chunks)} chunks sequentially")
            for ch in chunks:
                qa_results.append(_process_single(ch))

    # Write per-document JSONL (dedup by question)
    qa_jsonl = qa_dir / "qa.jsonl"
    with qa_jsonl.open("w", encoding="utf-8") as jf:
        seen = set()
        for r in qa_results:
            pairs = r.get("qa_pairs", [])
            for qa in pairs:
                key = (r.get("chunk_id"), qa["question"].strip().lower())
                if key in seen: 
                    continue
                seen.add(key)
                jf.write(json.dumps({
                    "doc": doc_name,
                    "chunk_id": r.get("chunk_id"),
                    **qa
                }, ensure_ascii=False) + "\n")

    # Metadata
    meta = {
        "document": doc_name,
        "num_chunks_processed": len(chunks),
        "num_qa_pairs_total": sum(len(r.get("qa_pairs", [])) for r in qa_results),
        "model": config["qa"]["model"],
        "temperature": config["qa"]["temperature"],
        "test_mode": test,
        "parallel_processing": (not iterate) and use_batching,
        "batch_size": batch_size if (not iterate) and use_batching else 1,
        "iterative_mode": iterate,
        "max_iterations": max_iterations if iterate else 1
    }
    (qa_dir / "qa_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Q&A generation complete for {doc_name}.")
    print(f"Generated {meta['num_qa_pairs_total']} Q&A pairs across {meta['num_chunks_processed']} chunks.")
    return True

# ===========================
# CLI
# ===========================
def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs for document chunks.")
    parser.add_argument("--config", "-c", type=str, help="Path to config file (default: config/config.json)")
    parser.add_argument("--force", action="store_true", help="Regenerate Q&A pairs even if already processed.")
    parser.add_argument("--test", action="store_true", help="Test mode: only process the first two chunks of each document.")
    parser.add_argument("--doc", type=str, help="Process only the specified document (stem name without extension).")
    parser.add_argument("--output-dir", type=str, help="Custom output directory for Q&A pairs.")
    parser.add_argument("--iterate", action="store_true", help="Enable iterative question generation for each chunk.")
    args = parser.parse_args()

    ensure_dependencies()
    load_dotenv()
    config = load_config(args.config)

    if args.output_dir:
        config.setdefault("qa", {})["output_dir"] = args.output_dir

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Add to .env: OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

    base_dir = Path(__file__).parent.parent
    chunks_dir = base_dir / "data" / "chunks"

    if not chunks_dir.exists() or not chunks_dir.is_dir():
        print(f"Chunks directory does not exist: {chunks_dir}")
        print("Please run chunk.py first.")
        sys.exit(1)

    if args.doc:
        doc_dirs = [chunks_dir / args.doc]
        if not doc_dirs[0].exists() or not doc_dirs[0].is_dir():
            print(f"Document chunks not found: {doc_dirs[0]}")
            sys.exit(1)
    else:
        doc_dirs = [d for d in chunks_dir.iterdir() if d.is_dir()]

    if not doc_dirs:
        print("No document chunks found.")
        return

    if not args.force:
        qa_base_dir = Path(config.get("qa", {}).get("output_dir", base_dir / "data" / "qa"))
        doc_dirs = [d for d in doc_dirs if not (qa_base_dir / d.name / "qa_metadata.json").exists()]

    if not doc_dirs:
        print("All documents have already been processed. Use --force to regenerate.")
        return

    print(f"Found {len(doc_dirs)} document(s) to process with model={config['qa']['model']}")
    for d in doc_dirs:
        process_document(d.name, config, api_key, args.test, args.force, args.iterate)

if __name__ == "__main__":
    main()
