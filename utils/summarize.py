# utils/summarize.py
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv

# -------------------------
# JSON Schemas for strict outputs
# -------------------------
DOC_JSON_SCHEMA = {
  "type":"json_schema",
  "json_schema":{"name":"doc_summary","schema":{
    "type":"object","additionalProperties":False,
    "properties":{
      "abstract":{"type":"string"},
      "key_points":{"type":"array","items":{"type":"string"}},
      "entities":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{"type":{"type":"string"},"value":{"type":"string"}},
        "required":["type","value"]
      }},
      "tables_index":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "properties":{"title":{"type":"string"},"columns":{"type":"array","items":{"type":"string"}}},
        "required":["title","columns"]
      }},
      "numbers_and_units":{"type":"array","items":{"type":"string"}}
    },
    "required":["abstract","key_points","numbers_and_units"]
}}}

CHUNK_JSON_SCHEMA = {
  "type":"json_schema",
  "json_schema":{"name":"chunk_summary","schema":{
    "type":"object","additionalProperties":False,
    "properties":{
      "title":{"type":"string"},
      "bullets":{"type":"array","items":{"type":"string"}},
      "keywords":{"type":"array","items":{"type":"string"}},
      "numbers_and_units":{"type":"array","items":{"type":"string"}}
    },
    "required":["title","bullets","numbers_and_units"]
}}}

# -------------------------
# Config & env
# -------------------------
def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent.parent / "config" / "config.json"

    if not config_path.exists():
        # minimal default
        config_path.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "summary": {
                "model": "openai/gpt-5",
                "temperature": 0.2,
                "max_tokens": 1200
            }
        }
        config_path.write_text(json.dumps(default, indent=2), encoding="utf-8")
        return default

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}", file=sys.stderr)
        sys.exit(1)

def get_openrouter_config(cfg: Dict[str, Any], cli_model: Optional[str]) -> Tuple[str, float, int]:
    model = cli_model or os.getenv("OPENROUTER_MODEL") or cfg.get("summary", {}).get("model") or "openai/gpt-5"
    temperature = float(cfg.get("summary", {}).get("temperature", 0.2))
    max_tokens = int(cfg.get("summary", {}).get("max_tokens", 1200))
    return model, temperature, max_tokens

# -------------------------
# IO helpers
# -------------------------
def prefer_md_or_txt(path: Path) -> Tuple[str, str]:
    """If sibling .md exists use it, else read .txt. Returns (text, source_type)."""
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8", errors="ignore"), "md"
    md = path.with_suffix(".md")
    if md.exists():
        return md.read_text(encoding="utf-8", errors="ignore"), "md"
    return path.read_text(encoding="utf-8", errors="ignore"), "txt"

def iter_chunk_files(chunks_root: Path) -> List[Path]:
    files: List[Path] = []
    if not chunks_root.exists():
        return files
    for doc_dir in sorted([p for p in chunks_root.iterdir() if p.is_dir()]):
        for f in sorted(doc_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in {".txt", ".md"} and f.name.startswith("chunk_"):
                files.append(f)
    return files

# -------------------------
# LLM call (OpenRouter)
# -------------------------
def call_openrouter(system: str, user_payload: Any, model: str,
                    temperature: float, max_tokens: int, api_key: str,
                    response_format: Optional[dict] = None) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai"
    }
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    if response_format:
        body["response_format"] = response_format

    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def parse_json_strict(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

# -------------------------
# Prompts (strict JSON only)
# -------------------------
DOC_SYSTEM = (
    "You are an extractive, precise technical summarizer. "
    "Output STRICT JSON only. No markdown, no commentary. "
    "Preserve numbers, units, and part codes exactly. If unsure, use empty strings/arrays."
)

CHUNK_SYSTEM = (
    "You create concise, extractive section metadata for training datasets. "
    "Output STRICT JSON only. No markdown, no commentary. "
    "Do not invent facts; preserve numbers/units exactly."
)

def make_doc_payload(text: str) -> Dict[str, Any]:
    text = text[:120_000]
    return {
        "task": "document_summary",
        "rules": [
            "Do NOT hallucinate.",
            "Preserve all numbers/units/part codes exactly.",
            "Prefer short verbatim quotes when listing key_points.",
            "If a field doesn't apply, return an empty string/array."
        ],
        "schema": {
            "abstract": "string",
            "key_points": ["string"],
            "entities": [{"type": "string", "value": "string"}],
            "tables_index": [{"title": "string", "columns": ["string"]}],
            "numbers_and_units": ["string"]
        },
        "text": text
    }

def make_chunk_payload(text: str) -> Dict[str, Any]:
    text = text[:60_000]
    return {
        "task": "chunk_summary",
        "rules": [
            "Title must be short and factual.",
            "Bullets must be extractive (no rewording of numbers/units).",
            "Keywords are 3-8 lower-case tags."
        ],
        "schema": {
            "title": "string",
            "bullets": ["string", "string", "string"],
            "keywords": ["string"],
            "numbers_and_units": ["string"]
        },
        "text": text
    }

# -------------------------
# Validators (lightweight)
# -------------------------
def numbers_exist_in_text(numbers: List[str], text: str) -> int:
    cnt = 0
    lo = text.lower()
    for n in numbers:
        s = n.strip().lower()
        if not s:
            continue
        if s in lo:
            cnt += 1
    return cnt

# -------------------------
# Main runners
# -------------------------
def summarize_doclevel(text_dir: Path, out_dir: Path, cfg: Dict[str, Any], force: bool,
                       model: str, temp: float, max_toks: int, api_key: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in text_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    if not candidates:
        print("No documents found in data/text.", file=sys.stderr)
        return

    print(f"Summarizing {len(candidates)} document(s) → {out_dir}")
    for p in candidates:
        out_json = out_dir / f"{p.stem}_summary.json"
        if out_json.exists() and not force:
            print(f"Skipping (exists): {out_json.name}")
            continue

        text, _ = prefer_md_or_txt(p)
        try:
            raw = call_openrouter(DOC_SYSTEM, make_doc_payload(text), model, temp, max_toks, api_key,
                                  response_format=DOC_JSON_SCHEMA)
            obj = parse_json_strict(raw)

            found = numbers_exist_in_text(obj.get("numbers_and_units", []), text)
            obj["_validation"] = {"numbers_found_in_text": found}

            out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✓ {p.name} → {out_json.name}")
        except Exception as e:
            print(f"✖ Failed {p.name}: {e}", file=sys.stderr)
            (out_dir / f"{p.stem}_summary.error.txt").write_text(str(e), encoding="utf-8")

def summarize_chunklevel(chunks_root: Path, out_root: Path, cfg: Dict[str, Any], force: bool,
                         model: str, temp: float, max_toks: int, api_key: str):
    if not chunks_root.exists():
        print(f"Chunks directory not found: {chunks_root}", file=sys.stderr)
        sys.exit(1)

    files = iter_chunk_files(chunks_root)
    if not files:
        print("No chunk files found. Run the chunker first.", file=sys.stderr)
        return

    print(f"Summarizing {len(files)} chunk(s) → {out_root}")
    for f in files:
        doc_out_dir = out_root / f.parent.name
        doc_out_dir.mkdir(parents=True, exist_ok=True)
        out_json = doc_out_dir / (f.stem + ".json")

        if out_json.exists() and not force:
            continue

        text, _ = prefer_md_or_txt(f)
        try:
            raw = call_openrouter(CHUNK_SYSTEM, make_chunk_payload(text), model, temp, max_toks, api_key,
                                  response_format=CHUNK_JSON_SCHEMA)
            obj = parse_json_strict(raw)

            found = numbers_exist_in_text(obj.get("numbers_and_units", []), text)
            obj["_validation"] = {"numbers_found_in_text": found, "source_chunk": f.name}

            out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            (doc_out_dir / (f.stem + ".error.txt")).write_text(str(e), encoding="utf-8")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Summarize docs or chunks into strict JSON (for finetune dataset assembly).")
    parser.add_argument("--config", "-c", type=str, help="Path to config file (default: config/config.json)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist.")
    parser.add_argument("--model", type=str, help="Override model id (else uses config/env).")
    parser.add_argument("--level", choices=["doc", "chunk"], default="doc", help="Summarize whole documents or per chunk.")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY missing in environment (.env).", file=sys.stderr)
        sys.exit(1)

    model, temp, max_toks = get_openrouter_config(cfg, args.model)

    base = Path(__file__).parent.parent
    text_dir = base / "data" / "text"
    chunks_dir = base / "data" / "chunks"
    summaries_dir = base / "data" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if args.level == "doc":
        summarize_doclevel(text_dir, summaries_dir, cfg, args.force, model, temp, max_toks, api_key)
    else:
        out_root = summaries_dir / "chunks"
        summarize_chunklevel(chunks_dir, out_root, cfg, args.force, model, temp, max_toks, api_key)

    print("Done.")

if __name__ == "__main__":
    main()
