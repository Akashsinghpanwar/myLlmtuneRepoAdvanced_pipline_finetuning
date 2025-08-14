# utils/chunk.py
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# -----------------------------
# Config loader (unchanged API)
# -----------------------------
def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from specified config file or default config/config.json."""
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent.parent / "config" / "config.json"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please run summarize.py first to create a default config file.")
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

    # Backward compatible defaults
    cfg.setdefault("chunk", {})
    cfg["chunk"].setdefault("min_length", 1200)
    cfg["chunk"].setdefault("max_length", 3000)
    cfg["chunk"].setdefault("length_unit", "chars")  # "chars" | "tokens"
    cfg["chunk"].setdefault("soft_min_ratio", 0.6)   # flush if current >= soft_min and next would overflow
    return cfg

# -----------------------------
# Utility: token/char length
# -----------------------------
def _estimate_tokens(text: str) -> int:
    # Cheap token proxy: ~1.3 * words
    return max(1, int(len(text.split()) * 1.3))

def _measure(text: str, unit: str) -> int:
    if unit == "tokens":
        return _estimate_tokens(text)
    return len(text)

# -----------------------------------
# Markdown-aware segmentation helpers
# -----------------------------------
# Handles all Markdown alignments: --- | :--- | ---: | :---:
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*(?::?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```(\w+)?\s*$")
_LIST_RE = re.compile(r"^\s*([\-*+]\s+|\d+\.\s+)")

def _strip_trailing_blank(lines: List[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()

def _split_into_sentences(text: str) -> List[str]:
    # Handles common sentence endings, avoids common abbrevs
    abbrevs = r"(e\.g|i\.e|etc|mr|mrs|ms|dr|prof|inc|ltd|vs)\."
    pattern = rf'(?<!\b{abbrevs})(?<!\w\.\w\.)(?<=\.|\?|!)\s+'
    return re.split(pattern, text)

def _is_loose_csv_line(line: str) -> bool:
    # Many commas, not ending with sentence punctuation
    return (line.count(",") >= 3) and not re.search(r"[.!?]\s*$", line)

def _join_lines(lines: List[str]) -> str:
    return "\n".join(lines).strip()

def _consume_fenced_block(i: int, lines: List[str]) -> Tuple[int, Dict[str, Any]]:
    lang = _CODE_FENCE_RE.match(lines[i]).group(1) or ""
    block_lines = [lines[i]]
    i += 1
    while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
        block_lines.append(lines[i])
        i += 1
    if i < len(lines):
        block_lines.append(lines[i])  # closing fence
        i += 1
    kind = "csv" if lang.lower() == "csv" else "code"
    return i, {"type": kind, "text": _join_lines(block_lines), "lang": lang}

def _consume_table(i: int, lines: List[str]) -> Tuple[int, Dict[str, Any]]:
    # Expect header row at i, sep row at i+1; then collect rows while they look like table rows
    tbl_lines = [lines[i]]
    i += 1
    if i < len(lines) and _TABLE_SEP_RE.match(lines[i]):
        tbl_lines.append(lines[i])
        i += 1
        while i < len(lines) and (_TABLE_ROW_RE.match(lines[i]) or not lines[i].strip()):
            if lines[i].strip():
                tbl_lines.append(lines[i])
            i += 1
        return i, {"type": "table", "text": _join_lines(tbl_lines)}
    else:
        # Not a valid markdown table; treat as paragraph
        return i, {"type": "paragraph", "text": lines[i-1].strip()}

def _consume_list(i: int, lines: List[str]) -> Tuple[int, Dict[str, Any]]:
    lst = [lines[i]]
    i += 1
    while i < len(lines) and (_LIST_RE.match(lines[i]) or (lines[i].strip() == "")):
        lst.append(lines[i])
        i += 1
    return i, {"type": "list", "text": _join_lines(lst)}

def _consume_loose_csv(i: int, lines: List[str]) -> Tuple[int, Dict[str, Any]]:
    csv_lines = [lines[i]]
    i += 1
    while i < len(lines) and _is_loose_csv_line(lines[i]):
        csv_lines.append(lines[i])
        i += 1
    return i, {"type": "csv", "text": _join_lines(csv_lines)}

def _parse_blocks_from_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Parse markdown into atomic blocks: heading, paragraph, table, csv, code, list, hr."""
    lines = md_text.splitlines()
    blocks: List[Dict[str, Any]] = []
    i = 0
    para_buf: List[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            blocks.append({"type": "paragraph", "text": _join_lines(para_buf)})
            para_buf = []

    while i < len(lines):
        line = lines[i]

        # Code fence
        if _CODE_FENCE_RE.match(line):
            flush_para()
            i, blk = _consume_fenced_block(i, lines)
            blocks.append(blk)
            continue

        # Headings
        if re.match(r"^\s*#{1,6}\s+", line):
            flush_para()
            blocks.append({"type": "heading", "text": line.strip()})
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^\s*([-*_]\s*){3,}$", line):
            flush_para()
            blocks.append({"type": "hr", "text": line.strip()})
            i += 1
            continue

        # Markdown table
        if _TABLE_ROW_RE.match(line):
            # Peek next line for separator
            if i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i+1]):
                flush_para()
                i, blk = _consume_table(i, lines)
                blocks.append(blk)
                continue

        # Loose CSV run
        if _is_loose_csv_line(line):
            flush_para()
            i, blk = _consume_loose_csv(i, lines)
            blocks.append(blk)
            continue

        # Lists
        if _LIST_RE.match(line):
            flush_para()
            i, blk = _consume_list(i, lines)
            blocks.append(blk)
            continue

        # Blank line => paragraph boundary
        if not line.strip():
            if para_buf and para_buf[-1].strip() == "":
                # collapse multiple blanks
                i += 1
                continue
            para_buf.append("")  # keep one blank to separate paragraphs
            i += 1
            continue

        # Default: accumulate to paragraph buffer
        para_buf.append(line)
        i += 1

    flush_para()
    # Drop trailing blanks in paragraph blocks
    for b in blocks:
        if b["type"] == "paragraph":
            b["text"] = re.sub(r"\n{3,}", "\n\n", b["text"]).strip()
    return [b for b in blocks if b["text"]]

# -----------------------------
# Packing blocks into chunks
# -----------------------------
def _split_paragraph_to_fit(text: str, max_len: int, unit: str) -> List[str]:
    # Split by sentences to fit max; fallback to hard wrap words
    sents = _split_into_sentences(text)
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        slen = _measure(s, unit)
        if cur == 0 or cur + slen + 1 <= max_len:
            buf.append(s)
            cur += slen + 1
        else:
            chunks.append(" ".join(buf).strip())
            buf = [s]
            cur = slen
    if buf:
        chunks.append(" ".join(buf).strip())

    # Rare case: single sentence still too long -> hard wrap by words
    fixed: List[str] = []
    for ch in chunks:
        if _measure(ch, unit) <= max_len:
            fixed.append(ch)
        else:
            words = ch.split()
            tmp: List[str] = []
            acc = 0
            buf2: List[str] = []
            for w in words:
                wl = _measure(w, unit) + 1
                if acc + wl > max_len and buf2:
                    fixed.append(" ".join(buf2))
                    buf2 = [w]
                    acc = _measure(w, unit)
                else:
                    buf2.append(w)
                    acc += wl
            if buf2:
                fixed.append(" ".join(buf2))
    return fixed

def _pack_blocks(blocks: List[Dict[str, Any]], min_len: int, max_len: int, unit: str, soft_min_ratio: float) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    soft_min = int(min_len * soft_min_ratio)

    def flush():
        nonlocal buf, cur
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(text)
        buf = []
        cur = 0

    for b in blocks:
        btxt = b["text"].strip()
        if not btxt:
            continue

        # Atomic blocks (tables/csv/code): never split; allow overflow if needed
        if b["type"] in ("table", "csv", "code"):
            blen = _measure(btxt, unit)
            if cur > 0 and cur >= soft_min and cur + blen + 2 > max_len:
                flush()
            # Add atomically and flush to keep blocks intact
            buf.append(btxt)
            cur += blen + 2
            flush()
            continue

        # Headings: prefer to start a new chunk if current is non-empty
        if b["type"] == "heading":
            if cur >= soft_min:
                flush()
            buf.append(btxt)
            cur += _measure(btxt, unit) + 2
            continue

        # Lists and paragraphs: can be split
        blen = _measure(btxt, unit)
        if blen > max_len:
            parts = _split_paragraph_to_fit(btxt, max_len, unit)
            for p in parts:
                plen = _measure(p, unit)
                if cur > 0 and cur + plen + 2 > max_len and cur >= soft_min:
                    flush()
                buf.append(p)
                cur += plen + 2
        else:
            if cur > 0 and cur + blen + 2 > max_len and cur >= soft_min:
                flush()
            buf.append(btxt)
            cur += blen + 2

    flush()
    # Ensure min length (best-effort): merge last two if final is tiny
    if len(chunks) >= 2 and _measure(chunks[-1], unit) < int(min_len * 0.5):
        last = chunks.pop()
        chunks[-1] = (chunks[-1] + "\n\n" + last).strip()
    return chunks

# -----------------------------
# I/O helpers
# -----------------------------
def _read_source_text(file_path: Path) -> Tuple[str, str]:
    """
    Prefer Markdown if sibling .md exists; else read .txt.
    Returns (text, source_type) where source_type in {"md","txt"}.
    """
    md_candidate = file_path.with_suffix(".md")
    if md_candidate.exists():
        return md_candidate.read_text(encoding="utf-8", errors="ignore"), "md"
    return file_path.read_text(encoding="utf-8", errors="ignore"), "txt"

# -----------------------------
# Main processing
# -----------------------------
def process_file(file_path: Path, output_dir: Path, config: Dict[str, Any], test: bool = False) -> Tuple[int, int]:
    """
    Process a single file and create chunks.
    Returns (#segments_detected, #chunks_written)
    """
    min_length = int(config["chunk"]["min_length"])
    max_length = int(config["chunk"]["max_length"])
    unit = config["chunk"].get("length_unit", "chars")
    soft_min_ratio = float(config["chunk"].get("soft_min_ratio", 0.6))

    try:
        text, source_type = _read_source_text(file_path)

        # Parse to blocks (markdown-aware if md, otherwise treat as paragraph text)
        if source_type == "md":
            blocks = _parse_blocks_from_markdown(text)
        else:
            blocks = [{"type": "paragraph", "text": text}]

        # Count segments (reporting)
        segment_count = sum(1 for b in blocks if b["type"] in ("heading","paragraph","table","csv","code","list"))

        # Pack into chunks
        all_chunks = _pack_blocks(blocks, min_length, max_length, unit, soft_min_ratio)

        if test:
            print(f"\nTest mode: Showing chunks for {file_path.name} (source={source_type})")
            for i, chunk in enumerate(all_chunks[:6]):
                print(f"\n--- Chunk {i+1}/{len(all_chunks)} ({unit}={_measure(chunk, unit)}) ---")
                print(chunk)
                print("---")
            if len(all_chunks) > 6:
                print(f"\n... and {len(all_chunks) - 6} more chunks not shown ...\n")
            return segment_count, len(all_chunks)

        # Write chunks (+ JSONL index)
        file_chunks_dir = output_dir / file_path.stem
        file_chunks_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = file_chunks_dir / "chunks.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for i, chunk in enumerate(all_chunks, start=1):
                chunk_filename = f"chunk_{i:03d}.txt"
                (file_chunks_dir / chunk_filename).write_text(chunk, encoding="utf-8")
                jf.write(json.dumps(
                    {
                        "file": file_path.name,
                        "chunk_id": i,
                        "filename": chunk_filename,
                        "text": chunk
                    },
                    ensure_ascii=False
                ) + "\n")

        # Metadata
        def is_tableish(s: str) -> bool:
            rows = [ln for ln in s.splitlines() if "|" in ln]
            table_rows = sum(1 for r in rows if _TABLE_ROW_RE.match(r))
            fenced = "```csv" in s or "```" in s
            return table_rows >= 2 or fenced

        metadata = {
            "original_file": file_path.name,
            "source_used": source_type,
            "length_unit": unit,
            "min_length": min_length,
            "max_length": max_length,
            "segments_detected": segment_count,
            "num_chunks": len(all_chunks),
            "chunks": [
                {
                    "id": i,
                    "filename": f"chunk_{i:03d}.txt",
                    "length": _measure(ch, unit),
                    "is_tableish": is_tableish(ch),
                } for i, ch in enumerate(all_chunks, start=1)
            ]
        }
        (file_chunks_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"Processed {file_path.name}: {segment_count} segments → {len(all_chunks)} chunks")
        return segment_count, len(all_chunks)

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}", file=sys.stderr)
        return 0, 0

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate high-quality chunks with markdown-aware boundaries.")
    parser.add_argument("--force", action="store_true", help="Regenerate chunks for all files, even if already processed.")
    parser.add_argument("--config", "-c", type=str, help="Path to config file (default: config/config.json)")
    parser.add_argument("--test", action="store_true", help="Test mode: show chunks in terminal without saving files.")
    parser.add_argument("--file", type=str, help="Process only the specified file (relative to data/text).")
    args = parser.parse_args()

    config = load_config(args.config)

    base_dir = Path(__file__).parent.parent
    text_dir = base_dir / "data" / "text"
    chunks_dir = base_dir / "data" / "chunks"

    if not text_dir.exists() or not text_dir.is_dir():
        print(f"Text directory does not exist: {text_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.test:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        if args.force:
            import shutil
            print("Force flag detected. Deleting existing chunks...")
            for item in chunks_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"Deleted: {item.name}")
                elif item.is_file():
                    item.unlink()
                    print(f"Deleted: {item.name}")

    # Select files (prefer .txt list for compatibility, but .md will be read if present)
    if args.file:
        file_path = text_dir / args.file
        if not file_path.exists() or not file_path.is_file():
            print(f"File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        files = [file_path]
    else:
        files = [f for f in text_dir.iterdir() if f.is_file() and f.suffix.lower() in {".txt", ".md"}]

    if not files:
        print("No files to process.")
        return

    if not args.force and not args.test:
        files = [f for f in files if not (chunks_dir / f.stem).exists()]

    if not files:
        print("All files have already been processed. Use --force to regenerate chunks.")
        return

    unit = config["chunk"].get("length_unit", "chars")
    print(f"Found {len(files)} files to process with min={config['chunk']['min_length']} "
          f"max={config['chunk']['max_length']} unit={unit}")

    total_segments = 0
    total_chunks = 0
    for file in files:
        segments, chunks = process_file(file, chunks_dir, config, args.test)
        total_segments += segments
        total_chunks += chunks

    if not args.test:
        print(f"\nChunk generation complete. Processed {total_segments} segments into {total_chunks} chunks.")
    else:
        print(f"\nTest mode completed. Would process {total_segments} segments into {total_chunks} chunks.")

if __name__ == "__main__":
    main()
