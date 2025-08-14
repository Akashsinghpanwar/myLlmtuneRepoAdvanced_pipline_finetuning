# utils/ingest.py
import argparse
from pathlib import Path
import sys
import os
import json
from typing import Optional, List

import pandas as pd

# Azure DI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Optional .env loader (safe if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CLIENT: Optional[DocumentIntelligenceClient] = None
MODEL_ID = "prebuilt-layout"

# ---------- Helpers ----------
def _get_content_type_for_suffix(suffix: str) -> str:
    s = suffix.lower()
    if s == ".pdf":
        return "application/pdf"
    if s == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if s == ".doc":
        return "application/msword"
    if s == ".txt":
        return "text/plain"
    if s == ".md":
        return "text/markdown"
    if s == ".csv":
        return "text/csv"
    if s == ".json":
        return "application/json"
    return "application/octet-stream"

def _analyze_with_azure(path: Path):
    if CLIENT is None:
        raise RuntimeError("Azure client not initialized.")
    with path.open("rb") as f:
        body = f.read()
    poller = CLIENT.begin_analyze_document(
        model_id=MODEL_ID,
        body=body,
        content_type=_get_content_type_for_suffix(path.suffix),
        features=["keyValuePairs"],  # grab KV pairs too; harmless if none
    )
    return poller.result()

def _extract_paragraph_lines(result) -> List[str]:
    lines: List[str] = []
    for p in (getattr(result, "paragraphs", None) or []):
        txt = getattr(p, "content", None)
        if txt:
            lines.append(" ".join(str(txt).split()))
    return lines

def _extract_key_values(result) -> List[dict]:
    out = []
    for kv in (getattr(result, "key_value_pairs", None) or []):
        key_txt = getattr(getattr(kv, "key", None), "content", "") if getattr(kv, "key", None) else ""
        val_txt = getattr(getattr(kv, "value", None), "content", "") if getattr(kv, "value", None) else ""
        if key_txt or val_txt:
            out.append({"key": key_txt.strip(), "value": val_txt.strip()})
    return out

def _di_table_to_dataframe(table) -> pd.DataFrame:
    nrows = getattr(table, "row_count", 0) or 0
    ncols = getattr(table, "column_count", 0) or 0
    grid = [[""] * ncols for _ in range(nrows)]
    for cell in getattr(table, "cells", []) or []:
        r = getattr(cell, "row_index", 0) or 0
        c = getattr(cell, "column_index", 0) or 0
        txt = (getattr(cell, "content", "") or "").replace("\n", " ").strip()
        grid[r][c] = (grid[r][c] + " " + txt).strip() if grid[r][c] else txt
    return pd.DataFrame(grid)

def _is_numeric(s: str) -> bool:
    if s is None:
        return False
    st = str(s).strip().replace(",", "")
    if not st:
        return False
    try:
        float(st)
        return True
    except Exception:
        return False

def _sanitize_md(text: Optional[str]) -> str:
    if text is None: return ""
    return str(text).replace("|", r"\|").strip()

def _df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    header = [ _sanitize_md(x) for x in df.iloc[0].tolist() ]
    body = df.iloc[1:] if len(df) > 1 else pd.DataFrame(columns=df.columns)
    if all(h == "" for h in header):
        header = [f"Col {i+1}" for i in range(df.shape[1])]
        body = df

    sample = body if not body.empty else df
    aligns = []
    for col in range(sample.shape[1]):
        vals = sample.iloc[:, col].astype(str).tolist()
        numc = sum(1 for v in vals if _is_numeric(v))
        aligns.append("---:" if numc >= max(2, int(0.6 * len(vals))) else "---")

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(aligns) + " |")
    for _, row in body.iterrows():
        cells = [ _sanitize_md(x) for x in row.tolist() ]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ---------- Extractors ----------
def extract_text_plain(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_with_azure(path: Path):
    """
    Returns:
      text_str: plain text (paragraphs joined by newline)
      md_str:   structured markdown (title, counts, KV, text, tables)
      json_obj: dict with file, counts, paragraphs, key_values, tables (as rows)
    """
    result = _analyze_with_azure(path)

    # counts
    pages = len(getattr(result, "pages", []) or [])
    tables = getattr(result, "tables", None) or []
    paragraphs = _extract_paragraph_lines(result)
    kv_pairs = _extract_key_values(result)

    # plain text
    text_str = "\n".join(paragraphs)

    # markdown
    md_parts: List[str] = []
    md_parts.append(f"# {path.name}")
    md_parts.append("")
    md_parts.append(f"- **Pages detected:** {pages}")
    md_parts.append(f"- **Tables detected:** {len(tables)}")
    if kv_pairs:
        md_parts.append(f"- **Key/Value pairs detected:** yes")
    md_parts.append("")

    if kv_pairs:
        md_parts.append("## Key–Value Pairs")
        md_parts.append("")
        md_parts.append("| Key | Value |")
        md_parts.append("| --- | --- |")
        for kv in kv_pairs:
            md_parts.append(f"| {_sanitize_md(kv['key'])} | {_sanitize_md(kv['value'])} |")
        md_parts.append("")

    md_parts.append("## Extracted Text")
    md_parts.append("")
    md_parts.append(text_str if text_str.strip() else "_No text paragraphs found._")
    md_parts.append("")

    table_json_list = []
    if tables:
        md_parts.append("## Tables")
        md_parts.append("")
        for i, t in enumerate(tables, start=1):
            df = _di_table_to_dataframe(t)
            md_parts.append(f"### Table {i}")
            md_parts.append("")
            md_parts.append(_df_to_markdown(df))
            md_parts.append("")
            # add to JSON structure too
            table_json_list.append({
                "index": i,
                "rows": df.astype(str).fillna("").values.tolist()
            })

    md_str = "\n".join(md_parts)

    # json
    json_obj = {
        "file": path.name,
        "pages": pages,
        "paragraph_count": len(paragraphs),
        "table_count": len(tables),
        "key_value_pairs": kv_pairs,            # list of {key, value}
        "paragraphs": paragraphs,               # list of strings
        "tables": table_json_list               # list of {index, rows: [[..],..]}
    }

    return text_str, md_str, json_obj

def extract_text(path: Path) -> str:
    """
    Backwards-compatible function used by the loop.
    We still return a text string so the .txt file is produced as before.
    """
    suffix = path.suffix.lower()
    if suffix in {".pdf", ".docx", ".doc"}:
        text_str, _, _ = extract_with_azure(path)
        return text_str
    elif suffix in {".txt", ".md", ".py", ".csv", ".json"}:
        return extract_text_plain(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# ---------- Main ----------
def main():
    global CLIENT

    parser = argparse.ArgumentParser(description="Ingest documents from data/raw and output clean text/markdown/json to data/text.")
    parser.add_argument("--force", action="store_true", help="Reingest all files, even if already processed.")
    args = parser.parse_args()

    # Azure client
    endpoint = os.getenv("AZURE_DOCINTEL_ENDPOINT")
    key = os.getenv("AZURE_DOCINTEL_KEY")
    if not endpoint or not key:
        print("Set AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_KEY in your environment (or .env).", file=sys.stderr)
        sys.exit(1)
    CLIENT = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    out_dir = base_dir / "data" / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists() or not raw_dir.is_dir():
        print(f"Raw directory does not exist: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    files = list(raw_dir.iterdir())
    if not files:
        print(f"No files found in {raw_dir}")
        return

    for file in files:
        if not file.is_file():
            continue

        stem = file.stem
        txt_path = out_dir / f"{stem}.txt"
        md_path  = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"

        # Skip if all outputs exist and not forcing
        if all(p.exists() for p in (txt_path, md_path, json_path)) and not args.force:
            print(f"Skipping already ingested: {file.name}")
            continue

        try:
            if file.suffix.lower() in {".pdf", ".docx", ".doc"}:
                text_str, md_str, json_obj = extract_with_azure(file)
            elif file.suffix.lower() in {".txt", ".md", ".py", ".csv", ".json"}:
                # Plain loaders: keep txt; wrap simple md/json
                text_str = extract_text_plain(file)
                md_str = f"# {file.name}\n\n## Extracted Text\n\n{text_str}"
                json_obj = {
                    "file": file.name,
                    "pages": None,
                    "paragraph_count": None,
                    "table_count": 0,
                    "key_value_pairs": [],
                    "paragraphs": text_str.splitlines(),
                    "tables": []
                }
            else:
                raise ValueError(f"Unsupported file type: {file.suffix}")

            # write all three
            txt_path.write_text(text_str, encoding="utf-8")
            md_path.write_text(md_str, encoding="utf-8")
            json_path.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"Ingested: {file.name} -> {txt_path.name}, {md_path.name}, {json_path.name}")
        except Exception as e:
            print(f"Failed to ingest {file.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
