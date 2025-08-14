0) Inputs & config
Inputs: PDFs, Word, text, Markdown, code, CSV/JSON, and scanned/images with tables.

Secrets: .env (Azure DI keys, OpenRouter key).

Tuning knobs: config/config.json (summary/chunk/QA models, lengths, iteration, batching).

1) Ingest & OCR (layout-aware)
Tooling: Azure Document Intelligence (OCR + layout + tables).

Script: utils/batch_extract_plus_llm.py (or your Azure-only ingest).

What happens:

Extracts paragraphs + tables (structured).

Saves:

data/out/<doc>/text.txt (full text)

data/out/<doc>/text.raw.md (raw MD with tables)

data/out/<doc>/table_#.csv (each table)

data/out/<doc>/text.md (optionally LLM-beautified Markdown; no hallucinations, numbers/units preserved)

data/out/<doc>/raw.json (extraction stats)

Why images/tables work: Azure DI reads scanned pages and detects tables → exported to CSV + pretty Markdown.

2) Normalize & chunk (high-quality splits)
Script: utils/chunk.py (your improved version).

What happens:

Markdown-aware parsing (headings, lists, code, tables, fenced ```csv).

Keeps tables atomic (never split), sentence-aware for prose.

Length control by tokens or chars, soft min/max.

Outputs:

data/chunks/<doc>/chunk_001.txt, …

data/chunks/<doc>/metadata.json (lengths, tableish flags)

data/chunks/<doc>/chunks.jsonl (index of chunks)

3) Summaries (strict JSON, extractive)
Script: utils/summarize.py.

Two levels: --level doc and --level chunk.

Models: via OpenRouter (configurable; you set openai/gpt-5 in config).

Guarantees: STRICT JSON (schemas), no hallucination, numbers/units preserved.

Outputs:

data/summaries/<doc>_summary.json

data/summaries/chunks/<doc>/chunk_###.json

4) Q&A generation (coverage-driven, iterative)
Script: utils/generate_qa.py (your fixed version).

Uses: doc summary + each chunk text.

Controls: difficulty mix, factual vs reasoning ratio, iteration with no duplicates, optional table-focused questions.

Outputs per doc: data/qa/<doc>/chunk_###_qa.json (+ optional ..._iter_k.json and ..._qa_combined.json)

Metadata: data/qa/<doc>/qa_metadata.json

5) Dataset assembly (trainable JSONL + HF dataset)
Scripts: utils/create_dataset.py (or your newer builder).

Bundles three views:

Axess-RAG: data/dataset/axess_rag.jsonl (for indexing/retrieval)

OpenAI-Instruct: data/dataset/openai_instruct.jsonl (messages → assistant; ideal for SFT)

OpenAI-Chat: data/dataset/openai_chat.jsonl (multi-turn chat)

Validation: you already verified all lines parse as JSON.

(Optional) creates HF “saved to disk” dataset for Transformers training.

6) Finetuning (Open-source, RunPod)
Script: train_sft.py (QLoRA with TRL).

Base: openai-oss (or Qwen/Llama), using chat template.

Train: on openai_instruct.jsonl, eval on openai_chat.jsonl.

Serve: with vLLM + LoRA or merged weights; push adapter/model to HF.

Why this produces high-quality finetune data
OCR robustness: Azure DI handles scans + images + tables reliably.

Structure-aware chunking: tables preserved intact; sentences respected.

Strict JSON summaries & QA: schemas, no Markdown, no guesswork; numbers/units/part codes exact.

Coverage + iteration: Q&A spans all facts + reasoning, avoids duplicates, includes table lookups.

Clean training formats: ready messages JSONL for SFT, plus RAG view for retrieval.

Reproducible: single config, CLI steps, rich metadata at each stage.
