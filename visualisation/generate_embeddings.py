#!/usr/bin/env python
"""
Generate embeddings for questions in a dataset.
"""
import argparse
import os
from pathlib import Path
from typing import Optional

from embed_utils import (
    load_config,
    load_dataset_from_source,
    generate_embeddings,
    save_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for questions in a dataset")
    parser.add_argument(
        "--dataset", 
        "-d", 
        required=True,
        help="Dataset source (Hugging Face ID or local path)"
    )
    parser.add_argument(
        "--output", 
        "-o", 
        default=None,
        help="Output directory for embeddings (default: ./embeddings/<dataset_name>)"
    )
    parser.add_argument(
        "--text-column", 
        "-t", 
        default="question",
        help="Column name containing the text to embed (default: question)"
    )
    parser.add_argument(
        "--model", 
        "-m", 
        default="nomic-ai/modernbert-embed-base",
        help="Sentence transformer model to use (default: nomic-ai/modernbert-embed-base)"
    )
    parser.add_argument(
        "--batch-size", 
        "-b", 
        type=int, 
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    parser.add_argument(
        "--config", 
        "-c", 
        default=None,
        help="Path to config file (default: ../config/config.json)"
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="search_query: ",
        help="Prefix to add to each text item (default: 'search_query: ' for queries)"
    )
    args = parser.parse_args()
    
    # Load configuration if needed
    config = load_config(args.config)
    
    # Determine output directory
    if args.output is None:
        dataset_name = args.dataset.split("/")[-1] if "/" in args.dataset else Path(args.dataset).name
        output_dir = Path("embeddings") / dataset_name
    else:
        output_dir = Path(args.output)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset_from_source(args.dataset)
    
    # Check if the text column exists
    if args.text_column not in dataset.column_names:
        available_columns = ", ".join(dataset.column_names)
        raise ValueError(
            f"Text column '{args.text_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    # Generate embeddings
    print(f"Generating embeddings using model {args.model}...")
    embeddings, model = generate_embeddings(
        dataset=dataset,
        text_column=args.text_column,
        model_name=args.model,
        batch_size=args.batch_size,
        prefix=args.prefix
    )
    
    # Save embeddings
    print(f"Saving embeddings to {output_dir}...")
    save_embeddings(
        embeddings=embeddings,
        dataset=dataset,
        output_path=str(output_dir),
        text_column=args.text_column,
        model_name=args.model,
        prefix=args.prefix
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
