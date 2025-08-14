#!/usr/bin/env python
"""
Generate tags for questions in a dataset using OpenRouter API.
"""
import argparse
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as concurrent_futures_TimeoutError
import time
import requests

# Add dotenv import
from dotenv import load_dotenv

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from embed_utils import load_config, load_dataset_from_source

# Load environment variables from .env file in parent directory
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)


def setup_model(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Set up the model configuration using the OpenRouter API key from environment.
    
    Args:
        config: Configuration dictionary with tagging settings
    
    Returns:
        Dictionary with model configuration
    """
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. "
            "Set the OPENROUTER_API_KEY environment variable."
        )
    
    if config is None or "tagging" not in config:
        # Default configuration if not provided
        return {
            "api_key": api_key,
            "model": "google/gemini-2.0-flash-001",
            "temperature": 0.2,
            "max_tokens": 150
        }
    
    # Use configuration from config.json
    tagging_config = config.get("tagging", {})
    return {
        "api_key": api_key,
        "model": tagging_config.get("model", "google/gemini-2.0-flash-001"),
        "temperature": tagging_config.get("temperature", 0.2),
        "max_tokens": tagging_config.get("max_tokens", 150)
    }


def generate_tags_for_text(
    model_config: Dict[str, Any], 
    text: str, 
    max_tags: int = 5, 
    system_prompt: str = None
) -> List[str]:
    """
    Generate tags for a given text using OpenRouter API.
    
    Args:
        model_config: Model configuration dictionary
        text: Text to generate tags for
        max_tags: Maximum number of tags to generate
        system_prompt: Optional system prompt to use
        
    Returns:
        List of tags
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant that generates concise, descriptive tags for questions. "
            "Generate exactly {max_tags} tags that capture the key topics, concepts, and skills "
            "tested in the question. Each tag should be 1-3 words, lowercase with hyphens between words."
        )
    
    system_prompt = system_prompt.format(max_tags=max_tags)
    
    prompt = f"Generate {max_tags} tags for this question: {text}\n\nTags:"
    
    try:
        headers = {
            "Authorization": f"Bearer {model_config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/TrelisResearch/ADVANCED-fine-tuning",
            "X-Title": "ADVANCED-fine-tuning Tag Generator"
        }
        
        payload = {
            "model": model_config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": model_config.get("temperature", 0.2),
            "max_tokens": model_config.get("max_tokens", 150)
        }
        
        # Add a timeout to prevent hanging
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        tags_text = response_data["choices"][0]["message"]["content"].strip()
        
        # Handle different formats that might be returned
        if "," in tags_text:
            # Comma-separated format
            tags = [tag.strip() for tag in tags_text.split(",")]
        elif "\n" in tags_text:
            # Newline-separated format
            tags = [tag.strip() for tag in tags_text.split("\n")]
        else:
            # Single tag or space-separated format
            tags = [tags_text.strip()]
        
        # Clean up tags
        tags = [tag.strip().lower() for tag in tags]
        tags = [tag.replace(" ", "-") for tag in tags]
        
        # Remove any numbering or bullets
        tags = [tag.lstrip("0123456789. -#*") for tag in tags]
        
        # Limit to max_tags
        return tags[:max_tags]
    
    except requests.exceptions.Timeout:
        print(f"Error generating tags: Request timed out after 30 seconds")
        return ["timeout-error"]
    except Exception as e:
        print(f"Error generating tags: {e}")
        return ["error-generating-tags"]


def process_batch(args):
    """Process a batch of questions to generate tags."""
    model_config, texts, max_tags, system_prompt = args
    results = []
    for text in texts:
        try:
            tags = generate_tags_for_text(model_config, text, max_tags, system_prompt)
            results.append(tags)
        except Exception as e:
            print(f"Error in process_batch: {e}")
            results.append(["batch-processing-error"])
    return results


def generate_tags_batch(
    model_config: Dict[str, Any],
    texts: List[str],
    max_tags: int = 5,
    system_prompt: str = None,
    batch_size: int = 32,
    max_workers: int = 4
) -> List[List[str]]:
    """
    Generate tags for a batch of texts using parallel processing.
    
    Args:
        model_config: Model configuration dictionary
        texts: List of texts to generate tags for
        max_tags: Maximum number of tags to generate per text
        system_prompt: Optional system prompt to use
        batch_size: Size of batches to process in parallel
        max_workers: Maximum number of worker threads
        
    Returns:
        List of tag lists, one for each input text
    """
    all_tags = []
    
    # Process in batches to avoid overwhelming the API
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
            
            futures = []
            for batch in batches:
                futures.append(executor.submit(
                    process_batch, 
                    (model_config, batch, max_tags, system_prompt)
                ))
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Generating tags", total=len(futures)):
                try:
                    batch_tags = future.result(timeout=300)  # 5-minute timeout per batch
                    all_tags.extend(batch_tags)
                except concurrent_futures_TimeoutError:
                    print("Warning: Batch processing timed out after 5 minutes")
                    # Add placeholder tags for the entire batch
                    all_tags.extend([["batch-timeout-error"]] * len(batch))
                except Exception as e:
                    print(f"Warning: Batch processing failed: {e}")
                    # Add placeholder tags for the entire batch
                    all_tags.extend([["batch-error"]] * len(batch))
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
    except Exception as e:
        print(f"Error in tag generation: {e}")
        # If something goes wrong, return placeholder tags
        return [["generation-error"]] * len(texts)
    
    # Ensure we have the right number of tag lists
    if len(all_tags) != len(texts):
        print(f"Warning: Number of tag lists ({len(all_tags)}) doesn't match number of texts ({len(texts)})")
        # Pad with placeholder tags if necessary
        if len(all_tags) < len(texts):
            all_tags.extend([["missing-tags"]] * (len(texts) - len(all_tags)))
        # Truncate if we somehow got too many
        all_tags = all_tags[:len(texts)]
    
    return all_tags


def save_tags(
    tags: List[List[str]],
    dataset: Dataset,
    output_path: str,
    text_column: str = "question",
    model_name: str = "google/gemini-2.0-flash-001",
    max_tags: int = 5
) -> None:
    """
    Save tags along with their corresponding text and metadata.
    
    Args:
        tags: List of tag lists for each text
        dataset: Dataset containing the original text and metadata
        output_path: Path to save the tags
        text_column: Column name containing the text that was tagged
        model_name: Name of the model used for tagging
        max_tags: Maximum number of tags generated per text
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a DataFrame with the original data and tags
    df = pd.DataFrame(dataset)
    
    # Add tags as a new column
    df["tags"] = [",".join(tag_list) for tag_list in tags]
    
    # Save the DataFrame
    df.to_csv(output_dir / "dataset_with_tags.csv", index=False)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "text_column": text_column,
        "max_tags": max_tags,
        "num_samples": len(tags),
        "dataset_info": dataset.info.__dict__ if hasattr(dataset, "info") else {},
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Tags saved to {output_dir}")


def load_tags(input_path: str) -> Tuple[List[List[str]], Dict[str, Any], pd.DataFrame]:
    """
    Load tags, metadata, and dataset from a saved directory.
    
    Args:
        input_path: Path to the directory containing the saved tags
        
    Returns:
        Tuple of (list of tag lists, metadata dict, dataset dataframe)
    """
    input_dir = Path(input_path)
    
    # Load dataset with tags
    df = pd.read_csv(input_dir / "dataset_with_tags.csv")
    
    # Extract tags
    tags = [tag_str.split(",") for tag_str in df["tags"]]
    
    # Load metadata
    with open(input_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return tags, metadata, df


def main():
    parser = argparse.ArgumentParser(description="Generate tags for questions in a dataset")
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
        help="Output directory for tags (default: ./tags/<dataset_name>)"
    )
    parser.add_argument(
        "--text-column", 
        "-t", 
        default="question",
        help="Column name containing the text to tag (default: question)"
    )
    parser.add_argument(
        "--config", 
        "-c", 
        default=None,
        help="Path to config file (default: ../config/config.json)"
    )
    parser.add_argument(
        "--system-prompt",
        "-p",
        default=None,
        help="Custom system prompt for tag generation"
    )
    parser.add_argument(
        "--max-tags", 
        type=int, 
        default=5,
        help="Maximum number of tags to generate per text (default: 5)"
    )
    parser.add_argument(
        "--batch-size", 
        "-b", 
        type=int, 
        default=32,
        help="Batch size for tag generation (default: 32)"
    )
    parser.add_argument(
        "--max-workers", 
        "-w", 
        type=int, 
        default=4,
        help="Maximum number of worker threads (default: 4)"
    )
    args = parser.parse_args()
    
    # Load configuration if needed
    config = load_config(args.config)
    
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. "
            "Set the OPENROUTER_API_KEY environment variable."
        )
    
    # Determine output directory
    if args.output is None:
        dataset_name = args.dataset.split("/")[-1] if "/" in args.dataset else Path(args.dataset).name
        output_dir = Path("tags") / dataset_name
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
    
    # Setup model
    print(f"Setting up model...")
    model_config = setup_model(config)
    
    # Generate tags
    print(f"Generating tags for {len(dataset)} texts...")
    tags = generate_tags_batch(
        model_config=model_config,
        texts=dataset[args.text_column],
        max_tags=args.max_tags,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    # Save tags
    print(f"Saving tags to {output_dir}...")
    save_tags(
        tags=tags,
        dataset=dataset,
        output_path=str(output_dir),
        text_column=args.text_column,
        model_name=model_config["model"],
        max_tags=args.max_tags
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
