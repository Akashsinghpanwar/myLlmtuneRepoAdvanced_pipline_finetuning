"""
Utility functions for generating and working with embeddings.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from specified config file or default config/config.json."""
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent.parent / "config" / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_from_source(dataset_source: str) -> Dataset:
    """
    Load a dataset from a Hugging Face dataset ID or local path.
    
    Args:
        dataset_source: Either a Hugging Face dataset ID (e.g., 'Trelis/dataset-name') 
                        or a local path to a saved dataset
    
    Returns:
        Dataset object with the loaded data
    """
    try:
        if os.path.exists(dataset_source):
            # Load from local path
            dataset = load_from_disk(dataset_source)
        else:
            # Load from Hugging Face Hub
            dataset = load_dataset(dataset_source)
        
        # If it's a DatasetDict, get the 'train' split by default
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            return dataset["train"]
        
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {dataset_source}: {e}")


def generate_embeddings(
    dataset: Dataset, 
    text_column: str = "question", 
    model_name: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 32,
    cache_dir: Optional[str] = None,
    prefix: str = "search_query: "
) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Generate embeddings for text in a dataset using a sentence transformer model.
    
    Args:
        dataset: Dataset containing the text to embed
        text_column: Column name containing the text to embed
        model_name: Name of the sentence transformer model to use
        batch_size: Batch size for embedding generation
        cache_dir: Directory to cache the model
        prefix: Prefix to add to each text item (e.g., "search_query: " for queries)
        
    Returns:
        Tuple of (embeddings array, model)
    """
    # Load the model
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    
    # Get the text to embed
    texts = dataset[text_column]
    
    # Add prefix to each text item if specified
    if prefix:
        texts = [f"{prefix}{text}" for text in texts]
    
    # Generate embeddings
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    return embeddings, model


def save_embeddings(
    embeddings: np.ndarray, 
    dataset: Dataset, 
    output_path: str,
    text_column: str = "question",
    model_name: str = "nomic-ai/modernbert-embed-base",
    prefix: str = "search_query: "
) -> None:
    """
    Save embeddings along with their corresponding text and metadata.
    
    Args:
        embeddings: Array of embeddings
        dataset: Dataset containing the original text and metadata
        output_path: Path to save the embeddings
        text_column: Column name containing the text that was embedded
        model_name: Name of the model used for embedding
        prefix: Prefix that was added to the text before embedding
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "text_column": text_column,
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        "prefix": prefix,
        "dataset_info": dataset.info.__dict__ if hasattr(dataset, "info") else {},
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save the original text and any other columns as a CSV
    df = pd.DataFrame(dataset)
    df.to_csv(output_dir / "dataset.csv", index=False)
    
    print(f"Embeddings saved to {output_dir}")


def load_embeddings(input_path: str) -> Tuple[np.ndarray, Dict[str, Any], pd.DataFrame]:
    """
    Load embeddings, metadata, and dataset from a saved directory.
    
    Args:
        input_path: Path to the directory containing the saved embeddings
        
    Returns:
        Tuple of (embeddings array, metadata dict, dataset dataframe)
    """
    input_dir = Path(input_path)
    
    # Load embeddings
    embeddings = np.load(input_dir / "embeddings.npy")
    
    # Load metadata
    with open(input_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Load dataset
    df = pd.read_csv(input_dir / "dataset.csv")
    
    return embeddings, metadata, df


def compute_pairwise_similarities(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between two sets of embeddings.
    
    Args:
        embeddings_a: First set of embeddings
        embeddings_b: Second set of embeddings
        
    Returns:
        Matrix of pairwise cosine similarities
    """
    return cosine_similarity(embeddings_a, embeddings_b)


def find_closest_embeddings(
    query_embeddings: np.ndarray, 
    corpus_embeddings: np.ndarray,
    top_k: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the closest embeddings in a corpus for each query embedding.
    
    Args:
        query_embeddings: Query embeddings
        corpus_embeddings: Corpus embeddings to search in
        top_k: Number of closest embeddings to return
        
    Returns:
        Tuple of (indices of closest embeddings, similarity scores)
    """
    # Compute similarities
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    
    # Get top-k indices and scores
    top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(similarities, top_indices, axis=1)
    
    return top_indices, top_scores
