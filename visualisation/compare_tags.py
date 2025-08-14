#!/usr/bin/env python
"""
Compare tags between datasets and visualize their distribution.
"""
import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter

# Add dotenv import
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

from embed_utils import load_config, load_dataset_from_source
from generate_tags import setup_model, generate_tags_batch, save_tags, load_tags

# Load environment variables from .env file in parent directory
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)


def get_or_generate_tags(
    dataset_path: str,
    model_name: str = "google/gemini-2.0-flash-001",
    text_column: str = "question",
    max_tags: int = 5,
    batch_size: int = 32,
    max_workers: int = 4,
    system_prompt: str = None
) -> Tuple[List[List[str]], Dict[str, Any], pd.DataFrame]:
    """
    Get tags for a dataset, generating them if they don't exist.
    
    Args:
        dataset_path: Path to the dataset
        model_name: Model to use for tag generation
        text_column: Column containing the text to tag
        max_tags: Maximum number of tags to generate per text
        batch_size: Batch size for tag generation
        max_workers: Maximum number of worker threads
        system_prompt: Optional system prompt for tag generation
        
    Returns:
        Tuple of (tags, metadata, dataframe)
    """
    # Determine the expected tags directory
    dataset_name = dataset_path.split("/")[-1] if "/" in dataset_path else Path(dataset_path).name
    tags_dir = Path("tags") / dataset_name
    
    # Check if tags already exist
    if tags_dir.exists() and (tags_dir / "dataset_with_tags.csv").exists():
        print(f"Loading existing tags from {tags_dir}...")
        return load_tags(str(tags_dir))
    
    # Tags don't exist, need to generate them
    print(f"Tags not found for {dataset_path}, generating...")
    
    # Option 1: Use the generate_tags.py script directly
    try:
        print(f"Running generate_tags.py for {dataset_path}...")
        cmd = [
            "uv", "run", 
            str(Path(__file__).parent / "generate_tags.py"),
            "--dataset", dataset_path,
            "--text-column", text_column,
            "--max-tags", str(max_tags),
            "--batch-size", str(batch_size),
            "--max-workers", str(max_workers)
        ]
        
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
            
        # Run the generate_tags.py script
        subprocess.run(cmd, check=True)
        
        # Now the tags should exist, so load them
        if tags_dir.exists() and (tags_dir / "dataset_with_tags.csv").exists():
            print(f"Tags generated successfully, loading from {tags_dir}...")
            return load_tags(str(tags_dir))
    except Exception as e:
        print(f"Warning: Failed to run generate_tags.py: {e}")
        print("Falling back to inline tag generation...")
    
    # Option 2: Generate tags inline (fallback)
    # Load dataset
    dataset = load_dataset_from_source(dataset_path)
    
    # Load configuration
    config = load_config()
    
    # Setup model
    model_config = setup_model(config)
    
    # Generate tags
    tags = generate_tags_batch(
        model_config=model_config,
        texts=dataset[text_column],
        max_tags=max_tags,
        system_prompt=system_prompt,
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    # Save tags
    save_tags(
        tags=tags,
        dataset=dataset,
        output_path=str(tags_dir),
        text_column=text_column,
        model_name=model_config["model"],
        max_tags=max_tags
    )
    
    # Load the saved tags to ensure consistent format
    return load_tags(str(tags_dir))


def create_tag_vectors(
    tags_list: List[List[str]],
    min_tag_freq: int = 2
) -> Tuple[np.ndarray, List[str]]:
    """
    Create binary vectors for tags using CountVectorizer.
    
    Args:
        tags_list: List of tag lists for each text
        min_tag_freq: Minimum frequency for a tag to be included
        
    Returns:
        Tuple of (tag vectors array, list of tag names)
        
    Raises:
        ValueError: If no valid tags are found or vectorization fails
    """
    # Check if we have any tags
    if not tags_list or all(not tags for tags in tags_list):
        raise ValueError(
            "No tags found in the dataset. Cannot create meaningful tag vectors. "
            "Check that tag generation is working correctly."
        )
    
    # Flatten all tags to count frequencies
    all_tags = [tag for sublist in tags_list for tag in sublist]
    
    # Check if we have any tags after flattening
    if not all_tags:
        raise ValueError(
            "No tags found after processing. Cannot create meaningful tag vectors. "
            "Check that tag generation is working correctly."
        )
    
    tag_counts = Counter(all_tags)
    
    # Filter tags by frequency
    frequent_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_freq}
    
    # If no tags meet the frequency threshold, lower the threshold
    if not frequent_tags and min_tag_freq > 1:
        print(f"Warning: No tags meet the frequency threshold of {min_tag_freq}. Lowering threshold to 1.")
        frequent_tags = {tag for tag, count in tag_counts.items() if count >= 1}
    
    # If still no tags, use all tags regardless of frequency
    if not frequent_tags:
        print("Warning: Using all tags regardless of frequency.")
        frequent_tags = set(all_tags)
    
    # Convert tag lists to strings for CountVectorizer
    tag_strings = [" ".join(filter(lambda x: x in frequent_tags, tags)) for tags in tags_list]
    
    # Check if any tag strings are empty after filtering
    if all(not tag_str for tag_str in tag_strings):
        raise ValueError(
            "All tag strings are empty after filtering. Cannot create meaningful tag vectors. "
            "Try lowering the minimum tag frequency or generating more diverse tags."
        )
    
    # Create vectors
    try:
        vectorizer = CountVectorizer(binary=True)
        tag_vectors = vectorizer.fit_transform(tag_strings).toarray()
        
        # Get feature names
        tag_names = vectorizer.get_feature_names_out()
        
        # Check if we have any features
        if tag_vectors.shape[1] == 0:
            raise ValueError(
                "No features extracted by CountVectorizer. Cannot create meaningful tag vectors. "
                "Check that tags are properly formatted and not being filtered out."
            )
        
        return tag_vectors, tag_names
    except Exception as e:
        raise ValueError(f"Error in creating tag vectors: {e}")


def reduce_dimensions(
    vectors: np.ndarray,
    method: str = "tsne",
    n_components: int = 2
) -> np.ndarray:
    """
    Reduce dimensions of tag vectors for visualization.
    
    Args:
        vectors: Tag vectors array
        method: Dimension reduction method ('tsne', 'umap', or 'pca')
        n_components: Number of components to reduce to
        
    Returns:
        Reduced vectors array
    
    Raises:
        ValueError: If dimension reduction fails or input data is invalid
    """
    # Preprocessing to avoid numerical issues
    # Replace NaN, inf values with zeros
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if we have enough non-zero vectors for dimension reduction
    non_zero_rows = np.any(vectors != 0, axis=1)
    if np.sum(non_zero_rows) < 2:
        raise ValueError(
            "Not enough non-zero vectors for dimension reduction. "
            "This suggests the tag data is too sparse or uniform. "
            "Try increasing the number of tags per item or decreasing the minimum tag frequency."
        )
    
    # If we have zero vectors, add a tiny amount of noise to avoid numerical issues
    if not np.all(non_zero_rows):
        print("Warning: Zero vectors detected. Adding small noise to avoid numerical issues.")
        # Add a tiny amount of noise to all vectors
        noise = np.random.normal(0, 1e-6, vectors.shape)
        vectors = vectors + noise
    
    # Apply scaling to improve numerical stability
    # Normalize vectors to unit length to avoid extreme values
    row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    row_norms[row_norms == 0] = 1.0
    vectors_normalized = vectors / row_norms
    
    # Set up the reducer based on the selected method
    try:
        if method.lower() == "tsne":
            # Use more stable parameters for TSNE
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, max(5, vectors.shape[0] // 10)),
                max_iter=1000,
                random_state=42,
                init='pca',  # Use PCA initialization for better stability
                learning_rate='auto',  # Auto learning rate for better convergence
                metric='cosine'  # Use cosine distance which works better for sparse data
            )
        elif method.lower() == "umap":
            # UMAP with more stable parameters
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, max(2, vectors.shape[0] // 10)),
                min_dist=0.1,
                random_state=42,
                metric='cosine'  # Use cosine distance which works better for sparse data
            )
        else:  # pca - most stable option
            reducer = PCA(n_components=min(n_components, vectors.shape[0], vectors.shape[1]))
        
        # Try the selected method first
        try:
            # Temporarily suppress specific warnings during dimension reduction
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                reduced_vectors = reducer.fit_transform(vectors_normalized)
            return reduced_vectors
        except Exception as e:
            # If the selected method fails, fall back to PCA which is more stable
            if method.lower() != "pca":
                print(f"Warning: {method} failed with error: {e}. Falling back to PCA.")
                reducer = PCA(n_components=min(n_components, vectors.shape[0], vectors.shape[1]))
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        reduced_vectors = reducer.fit_transform(vectors_normalized)
                    return reduced_vectors
                except Exception as pca_error:
                    raise ValueError(f"Both {method} and PCA failed. Original error: {e}, PCA error: {pca_error}")
            else:
                # If PCA was the original method and it failed
                raise ValueError(f"PCA dimension reduction failed: {e}")
    except Exception as e:
        raise ValueError(f"Error in dimension reduction: {e}")


def plot_tag_distribution(
    df_list: List[pd.DataFrame],
    dataset_labels: List[str],
    output_file: Optional[str] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the distribution of top tags across datasets.
    
    Args:
        df_list: List of DataFrames with tags
        dataset_labels: List of labels for each dataset
        output_file: Output file path (if None, display the plot)
        top_n: Number of top tags to display
        figsize: Figure size
    """
    # Count tags in each dataset
    tag_counts = []
    for df in df_list:
        # Split the tags and count them
        all_tags = []
        for tag_str in df["tags"]:
            # Filter out empty tags and strip whitespace
            all_tags.extend([tag.strip() for tag in tag_str.split(",") if tag.strip()])
        tag_counts.append(Counter(all_tags))
    
    # Get the top N tags across all datasets
    combined_counts = Counter()
    for counts in tag_counts:
        combined_counts.update(counts)
    top_tags = [tag for tag, _ in combined_counts.most_common(top_n)]
    
    # Create data for plotting
    plot_data = []
    for i, (counts, label) in enumerate(zip(tag_counts, dataset_labels)):
        for tag in top_tags:
            plot_data.append({
                "Tag": tag,
                "Count": counts.get(tag, 0),
                "Dataset": label,
                "Percentage": counts.get(tag, 0) / len(df_list[i]) * 100
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig = px.bar(
        plot_df,
        x="Tag",
        y="Percentage",
        color="Dataset",
        barmode="group",
        title=f"Top {top_n} Tags Distribution Across Datasets",
        labels={"Percentage": "Percentage of Questions (%)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        xaxis_title="Tag",
        yaxis_title="Percentage of Questions (%)",
        xaxis_tickangle=-45,
        legend_title="Dataset",
    )
    
    # Save or display
    if output_file:
        fig.write_html(output_file)
        print(f"Tag distribution plot saved to {output_file}")
    else:
        fig.show()


def plot_combined_tag_vectors(
    vectors_list: List[np.ndarray],
    df_list: List[pd.DataFrame],
    dataset_labels: List[str],
    method: str = "tsne",
    n_components: int = 2,
    title: str = "Tag-Based Dataset Comparison",
    output_file: Optional[str] = None,
) -> None:
    """
    Plot combined tag vectors from multiple datasets.
    
    Args:
        vectors_list: List of tag vector arrays from each dataset
        df_list: List of DataFrames containing data from each dataset
        dataset_labels: List of labels for each dataset
        method: Dimension reduction method ('tsne', 'umap', or 'pca')
        n_components: Number of components to reduce to
        title: Plot title
        output_file: Output file path (if None, display the plot)
    """
    # Combine vectors
    combined_vectors = np.vstack(vectors_list)
    
    # Handle potential NaN or infinite values in vectors
    combined_vectors = np.nan_to_num(combined_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        # Reduce dimensions
        reduced_vectors = reduce_dimensions(combined_vectors, method, n_components)
        
        # Split back into original datasets
        start_idx = 0
        reduced_list = []
        for vectors in vectors_list:
            end_idx = start_idx + len(vectors)
            reduced_list.append(reduced_vectors[start_idx:end_idx])
            start_idx = end_idx
        
        # Create DataFrames for plotting
        plot_df_list = []
        
        for i, (reduced, df, label) in enumerate(zip(reduced_list, df_list, dataset_labels)):
            plot_df = pd.DataFrame(
                reduced, 
                columns=[f"Dimension {i+1}" for i in range(n_components)]
            )
            # Add text and tags for hover data
            plot_df["question"] = df["question"].values
            plot_df["tags"] = df["tags"].values
            plot_df["dataset"] = label
            plot_df_list.append(plot_df)
        
        # Combine for plotting
        plot_df = pd.concat(plot_df_list, ignore_index=True)
        
        # Create the plot
        if n_components == 3:
            fig = px.scatter_3d(
                plot_df,
                x="Dimension 1",
                y="Dimension 2",
                z="Dimension 3",
                color="dataset",
                hover_data=["question", "tags", "dataset"],
                title=title,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
        else:
            fig = px.scatter(
                plot_df,
                x="Dimension 1",
                y="Dimension 2",
                color="dataset",
                hover_data=["question", "tags", "dataset"],
                title=title,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title=title,
        )
        
        # Update hover template to make it more readable and fix the customdata issue
        fig.update_traces(
            hovertemplate=(
                "<b>Dataset:</b> %{customdata[2]}<br>"
                "<b>Question:</b> %{customdata[0]}<br>"
                "<b>Tags:</b> %{customdata[1]}<br>"
                "<extra></extra>"
            )
        )
        
        # Save or display
        if output_file:
            fig.write_html(output_file)
            print(f"Interactive plot saved to {output_file}")
        else:
            fig.show()
            
    except ValueError as e:
        print(f"Error creating visualization: {e}")
        print("Skipping combined tag vectors visualization.")


def compare_dataset_pair(
    dataset_a: Tuple[List[List[str]], Dict[str, Any], pd.DataFrame],
    dataset_b: Tuple[List[List[str]], Dict[str, Any], pd.DataFrame],
    output_dir: str,
    min_tag_freq: int = 2,
    method: str = "tsne",
    model_name: str = None,
) -> None:
    """
    Compare a pair of datasets based on their tags.
    
    Args:
        dataset_a: Tuple of (tags, metadata, df) for first dataset
        dataset_b: Tuple of (tags, metadata, df) for second dataset
        output_dir: Directory to save outputs
        min_tag_freq: Minimum frequency for a tag to be included
        method: Dimension reduction method
        model_name: Model name used for tagging (for display)
    """
    tags_a, meta_a, df_a = dataset_a
    tags_b, meta_b, df_b = dataset_b
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get dataset labels
    label_a = meta_a.get("dataset_info", {}).get("dataset_name", "Dataset A")
    label_b = meta_b.get("dataset_info", {}).get("dataset_name", "Dataset B")
    
    # Combine tags from both datasets
    all_tags = tags_a + tags_b
    
    # Create tag vectors
    vectors, tag_names = create_tag_vectors(all_tags, min_tag_freq)
    
    # Split vectors back to original datasets
    vectors_a = vectors[:len(tags_a)]
    vectors_b = vectors[len(tags_a):]
    
    # Plot tag distribution
    plot_tag_distribution(
        df_list=[df_a, df_b],
        dataset_labels=[label_a, label_b],
        output_file=str(output_path / f"{label_a}_vs_{label_b}_tag_distribution.html"),
    )
    
    # Plot combined tag vectors
    plot_combined_tag_vectors(
        vectors_list=[vectors_a, vectors_b],
        df_list=[df_a, df_b],
        dataset_labels=[label_a, label_b],
        method=method,
        title=f"Tag-Based Comparison: {label_a} vs {label_b}",
        output_file=str(output_path / f"{label_a}_vs_{label_b}_tag_vectors.html"),
    )
    
    print(f"Comparison between {label_a} and {label_b} completed.")


def main():
    parser = argparse.ArgumentParser(description="Compare tags between datasets")
    parser.add_argument(
        "--datasets", 
        "-d", 
        nargs="+", 
        required=True,
        help="Paths to the datasets to compare (1-5 datasets)"
    )
    parser.add_argument(
        "--labels", 
        "-l", 
        nargs="+", 
        default=None,
        help="Optional custom labels for the datasets"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        default="tag_comparison_results",
        help="Output directory for results (default: tag_comparison_results)"
    )
    parser.add_argument(
        "--text-column", 
        "-t", 
        default="question",
        help="Column name containing the text to compare (default: question)"
    )
    parser.add_argument(
        "--model", 
        "-m", 
        default="google/gemini-2.0-flash-001",
        help="Model to use for tag generation (default: google/gemini-2.0-flash-001)"
    )
    parser.add_argument(
        "--max-tags", 
        type=int, 
        default=5,
        help="Maximum number of tags to generate per text (default: 5)"
    )
    parser.add_argument(
        "--min-tag-freq", 
        type=int, 
        default=2,
        help="Minimum frequency for a tag to be included (default: 2)"
    )
    parser.add_argument(
        "--method", 
        choices=["tsne", "umap", "pca"], 
        default="tsne",
        help="Dimension reduction method (default: tsne)"
    )
    parser.add_argument(
        "--system-prompt",
        "-p",
        default=None,
        help="Custom system prompt for tag generation"
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Enable pairwise comparisons between datasets (default: disabled)"
    )
    parser.add_argument(
        "--fallback-to-pca",
        action="store_true",
        help="Fallback to PCA if the selected method fails (default: disabled)"
    )
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.datasets) > 5:
        print("Warning: More than 5 datasets provided. Only the first 5 will be used.")
        args.datasets = args.datasets[:5]
    
    if args.labels and len(args.labels) != len(args.datasets):
        print(f"Error: Number of labels ({len(args.labels)}) doesn't match number of datasets ({len(args.datasets)})")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use last component of dataset paths as labels if not provided
    if not args.labels:
        args.labels = [path.split("/")[-1] if "/" in path else Path(path).name for path in args.datasets]
    
    # Get or generate tags for each dataset
    print("Processing datasets...")
    dataset_data = []
    
    for i, dataset_path in enumerate(args.datasets):
        print(f"Processing dataset {i+1}/{len(args.datasets)}: {dataset_path}")
        tags, metadata, df = get_or_generate_tags(
            dataset_path=dataset_path,
            model_name=args.model,
            text_column=args.text_column,
            max_tags=args.max_tags,
            system_prompt=args.system_prompt
        )
        
        # Update metadata with custom label
        if "dataset_info" not in metadata:
            metadata["dataset_info"] = {}
        metadata["dataset_info"]["dataset_name"] = args.labels[i]
        
        dataset_data.append((tags, metadata, df))
    
    # Compare all pairs of datasets
    if args.pairwise and len(dataset_data) > 1:
        print("Comparing dataset pairs...")
        for i in range(len(dataset_data)):
            for j in range(i+1, len(dataset_data)):
                print(f"Comparing {args.labels[i]} vs {args.labels[j]}...")
                try:
                    compare_dataset_pair(
                        dataset_a=dataset_data[i],
                        dataset_b=dataset_data[j],
                        output_dir=str(output_dir),
                        min_tag_freq=args.min_tag_freq,
                        method=args.method,
                        model_name=args.model
                    )
                except ValueError as e:
                    print(f"Error comparing {args.labels[i]} vs {args.labels[j]}: {e}")
                    print(f"Skipping comparison between {args.labels[i]} and {args.labels[j]}.")
    
    # If there are more than 1 dataset, create a combined visualization
    if len(dataset_data) > 1:
        print("Creating combined visualization...")
        
        try:
            # Extract data
            all_tags = []
            all_dfs = []
            for tags, _, df in dataset_data:
                all_tags.extend(tags)
                all_dfs.append(df)
            
            # Create tag vectors
            vectors, tag_names = create_tag_vectors(all_tags, args.min_tag_freq)
            
            # Split vectors back to original datasets
            start_idx = 0
            vectors_list = []
            for tags, _, _ in dataset_data:
                end_idx = start_idx + len(tags)
                vectors_list.append(vectors[start_idx:end_idx])
                start_idx = end_idx
            
            # Plot combined tag vectors
            plot_combined_tag_vectors(
                vectors_list=vectors_list,
                df_list=all_dfs,
                dataset_labels=args.labels,
                method=args.method,
                title=f"Tag-Based Comparison of {len(args.datasets)} Datasets",
                output_file=str(output_dir / "combined_tag_comparison.html"),
            )
            
            # Plot tag distribution
            plot_tag_distribution(
                df_list=all_dfs,
                dataset_labels=args.labels,
                output_file=str(output_dir / "combined_tag_distribution.html"),
            )
        except ValueError as e:
            print(f"Error creating combined visualization: {e}")
            print("Skipping combined visualization.")
    
    print(f"All comparisons completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
