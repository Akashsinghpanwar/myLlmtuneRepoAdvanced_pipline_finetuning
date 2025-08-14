#!/usr/bin/env python
"""
Visualize train vs. eval split embeddings from a dataset.
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from datasets import load_from_disk

from embed_utils import load_embeddings


def load_dataset_split(dataset_path):
    """
    Load train and eval splits from a dataset.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Tuple of (train_df, eval_df)
    """
    try:
        # Load dataset
        dataset = load_from_disk(dataset_path)
        
        # Check if dataset has train and eval splits
        if 'train' in dataset and 'eval' in dataset:
            train_df = pd.DataFrame(dataset['train'])
            eval_df = pd.DataFrame(dataset['eval'])
            return train_df, eval_df
        else:
            # If no explicit splits, assume it's all one dataset
            df = pd.DataFrame(dataset)
            # Check if there's a 'split' column
            if 'split' in df.columns:
                train_df = df[df['split'] == 'train']
                eval_df = df[df['split'] == 'eval']
                return train_df, eval_df
            else:
                print("No train/eval splits found in dataset. Using entire dataset as train.")
                return df, pd.DataFrame()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def generate_embeddings(texts, model_name="nomic-ai/modernbert-embed-base"):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the embedding model
        
    Returns:
        Array of embeddings
    """
    from sentence_transformers import SentenceTransformer
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Add prefix for modernbert
    if "modernbert" in model_name.lower():
        prefixed_texts = [f"search_query: {text}" for text in texts]
    else:
        prefixed_texts = texts
    
    # Generate embeddings
    embeddings = model.encode(
        prefixed_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def reduce_dimensions(embeddings, method="tsne", n_components=2, **kwargs):
    """
    Reduce dimensions of embeddings using the specified method.
    
    Args:
        embeddings: Embeddings array
        method: Dimension reduction method ('pca', 'tsne', or 'umap')
        n_components: Number of components to reduce to
        **kwargs: Additional arguments for the dimension reduction method
        
    Returns:
        Reduced embeddings
    """
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method.lower() == "tsne":
        default_kwargs = {"perplexity": 30, "n_iter": 1000, "random_state": 42}
        default_kwargs.update(kwargs)
        reducer = TSNE(n_components=n_components, **default_kwargs)
    elif method.lower() == "umap":
        default_kwargs = {
            "n_neighbors": 15, 
            "min_dist": 0.1, 
            "random_state": 42,
            "n_jobs": 1
        }
        default_kwargs.update(kwargs)
        reducer = umap.UMAP(n_components=n_components, **default_kwargs)
    else:
        raise ValueError(f"Unknown dimension reduction method: {method}")
    
    print(f"Reducing dimensions using {method}...")
    return reducer.fit_transform(embeddings)


def plot_split_embeddings(
    reduced_embeddings,
    is_train,
    df,
    text_column="question",
    color_column=None,
    title="Train vs. Eval Split Visualization",
    output_file=None,
    interactive=True,
):
    """
    Plot train vs. eval split embeddings.
    
    Args:
        reduced_embeddings: Reduced embeddings array
        is_train: Boolean array indicating which points are from train split
        df: Combined DataFrame containing the original data
        text_column: Column name containing the text that was embedded
        color_column: Column name to use for additional coloring (e.g., cluster)
        title: Plot title
        output_file: Output file path (if None, display the plot)
        interactive: Whether to use interactive Plotly visualization
    """
    # Create a DataFrame with the reduced embeddings
    viz_df = pd.DataFrame(
        reduced_embeddings, 
        columns=[f"Dimension {i+1}" for i in range(reduced_embeddings.shape[1])]
    )
    
    # Add split information
    viz_df["Split"] = ["Train" if t else "Eval" for t in is_train]
    
    # Add text column
    if text_column in df.columns:
        viz_df["Text"] = df[text_column].values
    
    # Add color column if specified
    if color_column and color_column in df.columns:
        viz_df["Color"] = df[color_column].values
    
    if interactive:
        # Create interactive Plotly visualization
        if color_column and color_column in df.columns:
            # Use both split and color column
            fig = px.scatter(
                viz_df,
                x="Dimension 1",
                y="Dimension 2",
                color="Split",
                symbol="Color" if "Color" in viz_df.columns else None,
                hover_data=["Text"] if "Text" in viz_df.columns else None,
                title=title,
                color_discrete_map={"Train": "blue", "Eval": "red"},
            )
        else:
            # Use only split
            fig = px.scatter(
                viz_df,
                x="Dimension 1",
                y="Dimension 2",
                color="Split",
                hover_data=["Text"] if "Text" in viz_df.columns else None,
                title=title,
                color_discrete_map={"Train": "blue", "Eval": "red"},
            )
        
        # Update layout
        fig.update_layout(
            legend_title_text="Dataset Split",
            width=1000,
            height=800,
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Interactive plot saved to {output_file}")
        else:
            fig.show()
    else:
        # Create static matplotlib visualization
        plt.figure(figsize=(12, 10))
        
        # Plot train points
        train_mask = is_train
        plt.scatter(
            reduced_embeddings[train_mask, 0],
            reduced_embeddings[train_mask, 1],
            alpha=0.7,
            label="Train",
            color="blue",
        )
        
        # Plot eval points
        eval_mask = ~is_train
        plt.scatter(
            reduced_embeddings[eval_mask, 0],
            reduced_embeddings[eval_mask, 1],
            alpha=0.7,
            label="Eval",
            color="red",
        )
        
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize train vs. eval split embeddings.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset with train/eval splits")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap", "pca"], help="Dimension reduction method")
    parser.add_argument("--text-column", type=str, default="question", help="Column name for the text")
    parser.add_argument("--color-column", type=str, help="Optional column name for additional coloring")
    parser.add_argument("--output", type=str, help="Output file path (HTML for interactive, PNG for static)")
    parser.add_argument("--static", action="store_true", help="Use static matplotlib visualization instead of interactive Plotly")
    parser.add_argument("--model", type=str, default="nomic-ai/modernbert-embed-base", help="Embedding model name")
    parser.add_argument("--save-embeddings", action="store_true", help="Save generated embeddings")
    parser.add_argument("--embeddings-dir", type=str, default="embeddings", help="Directory to save embeddings")
    parser.add_argument("--show-clusters", action="store_true", help="Perform clustering and show cluster assignments")
    parser.add_argument("--num-clusters", type=int, default=0, help="Number of clusters (0 for auto-detection)")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    train_df, eval_df = load_dataset_split(args.dataset)
    
    if train_df is None or (len(eval_df) == 0 and train_df.empty):
        print("Failed to load dataset or no data found.")
        return
    
    # Combine train and eval data
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)
    is_train = np.array([True] * len(train_df) + [False] * len(eval_df))
    
    # Check if text column exists
    if args.text_column not in combined_df.columns:
        print(f"Text column '{args.text_column}' not found in dataset. Available columns: {combined_df.columns.tolist()}")
        return
    
    # Generate embeddings
    print(f"Generating embeddings for {len(combined_df)} examples...")
    texts = combined_df[args.text_column].tolist()
    embeddings = generate_embeddings(texts, args.model)
    
    # Perform clustering if requested
    color_column = args.color_column
    if args.show_clusters:
        from sklearn.cluster import KMeans
        from kneed import KneeLocator
        
        # Determine number of clusters
        num_clusters = args.num_clusters
        if num_clusters <= 0:
            # Auto-detect using elbow method
            max_k = min(20, len(embeddings) // 5)
            inertias = []
            k_values = range(2, max_k + 1)
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            
            # Find the elbow point
            try:
                kneedle = KneeLocator(
                    list(k_values), 
                    inertias, 
                    curve="convex", 
                    direction="decreasing"
                )
                
                if kneedle.elbow is not None:
                    num_clusters = kneedle.elbow
                    print(f"Auto-detected {num_clusters} clusters using elbow method")
                else:
                    num_clusters = 3
                    print("Elbow detection failed, defaulting to 3 clusters")
            except Exception as e:
                num_clusters = 3
                print(f"Error in elbow detection: {e}, defaulting to 3 clusters")
        
        # Perform clustering
        print(f"Clustering data into {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Add cluster column to DataFrame
        combined_df["cluster"] = clusters
        color_column = "cluster"
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings_dir = Path(args.embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique name based on dataset path
        dataset_name = Path(args.dataset).name
        output_path = embeddings_dir / f"{dataset_name}_split_embeddings.npz"
        
        # Save embeddings and metadata
        np.savez(
            output_path,
            embeddings=embeddings,
            is_train=is_train,
            text=combined_df[args.text_column].values,
        )
        print(f"Embeddings saved to {output_path}")
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(embeddings, method=args.method)
    
    # Prepare output file path
    output_file = args.output
    if output_file:
        # Convert to absolute path
        output_file = str(Path(output_file).absolute())
        
        # Add appropriate extension if missing
        if not (output_file.endswith(".html") or output_file.endswith(".png")):
            if args.static:
                output_file = f"{output_file}.png"
            else:
                output_file = f"{output_file}.html"
    
    plot_split_embeddings(
        reduced_embeddings,
        is_train,
        combined_df,
        text_column=args.text_column,
        color_column=color_column,
        title=f"Train vs. Eval Split Visualization ({args.method.upper()})",
        output_file=output_file,
        interactive=not args.static,
    )
    
    # Print the path to the output file
    if output_file:
        print(f"\nVisualization saved to: {output_file}")
        print(f"You can open this file in your browser to view the visualization.")


if __name__ == "__main__":
    main()
