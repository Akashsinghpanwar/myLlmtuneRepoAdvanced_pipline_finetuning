#!/usr/bin/env python
"""
Visualize embeddings from a single dataset.
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

from embed_utils import load_embeddings


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
            # Use a valid n_jobs value (1 is safe and works with random_state)
            "n_jobs": 1
        }
        default_kwargs.update(kwargs)
        reducer = umap.UMAP(n_components=n_components, **default_kwargs)
    else:
        raise ValueError(f"Unknown dimension reduction method: {method}")
    
    # Use ensure_all_finite instead of force_all_finite to avoid deprecation warning
    return reducer.fit_transform(embeddings)


def plot_embeddings_matplotlib(
    reduced_embeddings, 
    df, 
    text_column="question",
    color_column=None,
    title="Embedding Visualization",
    output_file=None,
    figsize=(12, 10),
):
    """
    Plot reduced embeddings using matplotlib.
    
    Args:
        reduced_embeddings: Reduced embeddings array (2D)
        df: DataFrame containing the original data
        text_column: Column name containing the text that was embedded
        color_column: Column name to use for coloring points
        title: Plot title
        output_file: Output file path (if None, display the plot)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if color_column and color_column in df.columns:
        # Get unique categories and assign colors
        categories = df[color_column].unique()
        
        # Plot each category with a different color
        for i, category in enumerate(categories):
            mask = df[color_column] == category
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                alpha=0.7,
                label=category,
            )
        plt.legend(title=color_column)
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_embeddings_plotly(
    reduced_embeddings, 
    df, 
    text_column="question",
    color_column=None,
    title="Embedding Visualization",
    output_file=None,
):
    """
    Plot reduced embeddings using Plotly for interactive visualization.
    
    Args:
        reduced_embeddings: Reduced embeddings array (2D or 3D)
        df: DataFrame containing the original data
        text_column: Column name containing the text that was embedded
        color_column: Column name to use for coloring points
        title: Plot title
        output_file: Output file path (if None, display the plot)
    """
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(
        reduced_embeddings, 
        columns=[f"Dimension {i+1}" for i in range(reduced_embeddings.shape[1])]
    )
    
    # Add text and color columns
    plot_df["text"] = df[text_column].values
    
    if color_column and color_column in df.columns:
        plot_df["color"] = df[color_column].values
        color = "color"
    else:
        color = None
    
    # Create the plot
    if reduced_embeddings.shape[1] == 3:
        # 3D plot
        fig = px.scatter_3d(
            plot_df,
            x="Dimension 1",
            y="Dimension 2",
            z="Dimension 3",
            color=color,
            hover_data=["text"],
            title=title,
        )
    else:
        # 2D plot
        fig = px.scatter(
            plot_df,
            x="Dimension 1",
            y="Dimension 2",
            color=color,
            hover_data=["text"],
            title=title,
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        legend_title_text=color_column if color_column else "",
        height=800,
        width=1000,
    )
    
    # Save or display
    if output_file:
        fig.write_html(output_file)
        print(f"Interactive plot saved to {output_file}")
    else:
        fig.show()


def analyze_cluster_count(embeddings, max_k=20, output_file=None):
    """
    Perform elbow method analysis to help determine the optimal number of clusters.
    
    Args:
        embeddings: Embeddings array
        max_k: Maximum number of clusters to try
        output_file: Output file path (if None, display the plot)
        
    Returns:
        Recommended number of clusters
    """
    print("Performing elbow method analysis...")
    
    # Range of k values to try
    k_values = range(2, min(max_k + 1, len(embeddings) // 5))
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        # Run k-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score if there are at least 2 clusters
        if k > 1 and len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(embeddings, labels))
        else:
            silhouette_scores.append(0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow method plot
    ax1.plot(list(k_values), inertias, 'o-', color='blue')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (Sum of squared distances)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(alpha=0.3)
    
    # Silhouette score plot
    ax2.plot(list(k_values), silhouette_scores, 'o-', color='green')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Automatically determine recommended k based on elbow method
    # Calculate the rate of change of inertia
    inertia_changes = np.diff(inertias)
    inertia_ratios = np.abs(np.diff(inertia_changes) / inertia_changes[:-1])
    
    # Find the point where the rate of change significantly decreases (elbow)
    elbow_k = np.argmax(inertia_ratios) + 2
    
    # Find the k with highest silhouette score
    silhouette_k = np.argmax(silhouette_scores) + 2
    
    # Add annotations for recommended k values
    ax1.axvline(x=elbow_k, linestyle='--', color='red', alpha=0.7)
    ax1.text(elbow_k + 0.2, max(inertias) * 0.9, f'Elbow k={elbow_k}', color='red')
    
    ax2.axvline(x=silhouette_k, linestyle='--', color='red', alpha=0.7)
    ax2.text(silhouette_k + 0.2, max(silhouette_scores) * 0.9, f'Best k={silhouette_k}', color='red')
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Cluster analysis saved to {output_file}")
    else:
        plt.show()
    
    # Return the silhouette method's recommendation as it's often more reliable
    return silhouette_k


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings from a dataset")
    parser.add_argument(
        "--embeddings", 
        "-e", 
        required=True,
        help="Path to the directory containing embeddings"
    )
    parser.add_argument(
        "--method", 
        "-m", 
        choices=["pca", "tsne", "umap"],
        default="tsne",
        help="Dimension reduction method (default: tsne)"
    )
    parser.add_argument(
        "--components", 
        "-c", 
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of dimensions to reduce to (default: 2)"
    )
    parser.add_argument(
        "--color-by", 
        "-cb",
        default=None,
        help="Column to color points by (default: none)"
    )
    parser.add_argument(
        "--output", 
        "-o", 
        default=None,
        help="Output file path (default: display plot)"
    )
    parser.add_argument(
        "--interactive", 
        "-i", 
        action="store_true",
        help="Use interactive Plotly visualization"
    )
    parser.add_argument(
        "--cluster-analysis", 
        "-ca", 
        action="store_true",
        help="Perform cluster count analysis (elbow method)"
    )
    parser.add_argument(
        "--max-clusters", 
        "-mc", 
        type=int,
        default=20,
        help="Maximum number of clusters to try for analysis (default: 20)"
    )
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings, metadata, df = load_embeddings(args.embeddings)
    
    # Get text column from metadata
    text_column = metadata.get("text_column", "question")
    
    # Perform cluster analysis if requested
    if args.cluster_analysis:
        cluster_output_file = None
        if args.output:
            # Use the same base name but different extension/suffix
            output_path = Path(args.output)
            cluster_output_file = output_path.with_stem(f"{output_path.stem}_cluster_analysis")
            if not cluster_output_file.suffix:
                cluster_output_file = cluster_output_file.with_suffix(".png")
        
        recommended_k = analyze_cluster_count(
            embeddings, 
            max_k=args.max_clusters,
            output_file=cluster_output_file
        )
        print(f"Recommended number of clusters: {recommended_k}")
    
    # Reduce dimensions
    print(f"Reducing dimensions using {args.method}...")
    reduced_embeddings = reduce_dimensions(
        embeddings, 
        method=args.method, 
        n_components=args.components
    )
    
    # Determine output file extension
    output_file = args.output
    if output_file and args.interactive and not output_file.endswith(".html"):
        output_file = f"{output_file}.html"
    elif output_file and not args.interactive and not any(
        output_file.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".pdf"]
    ):
        output_file = f"{output_file}.png"
    
    # Create plot title
    model_name = metadata.get("model_name", "").split("/")[-1]
    title = f"Embedding Visualization ({model_name}, {args.method.upper()})"
    
    # Plot
    if args.interactive:
        plot_embeddings_plotly(
            reduced_embeddings,
            df,
            text_column=text_column,
            color_column=args.color_by,
            title=title,
            output_file=output_file,
        )
    else:
        plot_embeddings_matplotlib(
            reduced_embeddings,
            df,
            text_column=text_column,
            color_column=args.color_by,
            title=title,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
