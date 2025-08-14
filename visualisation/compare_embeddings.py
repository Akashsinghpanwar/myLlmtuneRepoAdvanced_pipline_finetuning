#!/usr/bin/env python
"""
Compare embeddings between datasets and visualize their similarity.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
from sklearn.metrics.pairwise import cosine_similarity

from embed_utils import (
    load_embeddings,
    compute_pairwise_similarities,
    find_closest_embeddings,
)


def plot_similarity_histogram(
    similarities,
    title="Distribution of Cosine Similarities",
    output_file=None,
    figsize=(10, 6),
):
    """
    Plot a histogram of similarity scores.
    
    Args:
        similarities: Array of similarity scores
        title: Plot title
        output_file: Output file path (if None, display the plot)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(similarities, bins=50, alpha=0.7)
    
    # Add mean and median lines
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    plt.axvline(mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='g', linestyle='--', label=f'Median: {median_sim:.3f}')
    
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def plot_combined_embeddings(
    embeddings_list,
    df_list,
    text_columns,
    dataset_labels,
    method="umap",
    n_components=2,
    title="Combined Embedding Visualization",
    output_file=None,
    interactive=False,
):
    """
    Plot combined embeddings from multiple datasets.
    
    Args:
        embeddings_list: List of embeddings arrays from each dataset
        df_list: List of DataFrames containing data from each dataset
        text_columns: List of column names for text in each dataset
        dataset_labels: List of labels for each dataset
        method: Dimension reduction method ('tsne' or 'umap')
        n_components: Number of components to reduce to
        title: Plot title
        output_file: Output file path (if None, display the plot)
        interactive: Whether to use interactive Plotly visualization
        
    Raises:
        ValueError: If embeddings are invalid or dimension reduction fails
    """
    # Combine embeddings
    combined_embeddings = np.vstack(embeddings_list)
    
    # Check for NaN or Inf values
    if np.isnan(combined_embeddings).any() or np.isinf(combined_embeddings).any():
        print("Warning: NaN or Inf values found in embeddings. Replacing with zeros.")
        combined_embeddings = np.nan_to_num(combined_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for zero vectors
    zero_vectors = np.all(combined_embeddings == 0, axis=1)
    if np.any(zero_vectors):
        zero_count = np.sum(zero_vectors)
        total_count = combined_embeddings.shape[0]
        zero_percent = (zero_count / total_count) * 100
        print(f"Warning: {zero_count} zero vectors found ({zero_percent:.2f}% of data).")
        
        if zero_percent > 50:
            raise ValueError(
                f"Too many zero vectors ({zero_percent:.2f}% of data). "
                "This suggests issues with the embedding generation. "
                "Please check your data and embedding process."
            )
    
    # Check for sparse data
    non_zero_features = np.count_nonzero(combined_embeddings, axis=1)
    if np.median(non_zero_features) < 0.05 * combined_embeddings.shape[1]:
        print("Warning: Data appears to be very sparse. Dimension reduction may not produce meaningful results.")
    
    try:
        # Reduce dimensions
        if method.lower() == "tsne":
            # Adjust perplexity based on dataset size
            perplexity = min(30, max(5, combined_embeddings.shape[0] // 10))
            print(f"Using TSNE with perplexity={perplexity}")
            reducer = TSNE(
                n_components=n_components, 
                perplexity=perplexity, 
                n_iter=1000, 
                random_state=42,
                init='pca',  # Use PCA initialization for better stability
                learning_rate='auto'  # Auto learning rate for better convergence
            )
        else:  # umap
            # Adjust n_neighbors based on dataset size
            n_neighbors = min(15, max(2, combined_embeddings.shape[0] // 10))
            print(f"Using UMAP with n_neighbors={n_neighbors}")
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42
            )
        
        # Perform dimension reduction
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                reduced_embeddings = reducer.fit_transform(combined_embeddings)
        except Exception as e:
            # If the selected method fails, try PCA as fallback
            if method.lower() != "pca":
                print(f"Warning: {method} failed with error: {e}. Trying PCA instead.")
                from sklearn.decomposition import PCA
                try:
                    reducer = PCA(n_components=min(n_components, combined_embeddings.shape[0], combined_embeddings.shape[1]))
                    reduced_embeddings = reducer.fit_transform(combined_embeddings)
                except Exception as pca_error:
                    raise ValueError(f"Both {method} and PCA failed. Original error: {e}, PCA error: {pca_error}")
            else:
                raise ValueError(f"Dimension reduction failed: {e}")
        
        # Split back into original datasets
        start_idx = 0
        reduced_list = []
        for embeddings in embeddings_list:
            end_idx = start_idx + len(embeddings)
            reduced_list.append(reduced_embeddings[start_idx:end_idx])
            start_idx = end_idx
        
        if interactive:
            # Create DataFrames for plotting
            plot_df_list = []
            
            for i, (reduced, df, text_column, label) in enumerate(zip(reduced_list, df_list, text_columns, dataset_labels)):
                plot_df = pd.DataFrame(
                    reduced, 
                    columns=[f"Dimension {i+1}" for i in range(n_components)]
                )
                plot_df["text"] = df[text_column].values
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
                    hover_data=["text"],
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x="Dimension 1",
                    y="Dimension 2",
                    color="dataset",
                    hover_data=["text"],
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                width=1000,
            )
            
            # Save or display
            if output_file:
                fig.write_html(output_file)
                print(f"Interactive plot saved to {output_file}")
            else:
                fig.show()
        else:
            # Matplotlib visualization
            plt.figure(figsize=(12, 10))
            
            # Plot each dataset with a different color
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Default colors for up to 5 datasets
            
            for i, (reduced, label) in enumerate(zip(reduced_list, dataset_labels)):
                plt.scatter(
                    reduced[:, 0], 
                    reduced[:, 1], 
                    alpha=0.7, 
                    label=label,
                    color=colors[i % len(colors)]
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
    except Exception as e:
        raise ValueError(f"Error in dimension reduction or plotting: {e}")


def create_similarity_report(
    df_a,
    df_b,
    text_column_a,
    text_column_b,
    closest_indices,
    similarity_scores,
    label_a,
    label_b,
    output_file=None,
):
    """
    Create a report of closest questions between datasets.
    
    Args:
        df_a: DataFrame containing data from first dataset
        df_b: DataFrame containing data from second dataset
        text_column_a: Column name for text in first dataset
        text_column_b: Column name for text in second dataset
        closest_indices: Indices of closest questions in dataset B for each question in dataset A
        similarity_scores: Similarity scores for closest questions
        label_a: Label for the first dataset
        label_b: Label for the second dataset
        output_file: Output file path (if None, return DataFrame)
        
    Returns:
        DataFrame with similarity report if output_file is None
    """
    # Create report DataFrame
    report_data = []
    
    for i in range(len(df_a)):
        question_a = df_a[text_column_a].iloc[i]
        closest_idx = closest_indices[i][0]  # Get the first (closest) index
        question_b = df_b[text_column_b].iloc[closest_idx]
        similarity = similarity_scores[i][0]  # Get the first (highest) similarity score
        
        report_data.append({
            f"question_{label_a}": question_a,
            f"question_{label_b}": question_b,
            "similarity": similarity,
        })
    
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values("similarity", ascending=False).reset_index(drop=True)
    
    if output_file:
        report_df.to_csv(output_file, index=False)
        print(f"Similarity report saved to {output_file}")
    
    return report_df


def get_or_generate_embeddings(dataset_path, model_name="nomic-ai/modernbert-embed-base", text_column="question"):
    """
    Get embeddings for a dataset, generating them if they don't exist.
    
    Args:
        dataset_path: Path to the dataset
        model_name: Model to use for embeddings
        text_column: Column containing the text to embed
        
    Returns:
        Tuple of (embeddings, metadata, dataframe)
    """
    # Extract dataset name from path
    dataset_name = dataset_path.split("/")[-1]
    
    # Check if embeddings already exist
    embeddings_dir = Path("embeddings") / dataset_name
    
    if embeddings_dir.exists() and (embeddings_dir / "embeddings.npy").exists():
        print(f"Found existing embeddings for {dataset_name} at {embeddings_dir}")
        return load_embeddings(str(embeddings_dir))
    
    # Generate embeddings using the generate_embeddings.py script
    print(f"Generating embeddings for {dataset_name}...")
    script_path = Path(__file__).parent / "generate_embeddings.py"
    
    # Run the script as a subprocess
    cmd = [
        sys.executable, 
        str(script_path),
        "--dataset", dataset_path,
        "--text-column", text_column,
        "--model", model_name
    ]
    
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate embeddings for {dataset_path}")
    
    # Load the newly generated embeddings
    return load_embeddings(str(embeddings_dir))


def compare_dataset_pair(dataset_a, dataset_b, output_dir, interactive=False, model_name=None):
    """
    Compare a pair of datasets and generate similarity reports and visualizations.
    
    Args:
        dataset_a: Tuple of (embeddings, metadata, df) for first dataset
        dataset_b: Tuple of (embeddings, metadata, df) for second dataset
        output_dir: Directory to save outputs
        interactive: Whether to use interactive Plotly visualization
        model_name: Model name used for embeddings (for display)
        
    Returns:
        DataFrame with similarity report
    """
    embeddings_a, metadata_a, df_a = dataset_a
    embeddings_b, metadata_b, df_b = dataset_b
    
    # Get text columns from metadata
    text_column_a = metadata_a.get("text_column", "question")
    text_column_b = metadata_b.get("text_column", "question")
    
    # Get labels from metadata
    label_a = metadata_a.get("dataset_name", "Dataset A")
    label_b = metadata_b.get("dataset_name", "Dataset B")
    
    # Check if embeddings have the same dimensions
    if embeddings_a.shape[1] != embeddings_b.shape[1]:
        raise ValueError(
            f"Embeddings have different dimensions: {embeddings_a.shape[1]} vs {embeddings_b.shape[1]}"
        )
    
    # Find closest questions in dataset B for each question in dataset A
    print(f"Finding closest questions between {label_a} and {label_b}...")
    closest_indices, similarity_scores = find_closest_embeddings(
        embeddings_a, embeddings_b, top_k=1
    )
    
    # Create similarity report
    print("Creating similarity report...")
    report_df = create_similarity_report(
        df_a,
        df_b,
        text_column_a,
        text_column_b,
        closest_indices,
        similarity_scores,
        label_a,
        label_b,
        output_file=output_dir / f"similarity_report_{label_a}_vs_{label_b}.csv",
    )
    
    # Plot similarity histogram
    print("Plotting similarity distribution...")
    if not model_name:
        model_name = metadata_a.get("model_name", "").split("/")[-1]
    
    histogram_title = f"Distribution of Cosine Similarities: {label_a} vs {label_b} ({model_name})"
    plot_similarity_histogram(
        similarity_scores.flatten(),
        title=histogram_title,
        output_file=output_dir / f"similarity_histogram_{label_a}_vs_{label_b}.png",
    )
    
    return report_df


def main():
    parser = argparse.ArgumentParser(description="Compare embeddings between datasets")
    parser.add_argument(
        "--datasets", 
        "-d", 
        required=True,
        nargs="+",
        help="Paths to the datasets to compare (1-5 datasets)"
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        default=None,
        help="Optional labels for the datasets (defaults to last component of dataset path)"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        default="embedding_comparison_results",
        help="Output directory for results (default: embedding_comparison_results)"
    )
    parser.add_argument(
        "--interactive", 
        "-i", 
        action="store_true",
        help="Use interactive Plotly visualization"
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
        default="nomic-ai/modernbert-embed-base",
        help="Model to use for embeddings (default: nomic-ai/modernbert-embed-base)"
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Enable pairwise comparisons between datasets (default: disabled)"
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimension reduction method (default: tsne)"
    )
    args = parser.parse_args()
    
    # Validate number of datasets
    if len(args.datasets) > 5:
        print("Warning: Only up to 5 datasets are supported for visualization. Using the first 5.")
        args.datasets = args.datasets[:5]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load embeddings for each dataset
    datasets = []
    for dataset_path in args.datasets:
        print(f"Processing dataset: {dataset_path}")
        try:
            embeddings, metadata, df = get_or_generate_embeddings(
                dataset_path, 
                model_name=args.model,
                text_column=args.text_column
            )
            
            # Add dataset name to metadata for later use
            metadata["dataset_name"] = dataset_path.split("/")[-1]
            datasets.append((embeddings, metadata, df))
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {e}")
            print(f"Skipping this dataset.")
    
    if not datasets:
        print("No valid datasets to process. Exiting.")
        return
    
    # Create labels for datasets
    if args.labels:
        if len(args.labels) < len(datasets):
            # Extend labels with default values if needed
            args.labels.extend([f"Dataset {i+1+len(args.labels)}" for i in range(len(datasets) - len(args.labels))])
        labels = args.labels[:len(datasets)]
    else:
        # Use last component of dataset path as label
        labels = [data[1]["dataset_name"] for data in datasets]
    
    # Update dataset metadata with custom labels
    for i, (embeddings, metadata, df) in enumerate(datasets):
        metadata["dataset_name"] = labels[i]
    
    # Compare all pairs of datasets if requested
    if args.pairwise and len(datasets) >= 2:
        print("Comparing dataset pairs...")
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                print(f"Comparing {labels[i]} vs {labels[j]}...")
                try:
                    compare_dataset_pair(
                        datasets[i], 
                        datasets[j], 
                        output_dir, 
                        interactive=args.interactive,
                        model_name=args.model.split("/")[-1]
                    )
                except Exception as e:
                    print(f"Error comparing {labels[i]} vs {labels[j]}: {e}")
                    print(f"Skipping this comparison.")
    
    # Plot combined embeddings for all datasets
    if len(datasets) > 1:
        print("Plotting combined embeddings for all datasets...")
        embeddings_list = [data[0] for data in datasets]
        df_list = [data[2] for data in datasets]
        text_columns = [data[1].get("text_column", "question") for data in datasets]
        
        model_name = args.model.split("/")[-1]
        combined_title = f"Combined Embedding Visualization ({model_name})"
        combined_plot_file = output_dir / ("combined_embeddings.html" if args.interactive else "combined_embeddings.png")
        
        try:
            plot_combined_embeddings(
                embeddings_list,
                df_list,
                text_columns,
                labels,
                method=args.method,
                n_components=2,
                title=combined_title,
                output_file=combined_plot_file,
                interactive=args.interactive,
            )
        except ValueError as e:
            print(f"Error creating combined visualization: {e}")
            print("Skipping combined visualization.")
    
    print(f"All results saved to {output_dir}")
    
    # Print summary statistics for each dataset pair if requested
    if args.pairwise and len(datasets) >= 2:
        print("\nSimilarity Statistics:")
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                try:
                    embeddings_a = datasets[i][0]
                    embeddings_b = datasets[j][0]
                    label_a = labels[i]
                    label_b = labels[j]
                    
                    # Compute similarities
                    similarities = cosine_similarity(embeddings_a, embeddings_b)
                    closest_indices = np.argmax(similarities, axis=1)
                    closest_similarities = np.take_along_axis(
                        similarities, 
                        np.expand_dims(closest_indices, axis=1), 
                        axis=1
                    )
                    
                    mean_sim = np.mean(closest_similarities)
                    median_sim = np.median(closest_similarities)
                    min_sim = np.min(closest_similarities)
                    max_sim = np.max(closest_similarities)
                    
                    print(f"\n{label_a} vs {label_b}:")
                    print(f"Mean similarity: {mean_sim:.4f}")
                    print(f"Median similarity: {median_sim:.4f}")
                    print(f"Min similarity: {min_sim:.4f}")
                    print(f"Max similarity: {max_sim:.4f}")
                except Exception as e:
                    print(f"Error computing similarity statistics for {label_a} vs {label_b}: {e}")


if __name__ == "__main__":
    main()
