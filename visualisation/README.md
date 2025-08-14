# Embedding Visualization Tools

This folder contains tools for visualizing and comparing embeddings of questions from datasets. The tools allow you to:

1. Generate embeddings for questions in datasets
2. Visualize embeddings within a single dataset
3. Visualize train vs. eval splits
4. Compare embeddings between multiple datasets
5. Generate tags for questions using Gemini Flash
6. Compare datasets based on tag distributions and visualizations

## Setup

Initialize the project and install dependencies:

```bash
cd /path/to/ADVANCED-fine-tuning/reasoning/visualisation
```

## Checking Embedding Files for Issues

If you encounter warnings about `divide by zero`, `overflow`, or `invalid value encountered in matmul` when running embedding comparison or visualization scripts, your embedding files may contain invalid vectors (NaNs, Infs, or all-zeros).

Use the provided `check_embeddings.py` script to check your `.npy` embedding files for these issues:

```bash
uv run check_embeddings.py path/to/embeddings1.npy path/to/embeddings2.npy
```

This will print, for each file:
- The shape of the embedding array
- Whether any NaN or Inf values are present
- The number of all-zero vectors (and a few example indices)

**If you find any issues, consider cleaning or regenerating your embeddings before running further analysis.**

## Usage

### 1. Generate Embeddings

Generate embeddings for questions in a dataset:

```bash
uv run generate_embeddings.py --dataset <dataset_source> --output <output_dir>
```

Arguments:
- `--dataset`, `-d`: Dataset source (Hugging Face ID or local path)
- `--output`, `-o`: Output directory for embeddings (default: ./embeddings/<dataset_name>)
- `--text-column`, `-t`: Column name containing the text to embed (default: question)
- `--model`, `-m`: Sentence transformer model to use (default: nomic-ai/modernbert-embed-base)
- `--batch-size`, `-b`: Batch size for embedding generation (default: 32)
- `--config`, `-c`: Path to config file (default: ../config/config.json)
- `--prefix`, `-p`: Prefix to add to each text item (default: 'search_query: ' for queries)

The modernbert model requires specific prefixes for different types of text:
- Use `search_query: ` (default) for questions/queries
- Use `search_document: ` for document content

Example:
```bash
# Generate embeddings for a Hugging Face dataset
uv run generate_embeddings.py --dataset "Trelis/touch-rugby-reasoning-flash-2.0-20k_chunks"

# Generate embeddings with a custom prefix
uv run generate_embeddings.py --dataset "Trelis/touch-rugby-reasoning-flash-2.0-20k_chunks" --prefix "search_document: "

# Generate embeddings for a local dataset
uv run generate_embeddings.py --dataset "../data/dataset" --text-column "question"
```

### 2. Visualize Embeddings

Visualize embeddings from a single dataset:

```bash
uv run visualize_embeddings.py --embeddings <embeddings_dir> [options]
```

Arguments:
- `--embeddings`, `-e`: Path to the directory containing embeddings
- `--method`, `-m`: Dimension reduction method (choices: pca, tsne, umap; default: tsne)
- `--components`, `-c`: Number of dimensions to reduce to (choices: 2, 3; default: 2)
- `--color-by`, `-cb`: Column to color points by (default: none)
- `--output`, `-o`: Output file path (default: display plot)
- `--interactive`, `-i`: Use interactive Plotly visualization
- `--cluster-analysis`, `-ca`: Perform cluster count analysis (elbow method)
- `--max-clusters`, `-mc`: Maximum number of clusters to try for analysis (default: 20)

Example:
```bash
# Basic visualization
uv run visualize_embeddings.py --embeddings "embeddings/touch-rugby-reasoning-flash-2.0-20k_chunks"

# Interactive visualization colored by category
uv run visualize_embeddings.py --embeddings "embeddings/touch-rugby-reasoning-flash-2.0-20k_chunks" --interactive --color-by "category"

# Save visualization to file
uv run visualize_embeddings.py --embeddings "embeddings/touch-rugby-reasoning-flash-2.0-20k_chunks" --output "visualization.png"

# Perform cluster analysis to determine optimal number of clusters
uv run visualize_embeddings.py --embeddings "embeddings/touch-rugby-reasoning-flash-2.0-20k_chunks" --cluster-analysis
```

### 3. Visualize Train vs. Eval Split

The `visualize_split.py` script allows you to visualize how your train and eval splits are distributed in the embedding space, helping you assess if your stratified splitting strategy is effective.

```bash
# Basic visualization of train vs. eval split
uv run visualize_split.py --dataset ../data/dataset

# Use UMAP instead of t-SNE for dimension reduction
uv run visualize_split.py --dataset ../data/dataset --method umap

# Show cluster assignments in the visualization
uv run visualize_split.py --dataset ../data/dataset --show-clusters

# Save the visualization to a specific file
uv run visualize_split.py --dataset ../data/dataset --output train_eval_viz

# Generate a static PNG image instead of interactive HTML
uv run visualize_split.py --dataset ../data/dataset --static
```

Arguments:
- `--dataset`: Path to the dataset with train/eval splits (required)
- `--method`: Dimension reduction method: tsne (default), umap, or pca
- `--text-column`: Column name for the text (default: "question")
- `--color-column`: Optional column name for additional coloring
- `--output`: Output file path (HTML for interactive, PNG for static)
- `--static`: Use static matplotlib visualization instead of interactive Plotly
- `--show-clusters`: Perform clustering and show cluster assignments
- `--num-clusters`: Number of clusters (0 for auto-detection using elbow method)
- `--save-embeddings`: Save generated embeddings for future use
- `--embeddings-dir`: Directory to save embeddings (default: "embeddings")

This visualization helps you:
1. Verify that your train and eval splits have good coverage across the embedding space
2. Identify potential biases in your splits
3. Understand how different question types are distributed
4. Assess if your stratified splitting strategy is working as expected

### 4. Compare Embeddings

Compare embeddings between multiple datasets (up to five):

```bash
uv run compare_embeddings.py --datasets <dataset_path_1> <dataset_path_2> [<dataset_path_3> <dataset_path_4> <dataset_path_5>]
```

The enhanced comparison tool now accepts dataset paths directly and supports visualizing up to five datasets simultaneously. It will automatically generate embeddings if they don't exist.

Arguments:
- `--datasets`, `-d`: Paths to the datasets to compare (1-5 datasets)
- `--labels`, `-l`: Optional custom labels for the datasets (defaults to last component of dataset paths)
- `--output-dir`, `-o`: Output directory for results (default: embedding_comparison_results)
- `--interactive`, `-i`: Use interactive Plotly visualization
- `--text-column`, `-t`: Column name containing the text to compare (default: question)
- `--model`, `-m`: Model to use for embeddings (default: nomic-ai/modernbert-embed-base)
- `--pairwise`: Enable pairwise comparisons between datasets (default: disabled)
- `--method`: Dimension reduction method (choices: tsne, umap, pca; default: tsne)

Example:
```bash
# Compare two datasets (automatically generates embeddings if needed)
uv run compare_embeddings.py --datasets "data/final_dataset" "Trelis/touch-rugby-reasoning-flash-2.0-5k_chunks"

# Compare multiple datasets with custom labels
uv run compare_embeddings.py --datasets "data/final_dataset" "data/raw_dataset" "data/augmented_dataset" "data/filtered_dataset" --labels "Final" "Raw" "Augmented" "Filtered" --interactive

# Enable pairwise comparisons (in addition to the combined visualization)
uv run compare_embeddings.py --datasets "data/final_dataset" "data/raw_dataset" --pairwise
```
e.g.
```bash
uv run compare_embeddings.py --datasets Trelis/touch-rugby-sonnet-3.5-5k_chunks Trelis/touch-rugby-o4-mini-5k_chunks Trelis/touch-rugby-pro-2.5-5k_chunks Trelis/touch-rugby-flash-2.0-5k_chunks --interactive
```

Features:
- Automatically generates embeddings if they don't exist
- Uses the last component of each dataset path as the label in visualizations
- Supports custom labels with the `--labels` parameter
- Creates a combined visualization of all datasets by default
- Optional pairwise comparisons with the `--pairwise` flag
- Robust handling of sparse data and zero vectors
- Adaptive dimension reduction parameters based on dataset size
- Clear error messages when issues are detected
- Generates both static and interactive visualizations

## Output

The comparison script generates several outputs in the `embedding_comparison_results` directory:

1. **Combined Embedding Visualization**: A plot showing the embeddings from all datasets in the same space
2. **Pairwise Comparisons** (if enabled with `--pairwise`):
   - **Similarity Reports**: CSV files containing pairs of questions from each dataset pair with their similarity scores
   - **Similarity Histograms**: Histograms showing the distribution of cosine similarities for each dataset pair

## Example Workflow

```bash
# Compare datasets directly (no need to generate embeddings separately)
uv run compare_embeddings.py --datasets "data/final_dataset" "Trelis/touch-rugby-reasoning-flash-2.0-5k_chunks" "Trelis/touch-rugby-sonnet-3.5-5k_chunks" "Trelis/touch-rugby-o4-mini-5k_chunks" --interactive

# The script will:
# 1. Generate embeddings for each dataset if they don't exist
# 2. Generate a combined visualization showing all datasets together
# 3. Compare all pairs of datasets if --pairwise is specified
```

This will generate a complete analysis of the similarities and differences between questions in all datasets.

### 5. Generate Tags

Generate tags for questions in a dataset using Gemini Flash:

```bash
uv run generate_tags.py --dataset <dataset_source> --output <output_dir>
```

Arguments:
- `--dataset`, `-d`: Dataset source (Hugging Face ID or local path)
- `--output`, `-o`: Output directory for tags (default: ./tags/<dataset_name>)
- `--text-column`, `-t`: Column name containing the text to tag (default: question)
- `--model`, `-m`: Gemini model to use (default: gemini-flash)
- `--max-tags`: Maximum number of tags to generate per text (default: 5)
- `--batch-size`, `-b`: Batch size for tag generation (default: 32)
- `--max-workers`, `-w`: Maximum number of worker threads (default: 4)
- `--config`, `-c`: Path to config file (default: ../config/config.json)
- `--system-prompt`, `-p`: Custom system prompt for tag generation

Example:
```bash
# Generate tags for a Hugging Face dataset
uv run generate_tags.py --dataset "Trelis/touch-rugby-reasoning-flash-2.0-20k_chunks"

# Generate tags for a local dataset with custom settings
uv run generate_tags.py --dataset "../data/dataset" --text-column "question" --max-tags 7
```

### 6. Compare Tags

Compare tags between multiple datasets and create interactive visualizations:

```bash
uv run compare_tags.py --datasets <dataset_path_1> <dataset_path_2> [<dataset_path_3> <dataset_path_4> <dataset_path_5>]
```

Arguments:
- `--datasets`, `-d`: Paths to the datasets to compare (1-5 datasets)
- `--labels`, `-l`: Optional custom labels for the datasets (defaults to last component of dataset paths)
- `--output-dir`, `-o`: Output directory for results (default: tag_comparison_results)
- `--text-column`, `-t`: Column name containing the text to compare (default: question)
- `--model`, `-m`: Model to use for tag generation (default: google/gemini-2.0-flash-001)
- `--max-tags`: Maximum number of tags to generate per text (default: 5)
- `--min-tag-freq`: Minimum frequency for a tag to be included (default: 2)
- `--method`: Dimension reduction method (choices: tsne, umap, pca; default: tsne)
- `--system-prompt`, `-p`: Custom system prompt for tag generation
- `--pairwise`: Enable pairwise comparisons between datasets (default: disabled)

Example:
```bash
# Compare two datasets (automatically generates tags if needed)
uv run compare_tags.py --datasets "data/final_dataset" "Trelis/touch-rugby-reasoning-flash-2.0-5k_chunks"

# Compare multiple datasets with custom labels
uv run compare_tags.py --datasets "data/final_dataset" "data/raw_dataset" "data/augmented_dataset" "data/filtered_dataset" --labels "Final" "Raw" "Augmented" "Filtered"

# Enable pairwise comparisons (in addition to the combined visualization)
uv run compare_tags.py --datasets "data/final_dataset" "data/raw_dataset" --pairwise
```

Features:
- Automatically generates tags if they don't exist
- Uses the last component of each dataset path as the label in visualizations
- Supports custom labels with the `--labels` parameter
- Creates a combined visualization of all datasets by default
- Optional pairwise comparisons with the `--pairwise` flag
- Robust error handling for sparse data and zero vectors
- Adaptive dimension reduction parameters based on dataset size
- Clear error messages when issues are detected
- Generates interactive visualizations for tag distributions and tag-based dataset comparisons
- Allows hovering over points to see the original question and its tags

## Output

The tag comparison script generates several outputs:

1. **Combined Tag Distribution Plot**: Interactive HTML visualization showing the distribution of top tags across all datasets
2. **Combined Tag-Based Vector Visualization**: Interactive plot showing all datasets in a reduced dimension space based on their tags
3. **Pairwise Comparisons** (if enabled with `--pairwise`):
   - **Tag Distribution Plots**: Interactive HTML visualizations showing the distribution of top tags for each dataset pair
   - **Tag-Based Vector Visualizations**: Interactive plots showing each dataset pair in a reduced dimension space

## Example Workflow

```bash
# Generate tags for a dataset
uv run generate_tags.py --dataset "data/final_dataset"

# Compare datasets based on tags
uv run compare_tags.py --datasets "data/final_dataset" "Trelis/touch-rugby-reasoning-flash-2.0-5k_chunks" "Trelis/touch-rugby-sonnet-3.5-5k_chunks" "Trelis/touch-rugby-o4-mini-5k_chunks"
```

This will generate a complete analysis of the tag distributions and similarities between questions in all datasets, with interactive visualizations that allow exploring individual questions and their associated tags.
