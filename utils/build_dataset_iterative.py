#!/usr/bin/env python
"""
Iteratively build a dataset with smart sampling and deduplication based on embedding similarity.

This script builds on generate_qa.py to create a dataset through iterative sampling,
calculating embeddings to detect and remove duplicates, and tracking acceptance rate
to determine when to stop sampling.
"""
import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables and API keys
load_dotenv()


def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    try:
        import datasets
        import huggingface_hub
        import sentence_transformers
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Required dependencies not installed. Installing now...")
        os.system("uv add datasets huggingface-hub sentence-transformers scikit-learn")
        print("Please restart the script after installation.")
        sys.exit(1)


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from specified config file or default config/config.json."""
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = Path(__file__).parent.parent / "config" / "config.json"
    
    if not config_path.exists() or not config_path.is_file():
        print(f"Config file not found: {config_path}")
        print("Please ensure config/config.json exists or pass --config <path>.")
        sys.exit(1)
    
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def setup_directories(base_dir: Optional[str] = None) -> Dict[str, Path]:
    """Set up the necessary directories for the iterative dataset building process."""
    if base_dir:
        base_path = Path(base_dir)
    else:
        base_path = Path(__file__).parent.parent
    
    directories = {
        "data": base_path / "data",
        "qa": base_path / "data" / "qa",
        "text": base_path / "data" / "text",
        "chunks": base_path / "data" / "chunks",
        "summaries": base_path / "data" / "summaries",
        "iterations": base_path / "data" / "iterations",
        "embeddings": base_path / "data" / "embeddings",
        "final_dataset": base_path / "data" / "final_dataset"
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def run_qa_generation(config: Dict[str, Any], iteration: int, doc_name: Optional[str] = None, test: bool = False, force: bool = False) -> Path:
    """
    Run the QA generation script for a specified document or all documents.
    
    Args:
        config: Configuration dictionary
        iteration: Current iteration number
        doc_name: Optional document name to process
        test: Whether to run in test mode
        force: Whether to force regeneration even if data exists
        
    Returns:
        Path to the generated QA data directory
    """
    import subprocess
    import shutil
    
    # Create iteration directory for tracking purposes
    iteration_dir = Path(__file__).parent.parent / "data" / "iterations" / f"iteration_{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an iteration-specific QA directory
    iteration_qa_dir = iteration_dir / "qa"
    
    # Check if this iteration already exists and we're not forcing regeneration
    if iteration_qa_dir.exists() and not force:
        print(f"Iteration {iteration} data already exists. Reusing existing data. Use --force to regenerate.")
        return iteration_qa_dir
    
    # Clear existing data if we're regenerating
    if iteration_qa_dir.exists():
        shutil.rmtree(iteration_qa_dir)
    iteration_qa_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a modified config with iteration-specific QA directory
    iteration_config = config.copy()
    
    # Make sure we have the qa section
    if "qa" not in iteration_config:
        iteration_config["qa"] = {}
    
    # Add iteration number to track the source
    iteration_config["qa"]["iteration"] = iteration
    
    # Specify the output directory in the config
    iteration_config["qa"]["output_dir"] = str(iteration_qa_dir)
    
    # Save temporary config
    temp_config_file = iteration_dir / "config.json"
    with temp_config_file.open("w", encoding="utf-8") as f:
        json.dump(iteration_config, f, indent=2)
    
    # Command to run the generate_qa.py script
    cmd = [
        "uv", "run", "python", 
        str(Path(__file__).parent / "generate_qa.py"),
        "--config", str(temp_config_file)
    ]
    
    # Add options
    if doc_name:
        cmd.extend(["--doc", doc_name])
    if test:
        cmd.append("--test")
    if force:
        cmd.append("--force")
    
    # Run the command
    print(f"Running QA generation for iteration {iteration}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running QA generation: {result.stderr}")
        sys.exit(1)
    
    return iteration_qa_dir


def collect_qa_data(qa_dir: Path) -> List[Dict[str, Any]]:
    """
    Collect QA data from the given directory.
    
    Args:
        qa_dir: Path to the QA directory
        
    Returns:
        List of QA items
    """
    qa_data = []
    
    # Iterate through document directories
    for doc_dir in qa_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        
        # Get document name
        doc_name = doc_dir.name
        
        # Iterate through QA files
        for qa_file in doc_dir.glob("chunk_*_qa.json"):
            try:
                with qa_file.open("r", encoding="utf-8") as f:
                    qa_result = json.load(f)
                
                chunk_id = qa_result.get("chunk_id")
                
                # Process each Q&A pair
                for qa_pair in qa_result.get("qa_pairs", []):
                    qa_data.append({
                        "document": doc_name,
                        "chunk_id": chunk_id,
                        "question": qa_pair.get("question", ""),
                        "answer": qa_pair.get("answer", ""),
                        "evaluation_criteria": qa_pair.get("evaluation_criteria", ""),
                        "difficulty": qa_pair.get("difficulty", 0),
                        "category": qa_pair.get("category", ""),
                        "model": qa_result.get("model", ""),
                        "iteration": qa_result.get("iteration", 0)
                    })
            except Exception as e:
                print(f"Error processing Q&A file {qa_file}: {e}")
    
    return qa_data


def calculate_embeddings(questions: List[str], model_name: str = "nomic-ai/modernbert-embed-base") -> np.ndarray:
    """
    Calculate embeddings for a list of questions.
    
    Args:
        questions: List of questions to embed
        model_name: Embedding model name
        
    Returns:
        Array of embeddings
    """
    from sentence_transformers import SentenceTransformer
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Add prefix for modernbert
    if "modernbert" in model_name.lower():
        prefixed_questions = [f"search_query: {q}" for q in questions]
    else:
        prefixed_questions = questions
    
    # Generate embeddings
    embeddings = model.encode(
        prefixed_questions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def deduplicate_qa_data(
    new_qa_data: List[Dict[str, Any]], 
    existing_qa_data: List[Dict[str, Any]], 
    similarity_threshold: float = 0.92
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate new QA data based on embedding similarity to existing data.
    
    Args:
        new_qa_data: New QA data to deduplicate
        existing_qa_data: Existing QA data to compare against
        similarity_threshold: Threshold for considering items duplicates
        
    Returns:
        Tuple of (deduplicated QA data, statistics)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if not new_qa_data:
        return [], {"accepted": 0, "total": 0, "coverage_gain": 0}
    
    if not existing_qa_data:
        # No existing data, everything is unique
        return new_qa_data, {"accepted": len(new_qa_data), "total": len(new_qa_data), "coverage_gain": len(new_qa_data)}
    
    # Get questions
    new_questions = [item["question"] for item in new_qa_data]
    existing_questions = [item["question"] for item in existing_qa_data]
    
    # Generate embeddings
    print("Generating embeddings for deduplication...")
    new_embeddings = calculate_embeddings(new_questions)
    existing_embeddings = calculate_embeddings(existing_questions)
    
    # Handle potential numerical issues by normalizing embeddings
    new_embeddings_norm = np.copy(new_embeddings)
    existing_embeddings_norm = np.copy(existing_embeddings)
    
    # Normalize embeddings to unit length to ensure valid cosine similarity
    norms_new = np.linalg.norm(new_embeddings_norm, axis=1, keepdims=True)
    norms_new[norms_new == 0] = 1.0  # Avoid division by zero
    new_embeddings_norm = new_embeddings_norm / norms_new
    
    norms_existing = np.linalg.norm(existing_embeddings_norm, axis=1, keepdims=True)
    norms_existing[norms_existing == 0] = 1.0  # Avoid division by zero
    existing_embeddings_norm = existing_embeddings_norm / norms_existing
    
    # Calculate similarities between new and existing questions
    print("Calculating similarities...")
    similarities = np.zeros((len(new_embeddings_norm), len(existing_embeddings_norm)))
    
    # Calculate similarities in a more numerically stable way
    for i in range(len(new_embeddings_norm)):
        for j in range(len(existing_embeddings_norm)):
            # Manual dot product for better numerical stability
            dot_product = np.sum(new_embeddings_norm[i] * existing_embeddings_norm[j])
            # Clip to valid cosine similarity range
            similarities[i, j] = np.clip(dot_product, -1.0, 1.0)
    
    # Find max similarity for each new question
    max_similarities = np.max(similarities, axis=1)
    
    # Handle any NaN values
    max_similarities = np.nan_to_num(max_similarities, nan=-1.0)
    
    # Identify unique questions
    unique_indices = np.where(max_similarities < similarity_threshold)[0]
    
    # Calculate coverage gain (questions with max similarity < 0.85)
    coverage_gain_indices = np.where(max_similarities < 0.85)[0]
    
    # Create deduplicated dataset
    deduplicated_data = [new_qa_data[i] for i in unique_indices]
    
    # Return statistics
    stats = {
        "accepted": len(unique_indices),
        "total": len(new_qa_data),
        "coverage_gain": len(coverage_gain_indices)
    }
    
    return deduplicated_data, stats


def save_iteration_stats(stats: Dict[str, Any], iteration: int, output_dir: Path) -> None:
    """
    Save statistics for the current iteration.
    
    Args:
        stats: Statistics dictionary
        iteration: Current iteration number
        output_dir: Output directory
    """
    stats_file = output_dir / "iteration_stats.json"
    
    # Load existing stats if any
    if stats_file.exists():
        with stats_file.open("r", encoding="utf-8") as f:
            all_stats = json.load(f)
    else:
        all_stats = {"iterations": []}
    
    # Add current iteration stats
    iteration_stats = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **stats
    }
    all_stats["iterations"].append(iteration_stats)
    
    # Save stats
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"Iteration {iteration} stats:")
    print(f"  Accepted: {stats['accepted']}/{stats['total']} ({stats['accepted']/max(stats['total'], 1):.2%})")
    print(f"  Coverage gain: {stats['coverage_gain']}/{stats['total']} ({stats['coverage_gain']/max(stats['total'], 1):.2%})")


def check_stopping_criteria(stats_file: Path, min_acceptance_rate: float = 0.2, min_coverage_gain: float = 0.05, look_back: int = 3) -> bool:
    """
    Check if the stopping criteria have been met.
    
    Args:
        stats_file: Path to the statistics file
        min_acceptance_rate: Minimum acceptance rate to continue
        min_coverage_gain: Minimum coverage gain rate to continue
        look_back: Number of iterations to look back for coverage gain
        
    Returns:
        True if stopping criteria met, False otherwise
    """
    if not stats_file.exists():
        return False
    
    try:
        with stats_file.open("r", encoding="utf-8") as f:
            all_stats = json.load(f)
        
        iterations = all_stats["iterations"]
        
        if len(iterations) < 2:
            return False
        
        # Check acceptance rate
        latest = iterations[-1]
        acceptance_rate = latest["accepted"] / max(latest["total"], 1)
        
        if acceptance_rate < min_acceptance_rate:
            print(f"Stopping criterion met: acceptance rate {acceptance_rate:.2%} < {min_acceptance_rate:.2%}")
            return True
        
        # Check coverage gain over the last few iterations
        if len(iterations) >= look_back:
            recent_iterations = iterations[-look_back:]
            avg_coverage_gain = np.mean([it["coverage_gain"] / max(it["total"], 1) for it in recent_iterations])
            
            if avg_coverage_gain < min_coverage_gain:
                print(f"Stopping criterion met: average coverage gain {avg_coverage_gain:.2%} < {min_coverage_gain:.2%}")
                return True
        
        return False
    
    except Exception as e:
        print(f"Error checking stopping criteria: {e}")
        return False


def save_final_dataset(qa_data: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save the final dataset.
    
    Args:
        qa_data: List of QA data items
        output_dir: Output directory
    """
    from datasets import Dataset
    
    # Convert to dataset
    dataset = Dataset.from_list(qa_data)
    
    # Save to disk
    dataset.save_to_disk(str(output_dir))
    print(f"Final dataset saved to {output_dir} with {len(qa_data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Iteratively build a dataset with smart sampling and deduplication")
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    parser.add_argument("--doc", "-d", type=str, help="Specific document to process")
    parser.add_argument("--max-iterations", "-m", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--test", "-t", action="store_true", help="Run in test mode (1 example per chunk)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size for each iteration")
    parser.add_argument("--similarity-threshold", "-s", type=float, default=0.92, help="Similarity threshold for deduplication")
    parser.add_argument("--min-acceptance-rate", "-a", type=float, default=0.2, help="Minimum acceptance rate to continue")
    parser.add_argument("--min-coverage-gain", "-g", type=float, default=0.05, help="Minimum coverage gain rate to continue")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of data even if it exists")
    args = parser.parse_args()
    
    # Ensure dependencies
    ensure_dependencies()
    
    # Load configuration
    config = load_config(args.config)
    
    # Store the config path for passing to other scripts
    if args.config:
        config["config_path"] = args.config
    
    # Set up directories
    dirs = setup_directories()
    
    # Initialize dataset
    all_qa_data = []
    
    # Check if we have a stats file to resume from
    stats_file = dirs["iterations"] / "iteration_stats.json"
    start_iteration = 1
    
    if stats_file.exists() and not args.force:
        try:
            with stats_file.open("r", encoding="utf-8") as f:
                all_stats = json.load(f)
                
            if "iterations" in all_stats and all_stats["iterations"]:
                # Get the last completed iteration
                completed_iterations = [stat["iteration"] for stat in all_stats["iterations"]]
                if completed_iterations:
                    last_iteration = max(completed_iterations)
                    
                    # Load all data from completed iterations
                    print(f"Found {last_iteration} completed iterations. Loading existing data...")
                    for iteration in range(1, last_iteration + 1):
                        iteration_qa_dir = dirs["iterations"] / f"iteration_{iteration}" / "qa"
                        if iteration_qa_dir.exists():
                            iteration_qa_data = collect_qa_data(iteration_qa_dir)
                            
                            # Add iteration information
                            for item in iteration_qa_data:
                                item["iteration"] = iteration
                            
                            # For the first iteration, accept all data
                            if iteration == 1:
                                all_qa_data.extend(iteration_qa_data)
                            else:
                                # For subsequent iterations, apply deduplication
                                unique_qa_data, _ = deduplicate_qa_data(
                                    iteration_qa_data, 
                                    all_qa_data, 
                                    similarity_threshold=args.similarity_threshold
                                )
                                all_qa_data.extend(unique_qa_data)
                    
                    # Start from the next iteration
                    start_iteration = last_iteration + 1
                    print(f"Resuming from iteration {start_iteration} with {len(all_qa_data)} existing QA pairs")
        except Exception as e:
            print(f"Error loading existing stats: {e}")
            print("Starting from iteration 1")
    
    # Run iterations
    for iteration in range(start_iteration, args.max_iterations + 1):
        # Run QA generation with iteration-specific output directory
        qa_dir = run_qa_generation(config, iteration, args.doc, args.test, args.force)
        
        # Collect QA data from this iteration
        new_qa_data = collect_qa_data(qa_dir)
        
        # Add iteration information
        for item in new_qa_data:
            item["iteration"] = iteration
        
        # Deduplicate against existing data
        unique_qa_data, stats = deduplicate_qa_data(
            new_qa_data, 
            all_qa_data, 
            similarity_threshold=args.similarity_threshold
        )
        
        # Add unique data to our dataset
        all_qa_data.extend(unique_qa_data)
        
        # Save iteration statistics
        save_iteration_stats(stats, iteration, dirs["iterations"])
        
        # Check stopping criteria
        if check_stopping_criteria(
            stats_file, 
            min_acceptance_rate=args.min_acceptance_rate,
            min_coverage_gain=args.min_coverage_gain
        ):
            print(f"Stopping criteria met after {iteration} iterations")
            break
    
    # Save final dataset
    save_final_dataset(all_qa_data, dirs["final_dataset"])


if __name__ == "__main__":
    main()
