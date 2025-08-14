import numpy as np
import argparse
import os


def check_embeddings(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    try:
        arr = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    print(f"Checking: {file_path}")
    print(f"  Shape: {arr.shape}")
    print(f"  Any NaN: {np.isnan(arr).any()}")
    print(f"  Any Inf: {np.isinf(arr).any()}")
    zero_rows = np.where(np.linalg.norm(arr, axis=1) == 0)[0]
    print(f"  Number of all-zero rows: {len(zero_rows)}")
    if len(zero_rows) > 0:
        print(f"    Example all-zero row indices: {zero_rows[:5]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Check for NaNs, Infs, and all-zero vectors in embedding .npy files.")
    parser.add_argument("embedding_files", nargs='+', help="Path(s) to .npy embedding files")
    args = parser.parse_args()
    for file_path in args.embedding_files:
        check_embeddings(file_path)

if __name__ == "__main__":
    main()
