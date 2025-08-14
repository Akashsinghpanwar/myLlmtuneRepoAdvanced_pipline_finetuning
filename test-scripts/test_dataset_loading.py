#!/usr/bin/env python
"""
Test script to verify dataset loading from Hugging Face Hub.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import the load_dataset_from_source function
from datasets import load_dataset, load_from_disk

# Import the load_dataset_from_source function directly
sys.path.append(str(Path(__file__).parent.parent / "visualisation"))
from embed_utils import load_dataset_from_source

def test_load_dataset(dataset_id):
    """Test loading a dataset directly using the datasets library."""
    print(f"\n=== Testing direct loading of dataset: {dataset_id} ===")
    
    try:
        # Try loading with split specified and force_download=True
        print("Attempting to load with split='train' and force_download=True...")
        dataset = load_dataset(dataset_id, split="train", download_mode="force_download")
        print(f"✅ Successfully loaded dataset with split='train' and force_download=True")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset features: {dataset.features}")
        print(f"First example: {dataset[0]}")
        return dataset
    except Exception as e:
        print(f"❌ Error loading with split='train' and force_download=True: {e}")
        
        try:
            # Try loading with split specified but without force_download
            print("\nAttempting to load with split='train'...")
            dataset = load_dataset(dataset_id, split="train")
            print(f"✅ Successfully loaded dataset with split='train'")
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset shape: {dataset.shape}")
            print(f"Dataset features: {dataset.features}")
            print(f"First example: {dataset[0]}")
            return dataset
        except Exception as e:
            print(f"❌ Error loading with split='train': {e}")
            
            try:
                # Try loading without split
                print("\nAttempting to load without specifying split...")
                dataset = load_dataset(dataset_id)
                print(f"✅ Successfully loaded dataset without split")
                print(f"Dataset type: {type(dataset)}")
                print(f"Dataset keys: {list(dataset.keys()) if hasattr(dataset, 'keys') else 'N/A'}")
                
                # If it's a DatasetDict, try to access the train split
                if hasattr(dataset, 'keys') and 'train' in dataset:
                    train_dataset = dataset['train']
                    print(f"Train split shape: {train_dataset.shape}")
                    print(f"First example from train: {train_dataset[0]}")
                return dataset
            except Exception as e2:
                print(f"❌ Error loading without split: {e2}")
                return None

def test_load_dataset_from_source(dataset_id):
    """Test loading a dataset using the load_dataset_from_source function."""
    print(f"\n=== Testing load_dataset_from_source: {dataset_id} ===")
    
    try:
        dataset = load_dataset_from_source(dataset_id)
        print(f"✅ Successfully loaded dataset using load_dataset_from_source")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset features: {dataset.features}")
        print(f"First example: {dataset[0]}")
        return dataset
    except Exception as e:
        print(f"❌ Error loading with load_dataset_from_source: {e}")
        return None

def main():
    """Main function to run tests."""
    # Test datasets
    datasets_to_test = [
        "Trelis/touch-rugby-o4-mini-5k_chunks-2_chunks",
        # Add more datasets to test if needed
    ]
    
    for dataset_id in datasets_to_test:
        # Test direct loading
        direct_dataset = test_load_dataset(dataset_id)
        
        # Test using load_dataset_from_source
        source_dataset = test_load_dataset_from_source(dataset_id)
        
        # Compare results
        if direct_dataset is not None and source_dataset is not None:
            print("\n=== Comparison ===")
            direct_shape = direct_dataset.shape if hasattr(direct_dataset, 'shape') else "N/A"
            source_shape = source_dataset.shape
            
            print(f"Direct loading shape: {direct_shape}")
            print(f"load_dataset_from_source shape: {source_shape}")
            
            if hasattr(direct_dataset, 'shape') and direct_shape == source_shape:
                print("✅ Shapes match!")
            else:
                print("❌ Shapes don't match!")

if __name__ == "__main__":
    main()
