#!/usr/bin/env python
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.create_dataset import ensure_dependencies

# Ensure required dependencies
ensure_dependencies()

# Import after ensuring dependencies
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def create_touch_rugby_dataset():
    """Create a small touch rugby dataset and push it to Hugging Face Hub."""
    # Define the touch rugby QA pairs
    qa_data = [
        {
            "question": "What is a conversion worth in touch rugby according to FIT rules?",
            "evaluation_criteria": "There are no conversions in touch rugby, correct answers should recognise this"
        },
        {
            "question": "Is a forward pass in touch rugby a penalty or a scrum under FIT rules?",
            "evaluation_criteria": "A penalty"
        },
        {
            "question": "How many players on the field for each team in touch rugby at the start of a drop off?",
            "evaluation_criteria": "4 players"
        },
        {
            "question": "How far back must defending players retreat after making a touch vs after conceding a penalty - in touch rugby according to FIT rules?",
            "evaluation_criteria": "10 metres after a penalty and 7 after a touch (or to their touch line, if that is less)"
        },
        {
            "question": "How does the game begin according to FIT rules for touch rugby?",
            "evaluation_criteria": "With a tap off at the center of the field. Fine also to say it starts with a coinflip to decide direction of play"
        },
        {
            "question": "In touch rugby according to FIT rules, who is permitted to seek clarification on a referee's decisions?",
            "evaluation_criteria": "Only team captains"
        },
        {
            "question": "In touch rugby according to FIT rules, what happens if an attacking team player passes the ball and it hits a defending player?",
            "evaluation_criteria": "A change of possession occurs. If the defending player deliberately made contact with the ball, then the attacking team retains possession and the touch count restarts as zero touch. If the attacking player deliberately hit a defending player, aiming for a rebound, then a change of possession occurs."
        },
        {
            "question": "In touch rugby according to FIT rules, what happens if the ball is passed forward by the team in posession and then intercepted and dropped by the opposing team?",
            "evaluation_criteria": "A penalty is awarded to the opposing team"
        },
        {
            "question": "In touch rugby according to FIT rules, how many touches is a team entitled to prior to a change in posession?",
            "evaluation_criteria": "Six touches"
        },
        {
            "question": "In touch rugby according to FIT rules, what if the ball - under the control of the Half - touches the ground in the in-goal area?",
            "evaluation_criteria": "A change of possession occurs and play restarts with a rollball at the nearest point on the 7 metre line"
        }
    ]
    
    # Create dataset
    dataset = Dataset.from_list(qa_data)
    
    # Create dataset dictionary with all data in the training split
    dataset_dict = DatasetDict({
        "train": dataset
    })
    
    # Save dataset locally
    dataset_dir = Path(__file__).parent.parent / "data" / "touch_rugby_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dict.save_to_disk(str(dataset_dir))
    print(f"Dataset saved locally to {dataset_dir}")
    
    # Check if already logged in
    api = HfApi()
    try:
        # Try to get user info to check if already logged in
        user_info = api.whoami()
        print(f"Already logged in as {user_info['name']}")
    except Exception:
        print("Error checking Hugging Face login status. Make sure you're logged in.")
        return
    
    # Push dataset to Hugging Face Hub
    repo_name = "Trelis/touch-rugby-qa-manual"
    try:
        dataset_dict.push_to_hub(
            repo_name,
            private=False
        )
        print(f"Dataset pushed to Hugging Face Hub: {repo_name}")
        print(f"Access it at: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")

if __name__ == "__main__":
    create_touch_rugby_dataset()
