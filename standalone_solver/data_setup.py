#!/usr/bin/env python3
"""
Data preparation script for the standalone solver.
Copies and prepares data for the ML competition task.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

def setup_working_directory(output_dir: str = "output"):
    """
    Set up working directory with data for the competition task.
    
    Args:
        output_dir: Directory to create for working files
        
    Returns:
        Path to the working directory
    """
    # Create output directory
    work_dir = Path(output_dir) / "working"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data subdirectory
    data_dir = work_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Created working directory: {work_dir}")
    return work_dir

def prepare_spooky_author_data(work_dir: Path):
    """
    Prepare the Spooky Author Identification dataset.
    
    Args:
        work_dir: Working directory path
    """
    data_dir = work_dir / "data"
    if not (data_dir.exists() or data_dir.is_symlink()):
        data_dir.mkdir(parents=True, exist_ok=True)  # Ensure data directory exists
    
    # Create description.md (only if data_dir is not a symlink to read-only source)
    if not data_dir.is_symlink():
        description_content = """# Spooky Author Identification
Use text classification to predict author."""
        
        with open(data_dir / "description.md", "w") as f:
            f.write(description_content)
    
    # Never write to symlinked data directories to avoid overwriting real MLEBench data
    if data_dir.is_symlink():
        print(f"Using existing MLEBench data from symlink: {data_dir}")
        return
    
    # Check if real data already exists (large files indicate real MLEBench data)
    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    
    if (train_file.exists() and train_file.stat().st_size > 10000 and 
        test_file.exists() and test_file.stat().st_size > 1000):
        print(f"Using existing large dataset files in {data_dir}")
        return
    
    # Create separate sample data directory to avoid any data corruption
    sample_data_dir = work_dir / "sample_data"
    sample_data_dir.mkdir(exist_ok=True)
    
    # Create sample training data (based on actual Spooky Author competition)
    train_data = [
    {"id": "id26305", "text": "This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.", "author": "EAP"},
    {"id": "id17569", "text": "It never once occurred to me that the fumbling might be a mere mistake.", "author": "HPL"},
    {"id": "id11008", "text": "In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.", "author": "EAP"},
    {"id": "id27763", "text": "How lovely is spring As it bursts into sing Warm sunshine and flowers bring many new hours of joy and delightful pleasant weather.", "author": "MWS"},
    {"id": "id12958", "text": "Finding nothing else, not even gold, the Captain wrote down its name on the brown paper bag he used for these specimens, and put it away again.", "author": "HPL"},
    {"id": "id18847", "text": "Had the state of things been anticipated, and had the lanterns been broken by accident or purposely left behind, the matter would be of little consequence; but as these torches were especially needed to light the party through a long and dark descent into an uncharted abyss.", "author": "HPL"},
    {"id": "id19445", "text": "The night-winds rustled by, and she thought of the sea-smell.", "author": "MWS"},
    {"id": "id13573", "text": "My father was a successful man in many ways, but unfortunately he was not much of a companion for his family.", "author": "EAP"},
    {"id": "id16618", "text": "I found my attention riveted upon the long vertical spires that rose from among those that had been noted down.", "author": "HPL"},
    {"id": "id24059", "text": "The entire apartment was fitted up in a style of the most princely magnificence wherein a bizarre taste had been carefully consulted at every point.", "author": "EAP"},
    ]
    
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(sample_data_dir / "train.csv", index=False)
    
    # Create test data
    test_data = [
    {"id": "id02310", "text": "Seldom have I experienced a more disagreeable sensation than on that occasion."},
    {"id": "id04081", "text": "He had evidently, by uprooting several paving stones, formed something of a circular space in a corner where the main wall met a powerful looking buttress."},
    {"id": "id07230", "text": "There were some things which he could not see at all, and others of which only the outlines wavered unsteadily."},
    {"id": "id09750", "text": "I cannot tell, but I think the following story will interest you: On Thursday, the 8th inst., a sailor named Finn was brought to the Nurses' Home, Bellevue Hospital, with a bullet wound of the head."},
    {"id": "id06570", "text": "These phantoms, for earth holds no beings like them, were dancing hand in hand round and round in an endless circle without the ghost of a sound."},
    ]
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(sample_data_dir / "test.csv", index=False)
    
    # Create a symlink from data directory to sample data if no data directory exists
    if not data_dir.exists():
        data_dir.symlink_to(sample_data_dir.resolve(), target_is_directory=True)
        print(f"Created data symlink: {data_dir} -> {sample_data_dir}")
    
    print(f"Created sample data in {sample_data_dir}")
    print(f"Sample train.csv with {len(train_df)} samples")
    print(f"Sample test.csv with {len(test_df)} samples")
    print("Authors in sample data:", train_df['author'].unique().tolist())

def create_instructions_file(work_dir: Path):
    """
    Create instructions file similar to MLE-bench format.
    
    Args:
        work_dir: Working directory path
    """
    instructions = """COMPETITION TASK
================

You are working on the Spooky Author Identification task.

OBJECTIVE:
- Use text classification to predict which author wrote each text sample
- Authors: EAP (Edgar Allan Poe), HPL (H.P. Lovecraft), MWS (Mary Wollstonecraft Shelley)

DATA:
- Training data: ./data/train.csv (contains id, text, author columns)
- Test data: ./data/test.csv (contains id, text columns)
- Task description: ./data/description.md

REQUIREMENTS:
1. Train a model using the training data
2. Make predictions on the test data
3. Save predictions in submission.csv with columns: id, author
4. Use 5-fold cross-validation to evaluate your approach
5. Print the cross-validation score

EVALUATION:
- Models will be evaluated on prediction accuracy
- Focus on approaches that work well with limited training data

CONSTRAINTS:
- Keep execution time reasonable (< 10 minutes)
- Use standard ML libraries (scikit-learn, pandas, numpy, etc.)
- Avoid overly complex deep learning for this small dataset
"""
    
    with open(work_dir / "instructions.txt", "w") as f:
        f.write(instructions)
    
    print("Created instructions.txt")

def main():
    """Main data setup function."""
    print("Setting up data for standalone solver...")
    
    # Setup working directory
    work_dir = setup_working_directory()
    
    # Prepare competition data
    prepare_spooky_author_data(work_dir)
    
    # Create instructions
    create_instructions_file(work_dir)
    
    print(f"\nData setup complete!")
    print(f"Working directory: {work_dir}")
    print(f"Data files:")
    print(f"  - {work_dir}/data/train.csv")
    print(f"  - {work_dir}/data/test.csv")
    print(f"  - {work_dir}/data/description.md")
    print(f"  - {work_dir}/instructions.txt")

if __name__ == "__main__":
    main()