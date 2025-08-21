#!/usr/bin/env python3
"""
Simple test with a working solution to verify the system works.
"""

from code_evaluator import CodeEvaluator
from pathlib import Path

# Create test solution that should work with small dataset
test_solution = """
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
import numpy as np

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Prepare data
X = train_df['text']
y = train_df['author']
X_test = test_df['text']

# Create simple pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Use Leave-One-Out CV for small dataset
loo = LeaveOneOut()
cv_scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')

print(f"Leave-One-Out CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")

# Train and predict
pipeline.fit(X, y)
test_predictions = pipeline.predict(X_test)

# Create submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'author': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print("Submission saved!")
print("Final CV Score:", cv_scores.mean())
"""

if __name__ == "__main__":
    # Test with our existing working directory
    work_dir = Path("output/working")
    
    if not work_dir.exists():
        print("Setting up data first...")
        from data_setup import setup_working_directory, prepare_spooky_author_data, create_instructions_file
        setup_working_directory(work_dir.parent)
        prepare_spooky_author_data(work_dir)
        create_instructions_file(work_dir)
    
    evaluator = CodeEvaluator(work_dir)
    result = evaluator.execute_code(test_solution, verbose=True)
    
    print("\nExecution Result:")
    print(f"Success: {result['success']}")
    print(f"CV Score: {result['cv_score']}")
    print(f"Submission exists: {result['submission_exists']}")
    
    if not result['success']:
        print(f"Error: {result['stderr']}")
    else:
        print("✅ Simple test passed!")