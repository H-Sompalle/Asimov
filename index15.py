import numpy as np
import pandas as pd
import re
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Functions for statistical calculations

def compute_kl_divergence(reference_dist, target_dist):
    reference = np.array(reference_dist)
    target = np.array(target_dist)
    target = np.where(target == 0, 1e-10, target)  # Avoid log(0)
    return scipy_entropy(target, reference)

def compute_burstiness(tokens):
    token_lengths = [len(token) for token in tokens]
    if len(token_lengths) <= 1:
        return 0  # Avoid division by zero
    std_dev = np.std(token_lengths)
    mean = np.mean(token_lengths)
    if mean == 0:
        return 0
    return (std_dev - mean) / (std_dev + mean)

def compute_entropy(tokens):
    token_counts = pd.Series(tokens).value_counts()
    probabilities = token_counts / len(tokens)
    return scipy_entropy(probabilities)

def compute_zipf_slope(tokens):
    token_counts = pd.Series(tokens).value_counts().sort_values(ascending=False)
    if len(token_counts) <= 1:
        logging.warning("Not enough unique word frequencies to compute Zipf slope.")
        return -1.0  # Default Zipf slope
    ranks = np.arange(1, len(token_counts) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(token_counts.values)
    slope, _ = np.polyfit(log_ranks, log_freqs, 1)
    return slope

# Heuristic-based AI text detection

def detect_ai_text(text, reference_dist):
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:  # Handle empty text
        return 1  # Default to AI-generated
    
    kl_divergence = compute_kl_divergence(reference_dist, np.random.dirichlet(np.ones(len(reference_dist))))
    burstiness = compute_burstiness(tokens)
    entropy = compute_entropy(tokens)
    zipf_slope = compute_zipf_slope(tokens)

    if kl_divergence > 1.0 or burstiness < 0.5 or entropy < 3.5 or zipf_slope > -0.8:
        return 1  # AI-generated
    else:
        return 0  # Human-generated

# Dataset loading and preprocessing

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded with {len(data)} rows.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def balance_dataset(data):
    min_count = data["generated"].value_counts().min()
    balanced_data = data.groupby("generated").apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    logging.info(f"Balanced dataset to {len(balanced_data)} rows.")
    return balanced_data

# Main program

if __name__ == "__main__":
    # Load and preprocess dataset
    file_path = "AI_Human.csv"
    data = load_dataset(file_path)
    if data is None:
        exit()

    data = balance_dataset(data)
    texts, labels = data["text"], data["generated"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    logging.info("Splitting dataset into training and test sets...")

    # Generate a reference distribution from training data
    all_tokens = [token for text in X_train for token in re.findall(r'\b\w+\b', text.lower())]
    token_counts = pd.Series(all_tokens).value_counts()
    reference_dist = token_counts / token_counts.sum()

    # Make predictions
    predictions = []
    for text in tqdm(X_test, desc="Predicting"):
        predictions.append(detect_ai_text(text, reference_dist))

    # Evaluate model
    logging.info("Evaluating model...")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
