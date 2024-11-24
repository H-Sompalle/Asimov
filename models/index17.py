import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from scipy.stats import entropy as scipy_entropy
import re
import string
import warnings
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')

# Suppress warnings
warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing: Tokenization and Stopword Removal
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

# Statistical Features
def compute_kl_divergence(reference_dist, target_dist):
    """Compute KL Divergence between two distributions."""
    target_dist = np.array(target_dist)
    reference_dist = np.array(reference_dist)
    target_dist = np.where(target_dist == 0, 1e-10, target_dist)  # Avoid log(0)
    return scipy_entropy(target_dist, reference_dist)

def compute_burstiness(tokens):
    """Measure burstiness: variability of token lengths."""
    token_lengths = [len(token) for token in tokens]
    if len(token_lengths) <= 1:
        return 0  # Avoid division by zero
    std_dev = np.std(token_lengths)
    mean = np.mean(token_lengths)
    if mean == 0:
        return 0
    return (std_dev - mean) / (std_dev + mean)

def compute_entropy(tokens):
    """Measure lexical diversity using entropy."""
    token_counts = pd.Series(tokens).value_counts()
    probabilities = token_counts / len(tokens)
    return scipy_entropy(probabilities)

def compute_lexical_diversity(tokens):
    """Measure lexical diversity (unique words / total words)."""
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)

# Detection Logic Based on Statistical Features
def detect_ai_text(text, reference_dist):
    tokens = preprocess_text(text)
    if not tokens:  # Handle empty text
        return 1  # Default to AI-generated
    
    # Compute Statistical Features
    lexical_diversity = compute_lexical_diversity(tokens)
    burstiness = compute_burstiness(tokens)
    entropy = compute_entropy(tokens)
    target_dist = pd.Series(tokens).value_counts(normalize=True)
    target_dist = target_dist.reindex(reference_dist.index, fill_value=0)
    kl_divergence = compute_kl_divergence(reference_dist.values, target_dist.values)

    # Decision logic based on thresholds
    is_ai_generated = (
        kl_divergence > 1.5 or 
        burstiness < 0.5 or 
        entropy < 3.5 or 
        lexical_diversity < 0.2
    )
    return is_ai_generated

# Dataset Preparation
def prepare_reference_distribution(texts):
    """Prepare reference distribution from human-written texts."""
    all_tokens = [token for text in texts for token in preprocess_text(text)]
    token_counts = pd.Series(all_tokens).value_counts(normalize=True)
    return token_counts

# Main Program
if __name__ == "__main__":
    # Load Dataset
    data_path = "AI_Human.csv"  # Replace with your dataset path
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Dataset not found at {data_path}. Please provide the correct path.")
        exit()

    # Preprocessing
    data = data.dropna(subset=["text", "generated"])  # Ensure no missing values
    texts = data["text"]
    labels = data["generated"]

    # Split data into training (reference generation) and test sets
    train_texts, test_texts = texts[:len(texts)//2], texts[len(texts)//2:]
    train_labels, test_labels = labels[:len(labels)//2], labels[len(labels)//2:]

    # Generate Reference Distribution
    reference_dist = prepare_reference_distribution(train_texts)

    # Predictions
    predictions = []
    for text in test_texts:
        is_ai = detect_ai_text(text, reference_dist)
        predictions.append(1 if is_ai else 0)

    # Evaluation Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    # Print Metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
