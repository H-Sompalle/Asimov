import pandas as pd
import spacy
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import spacy
from collections import Counter
import numpy as np
import math
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Configure logging to print to the console (terminal) and log to a file
logging.basicConfig(
    level=logging.DEBUG,  # Log level to capture DEBUG and above (INFO, WARNING, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(),  # Ensures logs are printed to the terminal
        logging.FileHandler("program.log")  # Save logs to a file named 'program.log'
    ]
)

# Example log messages for testing
logging.debug("Debug message - Should show in console and file.")
logging.info("Info message - Should show in console and file.")
logging.warning("Warning message - Should show in console and file.")
logging.error("Error message - Should show in console and file.")
logging.critical("Critical message - Should show in console and file.")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Clean text by removing special characters, excessive whitespace, and quotes.
    """
    # Remove special characters except alphanumeric and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?\\s]", "", str(text))
    # Remove excessive whitespace
    text = re.sub(r"\\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def preprocess_text(text):
    """
    Preprocess text using spaCy: tokenize and remove punctuation.
    """
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return tokens, doc

def compute_sentence_burstiness(doc):
    """
    Compute burstiness, which is the variation in sentence length.
    Burstiness = (std_dev of sentence lengths) / (mean of sentence lengths).
    """
    sentence_lengths = [len(sent) for sent in doc.sents]
    if len(sentence_lengths) < 2:
        return 0  # Avoid division by zero for short texts
    mean_length = np.mean(sentence_lengths)
    std_dev = np.std(sentence_lengths)
    burstiness = std_dev / mean_length
    return burstiness

def compute_kl_divergence(tokens, reference_dist):
    """
    Compute Kullback-Leibler (KL) divergence between the token distribution
    of the text and a reference distribution (e.g., human-written text).
    """
    token_counts = Counter(tokens)
    total_tokens = sum(token_counts.values())
    token_probs = {word: count / total_tokens for word, count in token_counts.items()}

    # Compute KL divergence
    kl_div = 0
    for word, ref_prob in reference_dist.items():
        text_prob = token_probs.get(word, 1e-10)  # Use a small probability for missing words
        kl_div += text_prob * math.log2(text_prob / ref_prob)
    return kl_div

def compute_semantic_coherence(doc):
    """
    Compute semantic coherence: average cosine similarity between adjacent sentences.
    """
    sentence_vectors = [sent.vector for sent in doc.sents if sent.vector_norm > 0]
    if len(sentence_vectors) < 2:
        return 1.0  # Perfect coherence for single-sentence text
    similarities = [
        np.dot(sentence_vectors[i], sentence_vectors[i + 1]) /
        (np.linalg.norm(sentence_vectors[i]) * np.linalg.norm(sentence_vectors[i + 1]))
        for i in range(len(sentence_vectors) - 1)
    ]
    return np.mean(similarities)

def compute_entropy(tokens):
    """
    Compute entropy of the token distribution.
    """
    freq_dist = Counter(tokens)
    total_words = sum(freq_dist.values())
    entropy = -sum((freq / total_words) * math.log2(freq / total_words) for freq in freq_dist.values())
    return entropy

def compute_zipf_slope(tokens):
    """
    Compute the slope of the Zipf distribution.
    """
    freq_dist = Counter(tokens)
    rank_freq = sorted(freq_dist.values(), reverse=True)
    log_ranks = np.log(range(1, len(rank_freq) + 1))
    log_freqs = np.log(rank_freq)
    slope, _ = np.polyfit(log_ranks, log_freqs, 1)  # Linear regression to find slope
    return slope

def detect_ai_text(text, reference_dist):
    """
    Detect whether the given text is AI-generated or human-generated using advanced metrics.
    """
    # Preprocess the text
    tokens, doc = preprocess_text(text)

    # Compute statistical features
    burstiness = compute_sentence_burstiness(doc)
    kl_divergence = compute_kl_divergence(tokens, reference_dist)
    semantic_coherence = compute_semantic_coherence(doc)
    entropy = compute_entropy(tokens)
    zipf_slope = compute_zipf_slope(tokens)

    # Heuristics for classification
    if kl_divergence > 1.5 or burstiness < 0.5 or semantic_coherence < 0.8 or entropy < 4.0 or zipf_slope > -0.5:
        return 1  # AI-generated
    else:
        return 0  # Human-generated

# Generate a reference distribution from a human-written corpus
def build_reference_distribution(corpus):
    """
    Build a reference word distribution from a human-written text corpus.
    """
    tokens = list(chain.from_iterable([preprocess_text(text)[0] for text in corpus]))
    token_counts = Counter(tokens)
    total_tokens = sum(token_counts.values())
    reference_dist = {word: count / total_tokens for word, count in token_counts.items()}
    return reference_dist

if __name__ == "__main__":
    logging.info("Loading dataset...")
    try:
        # Explicitly specify column data types
        df = pd.read_csv("AI_Human.csv", names=["text", "generated"], dtype={"text": str, "label": float})
    except Exception as e:
        logging.error(f"Error reading dataset: {e}")
        raise

    logging.info("Cleaning data...")
    # Handle mixed types and invalid labels
    df["generated"] = pd.to_numeric(df["generated"], errors="coerce")  # Convert invalid labels to NaN
    df = df.dropna(subset=["generated"])  # Drop rows with NaN labels
    df["generated"] = df["generated"].astype(int)  # Ensure labels are integers
    df["text"] = df["text"].fillna("").apply(clean_text)  # Clean and fill missing text

    # Ensure only valid labels (0 and 1) are kept
    df = df[df["generated"].isin([0, 1])]

    logging.info(f"Dataset loaded with {len(df)} samples.")
    
    logging.info("Splitting data into train and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(), df["generated"].tolist(), test_size=0.0000000001, random_state=42
    )

    logging.info("Building reference distribution...")
    human_texts = [text for text, label in zip(train_texts, train_labels) if label == 0]
    reference_dist = build_reference_distribution(human_texts)

    logging.info("Starting predictions...")
    predictions = []
    for text in tqdm(test_texts, desc="Predicting"):
        predictions.append(detect_ai_text(text, reference_dist))

    logging.info("Evaluating predictions...")
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=["Human", "AI-Generated"]))
    logging.info("Program finished successfully!")
