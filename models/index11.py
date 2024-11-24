import pandas as pd
import spacy
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import numpy as np
from collections import Counter
from itertools import chain
import math

# Configure logging to print to the console (terminal) and log to a file
logging.basicConfig(
    level=logging.DEBUG,  # Log level to capture DEBUG and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(),  # Print logs to the terminal
        logging.FileHandler("program.log")  # Save logs to a file named 'program.log'
    ]
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Clean text by removing special characters, excessive whitespace, and quotes.
    """
    text = re.sub(r"[^a-zA-Z0-9.,!?\\s]", "", str(text))  # Remove special characters
    text = re.sub(r"\\s+", " ", text)  # Remove excessive whitespace
    text = text.strip()  # Strip leading/trailing whitespace
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

def compute_zipf_slope(tokens, min_freq=2):
    """
    Compute the slope of the Zipf distribution, ignoring very rare words.
    """
    freq_dist = Counter(tokens)
    rank_freq = [freq for freq in freq_dist.values() if freq >= min_freq]

    if len(rank_freq) < 2:
        logging.warning("Not enough unique word frequencies to compute Zipf slope.")
        return 0

    rank_freq.sort(reverse=True)
    log_ranks = np.log(range(1, len(rank_freq) + 1))
    log_freqs = np.log(rank_freq)

    try:
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)
    except np.linalg.LinAlgError:
        logging.error("Linear regression failed in Zipf slope calculation.")
        return 0

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
        df = pd.read_csv("AI_Human.csv", header = 0, names=["text", "generated"], dtype={"text": str, "generated": int})
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    df = df.dropna()
    corpus = df[df["generated"] == 0]["text"].tolist()  # Human-written texts
    reference_dist = build_reference_distribution(corpus)

    logging.info("Splitting dataset into training and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["generated"], test_size=0.2, random_state=42
    )

    predictions = []
    for text in tqdm(test_texts, desc="Predicting"):
        predictions.append(detect_ai_text(text, reference_dist))

    logging.info("Evaluating model...")
    report = classification_report(test_labels, predictions)
    print(report)
