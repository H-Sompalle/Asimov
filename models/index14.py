import numpy as np
import pandas as pd
import logging
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to compute Zipf's slope
def compute_zipf_slope(tokens):
    freq_dist = Counter(tokens)
    rank_freq = sorted(freq_dist.values(), reverse=True)
    if len(rank_freq) < 2:  # Not enough unique frequencies
        logging.warning("Not enough unique word frequencies to compute Zipf slope.")
        return -1.0  # Default value for slope
    log_ranks = np.log(range(1, len(rank_freq) + 1))
    log_freqs = np.log(rank_freq)
    slope, _ = np.polyfit(log_ranks, log_freqs, 1)  # Linear regression to find slope
    return slope

# Function to compute KL divergence
def compute_kl_divergence(text, reference_dist):
    tokens = text.split()
    freq_dist = Counter(tokens)
    total_tokens = sum(freq_dist.values())
    kl_div = 0
    for word, freq in freq_dist.items():
        prob = freq / total_tokens
        if word in reference_dist:
            kl_div += prob * np.log(prob / reference_dist[word])
    return kl_div

# Function to compute burstiness
def compute_burstiness(text):
    tokens = text.split()
    token_lengths = [len(token) for token in tokens]
    mean_length = np.mean(token_lengths)
    std_length = np.std(token_lengths)
    if mean_length == 0:
        return 0
    return std_length / mean_length

# Function to compute entropy
def compute_entropy(text):
    tokens = text.split()
    freq_dist = Counter(tokens)
    total_tokens = sum(freq_dist.values())
    entropy = -sum((freq / total_tokens) * np.log2(freq / total_tokens) for freq in freq_dist.values())
    return entropy

# Main detection function
def detect_ai_text(text, reference_dist):
    tokens = text.split()
    kl_divergence = compute_kl_divergence(text, reference_dist)
    burstiness = compute_burstiness(text)
    entropy = compute_entropy(text)
    zipf_slope = compute_zipf_slope(tokens)

    logging.debug(f"KL: {kl_divergence}, Burstiness: {burstiness}, Entropy: {entropy}, Zipf: {zipf_slope}")

    # Example classification heuristic
    if kl_divergence > 1.2 or burstiness < 0.6 or entropy < 4.2 or zipf_slope > -0.7:
        return 1  # AI-generated
    else:
        return 0  # Human-generated

# Function to balance the dataset
def balance_dataset(df):
    human_data = df[df["generated"] == 0]
    ai_data = df[df["generated"] == 1]

    if len(human_data) < len(ai_data):
        human_data = human_data.sample(len(ai_data), replace=True, random_state=42)
    elif len(ai_data) < len(human_data):
        ai_data = ai_data.sample(len(human_data), replace=True, random_state=42)

    return pd.concat([human_data, ai_data]).sample(frac=1, random_state=42)

# Main program
def main():
    logging.info("Loading dataset...")
    # Replace this with actual dataset loading
    dataset = pd.read_csv("AI_Human.csv")  # Assuming data.csv with columns: text, generated (0 or 1)

    logging.info("Balancing dataset...")
    dataset = balance_dataset(dataset)

    logging.info("Splitting dataset into training and test sets...")
    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create reference distribution for KL divergence
    reference_texts = " ".join(train_df[train_df["generated"] == 0]["text"])
    reference_tokens = reference_texts.split()
    total_reference_tokens = len(reference_tokens)
    reference_dist = {word: freq / total_reference_tokens for word, freq in Counter(reference_tokens).items()}

    # Predictions
    predictions = []
    labels = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        text = row["text"]
        label = row["generated"]
        prediction = detect_ai_text(text, reference_dist)
        predictions.append(prediction)
        labels.append(label)

    logging.info("Evaluating model...")
    print(classification_report(labels, predictions, zero_division=0))

if __name__ == "__main__":
    main()
