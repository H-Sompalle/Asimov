import spacy
from collections import Counter
import numpy as np
import math
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

    # Display features (optional for debugging)
    print(f"Features:\n  Burstiness: {burstiness:.2f}\n  KL Divergence: {kl_divergence:.2f}")
    print(f"  Semantic Coherence: {semantic_coherence:.2f}\n  Entropy: {entropy:.2f}\n  Zipf Slope: {zipf_slope:.2f}")

    # Heuristics for classification
    if kl_divergence > 1.5 or burstiness < 0.5 or semantic_coherence < 0.8 or entropy < 4.0 or zipf_slope > -0.5:
        return "AI-generated"
    else:
        return "Human-generated"

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

# Test the program
if __name__ == "__main__":
    # Example human-written corpus for building reference distribution
    human_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "This is an example of human-written text. It has varying lengths and randomness.",
        "Human-written texts often follow natural language patterns."
    ]
    reference_dist = build_reference_distribution(human_corpus)

    test_texts = [
        """Cars. Cars have been around since they became famous in the 1900s, when Henry Ford created and built the first ModelT. Cars have played a major role in our every day lives since then. But now, people are starting to question if limiting car usage would be a good thing. To me, limiting the use of cars might be a good thing to do.

In like matter of this, article, ""In German Suburb, Life Goes On Without Cars,"" by Elizabeth Rosenthal states, how automobiles are the linchpin of suburbs, where middle class families from either Shanghai or Chicago tend to make their homes. Experts say how this is a huge impediment to current efforts to reduce greenhouse gas emissions from tailpipe. Passenger cars are responsible for 12 percent of greenhouse gas emissions in Europe...and up to 50 percent in some carintensive areas in the United States. Cars are the main reason for the greenhouse gas emissions because of a lot of people driving them around all the time getting where they need to go. Article, ""Paris bans driving due to smog,"" by Robert Duffer says, how Paris, after days of nearrecord pollution, enforced a partial driving ban to clear the air of the global city. It also says, how on Monday, motorist with evennumbered license plates were ordered to leave their cars at home or be fined a 22euro fine 31. The same order would be applied to oddnumbered plates the following day. Cars are the reason for polluting entire cities like Paris. This shows how bad cars can be because, of all the pollution that they can cause to an entire city.

Likewise, in the article, ""Carfree day is spinning into 
a big hit in Bogota,"" by Andrew Selsky says, how programs that's set 
to spread to other countries, millions of Columbians hiked, biked, 
skated, or took the bus to work during a carfree day, 
leaving streets of this capital city eerily devoid of 
traffic jams. It was the third straight year cars have been 
banned with only buses and taxis permitted for the Day Without 
Cars in the capital city of 7 million. People like the idea of 
having carfree days because, it allows them to lesson the pollution 
that cars put out of their exhaust from people driving all the time. 
The article also tells how parks and sports centers have bustled throughout the city uneven, pitted sidewalks have been replaced by broad, smooth sidewalks rushhour restrictions have dramatically cut traffic and new restaurants and upscale shopping districts have cropped up. Having no cars has been good for the country of Columbia because, it has aloud them to repair things that have needed repairs for a long time, traffic jams have gone down, and restaurants and shopping districts have popped up, all due to the fact of having less cars around.

In conclusion, the use of less cars and having carfree days, have had a big impact on the environment of cities because, it is cutting down the air pollution that the cars have majorly polluted, it has aloud countries like Columbia to repair sidewalks, and cut down traffic jams. Limiting the use of cars would be a good thing for America. So we should limit the use of cars by maybe riding a bike, or maybe walking somewhere that isn't that far from you and doesn't need the use of a car to get you there. To me, limiting the use of cars might be a good thing to do.",0.0
"Transportation is a large necessity in most countries worldwide. With no doubt, cars, buses, and other means of transportation make going from place to place easier and faster. However there's always a negative pollution. Although mobile transportation are a huge part of daily lives, we are endangering the Earth with harmful greenhouse gases, which could be suppressed.

A small suburb community in Germany called Vauban, has started a ""carfree"" lifestyle.""",
        "The sun sets in the west. Birds fly over the horizon. Nature feels alive."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}:")
        print(text)
        result = detect_ai_text(text, reference_dist)
        print(f"Prediction: {result}")
