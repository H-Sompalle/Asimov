import nltk
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
import numpy as np
import string

# Ensure you have necessary NLTK resources
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')


stop_words = set(stopwords.words("english"))

# Helper functions for analysis
def calculate_metrics(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]  # Filters out punctuation
    word_count = len(words)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
    punctuation_count = sum([1 for char in text if char in string.punctuation])
    punctuation_density = punctuation_count / len(text) if len(text) > 0 else 0
    
    pos_tags = pos_tag(words)
    pos_frequencies = FreqDist(tag for _, tag in pos_tags)
    
    # Feature metrics
    features = {
        "avg_sentence_length": avg_sentence_length,
        "punctuation_density": punctuation_density,
        "noun_frequency": pos_frequencies.get("NN", 0) / word_count if word_count > 0 else 0,
        "verb_frequency": pos_frequencies.get("VB", 0) / word_count if word_count > 0 else 0,
        "unique_word_ratio": len(set(words)) / word_count if word_count > 0 else 0,
        "stop_word_ratio": len([w for w in words if w in stop_words]) / word_count if word_count > 0 else 0
    }
    return features

def analyze_text(text):
    metrics = calculate_metrics(text)
    
    # Define thresholds based on common AI patterns for simple scoring
    scores = {
        "avg_sentence_length": 1 if metrics["avg_sentence_length"] > 20 else 0,
        "punctuation_density": 1 if metrics["punctuation_density"] < 0.03 else 0,
        "noun_frequency": 1 if metrics["noun_frequency"] > 0.2 else 0,
        "verb_frequency": 1 if metrics["verb_frequency"] < 0.1 else 0,
        "unique_word_ratio": 1 if metrics["unique_word_ratio"] < 0.5 else 0,
        "stop_word_ratio": 1 if metrics["stop_word_ratio"] > 0.5 else 0,
    }
    
    # Sum scores for analysis
    analysis_score = sum(scores.values())
    is_ai_generated = analysis_score >= 3  # Set threshold based on observations
    
    return metrics, scores, is_ai_generated

# Example of using the function
text = """
The Ford Mustang is an iconic American sports car, first introduced in 1964 by the Ford Motor Company. Known for its distinct, muscular styling and powerful performance, the Mustang quickly became a symbol of freedom and speed. Designed as a "pony car"—a term it inspired—it was aimed at young drivers looking for an affordable, sporty vehicle with personality.

The early Mustang models featured sleek, long hoods and short rear decks, with engine options that ranged from modest inline-6 engines to powerful V8s. Over the years, the Mustang evolved, with each generation seeing innovations in design, technology, and performance. Key models include the Boss, Shelby GT500, and Mach 1, each celebrated for its high-performance capabilities and racing legacy.

Modern Mustangs blend retro design cues with advanced engineering. Current models feature turbocharged EcoBoost engines as well as robust V8s, offering both efficiency and power. Advanced technology like independent rear suspension, driver assistance features, and customizable drive modes makes the Mustang as adept on the highway as on the racetrack.
"""

metrics, scores, is_ai_generated = analyze_text(text)

print("Metrics:", metrics)
print("Scores:", scores)
print("Is AI-Generated:", is_ai_generated)
