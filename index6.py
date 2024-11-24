import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import re

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')

# Download necessary NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define your AI detection function
def analyze_text(text):
    # Preprocess text
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    # Feature engineering
    features = {}
    features['average_word_length'] = sum(len(word) for word in tokens) / len(tokens) if tokens else 0
    features['punctuation_count'] = sum(1 for char in text if char in string.punctuation)
    features['capitalized_count'] = sum(1 for word in tokens if word[0].isupper())
    features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
    
    # Rules-based AI detection logic (placeholder for demo purposes)
    is_ai_generated = (
        features['average_word_length'] < 4.5 or 
        features['punctuation_count'] > 5 or 
        features['capitalized_count'] / (len(tokens) + 1) > 0.2
    )
    
    return features, is_ai_generated

# Load a subset of your labeled dataset for testing
# Replace 'path_to_your_labeled_dataset.csv' with the actual path to your dataset
data = pd.read_csv('AI_Human.csv').sample(100)  # Adjust the sample size as needed

# Initialize lists to store predictions and actual labels
predictions = []
actual_labels = data['generated'].tolist()  # Assuming 'label' column contains 1 for AI-generated, 0 for human

# Run the model and store results
for text in data['text']:  # Assuming 'text' column contains the text samples
    _, is_ai_generated = analyze_text(text)
    predictions.append(1 if is_ai_generated else 0)  # Convert prediction to binary

# Calculate evaluation metrics
accuracy = accuracy_score(actual_labels, predictions)
precision = precision_score(actual_labels, predictions)
recall = recall_score(actual_labels, predictions)
f1 = f1_score(actual_labels, predictions)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
