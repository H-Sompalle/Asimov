import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load CSV dataset
df = pd.read_csv('AI_Human.csv', header=None, dtype={1: str}, low_memory=False)
df.columns = ['text', 'label']

# Drop any rows with missing values
df = df.dropna()

# Strip any leading/trailing whitespace from labels and convert them to a consistent format
df['label'] = df['label'].str.strip()

# Filter out any rows with rare or inconsistent labels
df = df[df['label'].isin(['0.0', '1.0'])]

# Check class distribution again
print("Class distribution after cleaning:")
print(df['label'].value_counts())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Evaluate the model
predictions = clf.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# Save the trained model and vectorizer to disk for later use
with open("text_classifier_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")

# Function to make predictions on new inputs
def predict_text(text):
    # Load the model and vectorizer
    with open("text_classifier_model.pkl", "rb") as model_file:
        clf = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Vectorize the input text
    text_vec = vectorizer.transform([text])

    # Make a prediction
    prediction = clf.predict(text_vec)
    
    # Map the prediction to human-readable output if needed
    label_map = {"0.0": "Human-generated", "1.0": "AI-generated"}
    return label_map.get(prediction[0], "Unknown")

# Example usage for prediction on a new text
new_text = "This is an example sentence to classify."
result = predict_text(new_text)
print(f"The text is classified as: {result}")
