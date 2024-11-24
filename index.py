import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Proceed with splitting the data
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

# Evaluate
predictions = clf.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
