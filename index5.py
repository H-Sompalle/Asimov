import pickle

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
new_text = "The Ford Mustang is a sports car."
result = predict_text(new_text)
print(f"The text is classified as: {result}")
