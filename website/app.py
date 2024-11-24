from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Paths to model and vectorizer
MODEL_PATH = "text_classifier_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"



# Load model and vectorizer at startup
try:
    with open(MODEL_PATH, "rb") as model_file:
        clf = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    clf, vectorizer = None, None

# Function to make predictions
def predict_text(text):
    try:
        # Vectorize the input text
        text_vec = vectorizer.transform([text])

        # Make a prediction
        prediction = clf.predict(text_vec)

        # Map the prediction to human-readable output
        label_map = {0: "Human-generated", 1: "AI-generated"}
        label = label_map.get(prediction[0], "Unknown")

        return {"prediction": label, "raw_output": prediction[0]}
    except Exception as e:
        print(f"Error in predict_text: {e}")
        return None


# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not clf or not vectorizer:
        return jsonify({"error": "Model or vectorizer not loaded successfully. Check the server logs."}), 500

    try:
        data = request.json
        text_input = data.get("text", "")
        if not text_input:
            return jsonify({"error": "No text provided"}), 400

                # Make a prediction
        result = predict_text(text_input)

        if result:  # Ensure result is not None
            # Map the raw output to a human-readable prediction
            if result["raw_output"] == "1.0":
                actual = "AI Generated"
            elif result["raw_output"] == "0.0":
                actual = "Human Generated"
            else:
                actual = "Unknown"  # Handle unexpected outputs

            # Return the prediction response
            return jsonify({
                "text": text_input,
                "prediction": actual,
                "raw_output": result["raw_output"]
            })
        else:
            # Handle the case where result is None or prediction fails
            return jsonify({"error": "Error during prediction"}), 500

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# Feedback endpoint (optional)
@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    try:
        data = request.json
        feedback_text = data.get("feedback", "")
        if not feedback_text:
            return jsonify({"error": "No feedback provided"}), 400

        print("User feedback received:", feedback_text)
        return jsonify({"message": "Thank you for your feedback!"})
    except Exception as e:
        print(f"Error in /feedback endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# Run app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))  # Use PORT environment variable or default to 8080
    app.run(host='0.0.0.0', port=port)
