from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import numpy as np
import spacy
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app)

# Load Machine Learning Model & Tokenizer
try:
    model = tf.keras.models.load_model('sentiment_model.h5')  # Load TensorFlow model
    tf.config.run_functions_eagerly(True)  # Ensure eager execution

    with open('tokenizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)  # Load tokenizer
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Load SpaCy NLP Model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading SpaCy model: {e}")
    exit(1)

# Sentiment classification labels
def classify_sentiment(score):
    if score >= 0.8:
        return "Great"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Moderate"
    elif score >= 0.2:
        return "Bad"
    else:
        return "Terrible"

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()  # Clean text
    doc = nlp(text)  # Process text with SpaCy
    return ' '.join(token.lemma_ for token in doc)  # Lemmatize text

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        review_text = data.get("review", "").strip()

        if not review_text:
            return jsonify({"error": "No review text provided"}), 400

        # Process review text
        processed_review = preprocess_text(review_text)

        # Vectorize text correctly
        review_vector = vectorizer.transform([processed_review]).toarray().astype(np.float32)

        # Debugging: Check review vector shape
        print(f"Review Vector Shape: {review_vector.shape}, Type: {type(review_vector)}")

        # Ensure the model is loaded correctly
        if model is None:
            print("Error: Model is not loaded!")
            return jsonify({"error": "Model not loaded"}), 500

        # Predict sentiment
        prediction = model.predict(review_vector)

        # Debugging: Check model prediction output
        print(f"Model Prediction Output: {prediction}, Type: {type(prediction)}")

        # Handle empty or invalid prediction
        if prediction is None or len(prediction) == 0:
            return jsonify({"error": "Model returned no prediction"}), 500

        sentiment_score = float(prediction[0][0])  # Ensure valid float conversion
        sentiment_label = classify_sentiment(sentiment_score)

        return jsonify({"review": review_text, "sentiment": sentiment_label, "score": sentiment_score})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

