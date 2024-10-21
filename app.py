from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models
vectorizer = joblib.load('tfidf_vectorizer_new.pkl')
model = joblib.load('random_forest_model_mew.pkl')

@app.route('/')
def home():
    return "Flask API for ML Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request (JSON format)
        data = request.json
        text = data.get('text')

        # Input validation
        if not text:
            return jsonify({'error': 'No text provided!'}), 400

        # Transform input using the vectorizer
        transformed_text = vectorizer.transform([text])

        # Predict using the Random Forest model
        prediction = model.predict(transformed_text)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})  # prediction[0] will be 'positive', 'negative', etc.
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':  # Corrected line
    app.run(host='0.0.0.0', port=5000)
