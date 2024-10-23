from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model, scaler, and feature names
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    logging.debug("Model, scaler, and feature names loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model, scaler, or feature names: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.debug(f"Received data: {data}")

        # Validate that required fields are present and valid
        required_fields = ['OverallQual', 'GrLivArea', 'GarageCars']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                logging.error(f"Missing or invalid field: {field}")
                return jsonify({'error': f'Missing or invalid field: {field}'}), 400

        # Convert data into DataFrame
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        logging.debug(f"Input data after reindexing: {input_data}")

        # Scale input data
        input_scaled = scaler.transform(input_data)
        logging.debug(f"Scaled input data: {input_scaled}")

        # Make prediction
        prediction = model.predict(input_scaled)
        logging.debug(f"Prediction: {prediction}")
        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
