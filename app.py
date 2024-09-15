from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert data into DataFrame
    input_data = pd.DataFrame([data])
    
    # Align input data with the training data
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Return the prediction as JSON
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)