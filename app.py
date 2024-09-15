from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)