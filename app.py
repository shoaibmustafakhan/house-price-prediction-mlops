from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate that required fields are present and valid
        required_fields = ['OverallQual', 'GrLivArea', 'GarageCars']
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Missing or invalid field: {field}'}), 400

        # Convert data into DataFrame
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)