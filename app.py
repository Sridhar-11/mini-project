from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and imputer
MODEL_FILE_PATH = 'liver_disease_gb_model.pkl'
IMPUTER_FILE_PATH = 'imputer.pkl'

try:
    model_instance = joblib.load(MODEL_FILE_PATH)
    imputer_instance = joblib.load(IMPUTER_FILE_PATH)
    print("Model and imputer loaded successfully!")
except Exception as e:
    print(f"Error loading model or imputer: {e}")
    model_instance = None
    imputer_instance = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model_instance is None or imputer_instance is None:
        return jsonify({'result': "Model or imputer not loaded. Please check server logs."}), 500

    try:
        # Get input data from the form
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', type=str).lower()
        total_bilirubin = request.form.get('total_bilirubin', type=float)
        direct_bilirubin = request.form.get('direct_bilirubin', type=float)
        alkaline_phosphotase = request.form.get('alkaline_phosphotase', type=float)
        alamine_aminotransferase = request.form.get('alamine_aminotransferase', type=float)
        aspartate_aminotransferase = request.form.get('aspartate_aminotransferase', type=float)
        total_proteins = request.form.get('total_proteins', type=float)
        albumin = request.form.get('albumin', type=float)
        albumin_and_globulin_ratio = request.form.get('albumin_and_globulin_ratio', type=float)

        # Validate input values
        if not all([
            age > 0, total_bilirubin > 0, direct_bilirubin > 0,
            alkaline_phosphotase > 0, alamine_aminotransferase > 0,
            aspartate_aminotransferase > 0, total_proteins > 0,
            albumin > 0, albumin_and_globulin_ratio > 0
        ]):
            return jsonify({'result': "Invalid input: All values must be positive and non-zero."}), 400

        # Gender validation and encoding
        if gender not in ['male', 'female']:
            return jsonify({'result': "Invalid input: Gender must be 'male' or 'female'."}), 400
        gender_encoded = 1 if gender == 'male' else 0

        # Prepare input features
        input_features = [
            age, gender_encoded, total_bilirubin, direct_bilirubin,
            alkaline_phosphotase, alamine_aminotransferase,
            aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio
        ]

        # Log input features
        print("Input features (raw):", input_features)

        # Handle missing values using the imputer
        input_features_imputed = imputer_instance.transform([input_features])

        # Log imputed features
        print("Input features (imputed):", input_features_imputed)

        # Predict using the trained model
        prediction = model_instance.predict(input_features_imputed)

        # Log prediction
        print("Model prediction:", prediction)

        # Map the prediction result
        result = (
            "The person is likely to have liver disease." if prediction[0] == 1
            else "The person is likely to be healthy."
        )

        return jsonify({'result': result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'result': "An error occurred while processing your request."}), 500


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting the Flask app: {e}")
