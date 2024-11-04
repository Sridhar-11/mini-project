from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__, template_folder='templates')

# Placeholder for prediction function
def predict_liver_disease(features):
    """
    Dummy prediction function.
    In practice, load a trained model and predict based on input features.
    """
    # Replace with real model prediction code
    # Example: model.predict([features])
    return "Liver Disease Detected" if np.mean(features) > 1 else "No Liver Disease Detected"

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form data
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        total_bilirubin = float(request.form['totalBilirubin'])
        direct_bilirubin = float(request.form['directBilirubin'])
        alkaline_phosphatase = float(request.form['alkalinePhosphatase'])
        alamine_aminotransferase = float(request.form['alamineAminotransferase'])
        aspartate_aminotransferase = float(request.form['aspartateAminotransferase'])
        total_proteins = float(request.form['totalProteins'])
        albumin = float(request.form['albumin'])
        albumin_globulin_ratio = float(request.form['albuminGlobulinRatio'])

        # Organize input data for the model
        features = [
             age, sex, total_bilirubin, direct_bilirubin,
            alkaline_phosphatase, alamine_aminotransferase,
            aspartate_aminotransferase, total_proteins,
            albumin, albumin_globulin_ratio
        ]

        # Get prediction from model (dummy function here)
        prediction = predict_liver_disease(features)

        # Return the prediction to the client as JSON
        return jsonify(prediction=prediction)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
