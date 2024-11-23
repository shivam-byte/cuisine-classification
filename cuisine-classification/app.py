from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
models = {
    "KNeighbors": pickle.load(open("models/KNeighbors.pkl", "rb")),
    "Adaboost": pickle.load(open("models/adaboost_classifier.pkl", "rb")),
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
    "SVC": pickle.load(open("models/support_vector_classifier.pkl", "rb"))
}

# Input features
FEATURES = [
    "fish", "soy_sauce", "cayenne", "cumin", "starch",
    "roasted_sesame_seed", "sesame_oil", "scallion", "coriander", "coconut"
]

# Cuisine labels
CUISINES = ["Chinese", "Japanese", "Korean", "Indian", "Thai"]

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES, models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    inputs = [int(request.form[feature]) for feature in FEATURES]
    model_name = request.form['model']
    selected_model = models[model_name]

    # Make prediction
    prediction = selected_model.predict([inputs])[0]
    predicted_cuisine = CUISINES[prediction]

    return render_template('result.html', cuisine=predicted_cuisine, model=model_name)

if __name__ == "__main__":
    app.run(debug=True)
