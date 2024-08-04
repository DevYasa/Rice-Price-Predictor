from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('rice_price_predictor.pkl')


# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from request form
    features = [
        float(request.form['anuradhapura_producer_price']),
        float(request.form['kurunegala_producer_price']),
        float(request.form['polonnaruwa_producer_price']),
        float(request.form['production']),
        float(request.form['production_total']),
        float(request.form['exchange_rate']),
        float(request.form['fuel_price'])
    ]
    final_features = [np.array(features)]

    # Make prediction
    prediction = model.predict(final_features)

    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Rice Price: {prediction[0]:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
