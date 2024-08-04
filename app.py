from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('rice_price_predictor1.pkl')

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from request form
    features = [float(request.form['feature1']), float(request.form['feature2'])]  # Add more features as necessary
    final_features = [np.array(features)]

    # Make prediction
    prediction = model.predict(final_features)

    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Rice Price: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
