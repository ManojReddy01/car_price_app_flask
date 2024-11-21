
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved model
model_path = r'C:\Users\91950\Desktop\car price\car_price_model_again.pkl'
car_prediction_model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form and convert to list of floats
        features = [
            float(request.form['HONDA']),
            float(request.form['BMW']),
            float(request.form['TOYOTA']),
            float(request.form['NISSAN']),
            float(request.form['CONDITION']),
            float(request.form['ODOMETER']),
            float(request.form['No_of_Doors']),
        ]
        
        # Convert to 2D array for model prediction
        features_array = np.array([features])
        
        # Predict using the loaded model
        prediction = car_prediction_model.predict(features_array)[0]
        
        # Round the prediction to 2 decimal places
        prediction_rounded = round(prediction, 2)
        
        # Pass the rounded prediction to the result template
        return render_template('result.html', prediction=prediction_rounded)
    
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

