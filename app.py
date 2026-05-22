from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask App
app = Flask(__name__)

# Load Trained Model
model = pickle.load(open("model.pkl", "rb"))

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get Data From Form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Convert into array
    input_data = np.array([[cgpa, iq]])

    # Prediction
    prediction = model.predict(input_data)

    # Result
    if prediction[0] == 1:
        result = "Student Will Get Placed ✅"
    else:
        result = "Student Will NOT Get Placed ❌"

    # Send Result to HTML
    return render_template('index.html', result=result)

# Run App
if __name__ == "__main__":
    app.run(debug=True)