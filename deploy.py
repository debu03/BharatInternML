from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the scaler if it was used during model training
# scaler = joblib.load('scaler.pkl')  # Uncomment this line if scaler was used

# Load the model and any other necessary preprocessing steps
try:
    model = joblib.load('iris_model.pkl')
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None  # Assigning None if model loading fails

IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home():
    global model  # Ensure access to the model loaded globally

    if request.method == 'POST':
        try:
            sl = float(request.form['SepalLength'])
            sw = float(request.form['SepalWidth'])
            pl = float(request.form['PetalLength'])
            pw = float(request.form['PetalWidth'])
        except ValueError:
            return render_template('index.html', prediction="Invalid input. Please enter numerical values.")

        if model is not None:
            data = np.array([[sl, sw, pl, pw]])
            prediction = model.predict(data)
            image = prediction[0] + '.png'
            image = os.path.join(app.config['UPLOAD_FOLDER'], image)
            return render_template('index.html', prediction=prediction[0], image=image)
        else:
            return render_template('index.html', prediction="Model not available.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
