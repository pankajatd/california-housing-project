
"""
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  # fixed typo from reder_template

@app.route('/predict_api', methods=['POST'])  # fixed typo: method -> methods
def predict_api():
    try:
        # Convert form input to dict and then DataFrame
        input_data = request.form.to_dict()  # this gives key:value pairs
        # Convert all values to float
        for key in input_data:
            
            input_data[key] = float(input_data[key])

        # Create DataFrame with same columns as scaler was trained on
        df_input = pd.DataFrame([input_data])  

        # Scale
        transformed_data = scaler.transform(df_input)

        # Predict
        prediction = model.predict(transformed_data)

        # Return as JSON
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# driver code
if __name__ == "__main__":
    app.run(debug=True)
"""


##=================New Version====================##
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'MedInc': float(request.form['MedInc']),
            'HouseAge': float(request.form['HouseAge']),
            'AveRooms': float(request.form['AveRooms']),
            'AveBedrms': float(request.form['AveBedrms']),
            'Population': float(request.form['Population']),
            'AveOccup': float(request.form['AveOccup']),
            'Latitude': float(request.form['Latitude']),
            'Longitude': float(request.form['Longitude'])
        }

        df = pd.DataFrame([input_data])

        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)[0]

        return render_template(
            'home.html',
            prediction_text=f"Predicted House Price: ${prediction * 100000:.2f}"
        )

    except Exception as e:
        return render_template('home.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)

