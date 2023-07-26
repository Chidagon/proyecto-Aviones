import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)

model = joblib.load('rdf.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    
    if prediction[0] == 0:
        return render_template('resultado.html', prediction_text=f'el cliente no está satisfecho')
    else:
        return render_template('resultado.html', prediction_text=f'el cliente está satisfecho')

if __name__ == "__main__":
    app.run( port=80)