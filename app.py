import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import os

here = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
model_file = open("./model.pkl","rb")
model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Y value should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)