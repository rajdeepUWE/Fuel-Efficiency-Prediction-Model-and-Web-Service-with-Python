#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

app = Flask(__name__)

# Load the pre-trained model using pickle
with open('model_files/model.bin', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Machine Learning Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Use the predict_mpg function from ml_model module
        result = predict_mpg(model, data)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)



