#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Some pakages are needed,
# Tensorflow, keras, and flask
# If you are unfamiliar with how to install these, please go to their official website

import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# Assume packages are installed, and the first step would be to make a Flask object
application = Flask(__name__)

'''
Example 1: GET request
To quickly test the flask application, we want to create a route for a simple GET request
If we type/curl a url in a web browser/terminal, we want to see something return back.

This would be the simplest test example
application.route is a decorator that route any traffic to '/' to call the function running
in this case, if we curl http://127.0.0.1:5000/, you could see "Testing the landing page! This worked" under
'''
@application.route('/', methods=['GET'])
def running():
    return "Testing the landing page! This worked"


'''
Example 2: POST request
A Post request will send parameters in the post body as opposed to in the URL as with GET request.
In our example, if you want to deploy a deep learning model it will be easier and more secure to send parameter via a POST body
We'll specify a route, json-example, just to distinguish it from the previous example
We can get the information in the body by calling request.get_json() or request.json
Then we create a response message and return it back
'''
@application.route('/json-example', methods=['POST'])
def jsonexmaple():
    req_data = request.get_json(force=True)
    msg = req_data.get('message')
    out = {'output': msg}
    return jsonify(out)
# Once you run this server, you should be able to curl it in terminal via
# curl -X POST http://127.0.0.1:5000/json-example -d '{"message":"this is a message"}'


'''
Example 3: Model Inference
In this example we'll use a keras model for inference
Load  your model and any other preprocessing dependencies(tokenizer) in the global scope so that it is cache in memory
'''
with open('./dumps/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def get_model():
    # global model
    model = load_model('./dumps/whole_model.h5')
    model._make_predict_function()
    return model


model = get_model()


# This is the main route to perform the inference.
@application.route('/keras-example', methods=['POST'])
def keras_exmample():
    req_data = request.get_json(force=True)
    msg = req_data.get('message')

    # tokenizing and model inference
    seq = tokenizer.texts_to_sequences([msg])
    seq = pad_sequences(seq, maxlen=50, padding='post')
    pred = model.predict(seq).tolist()[0][0]
    out = {'prob': pred}
    return jsonify(out)


# run it via
# curl -X POST http://127.0.0.1:5000/keras-example -d '{"message":"this is a message"}' -H "Content-Type: application/json"
# it'll return model inference probability back

if __name__ == "__main__":
    application.run(debug=True)
