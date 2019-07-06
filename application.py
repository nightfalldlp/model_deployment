#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Some pakages are needed,
### Tensorflow, keras, and flask
### If you don't know how to install these, please go to their official website
import pickle
import tensorflow as tf
from flask import Flask,request,jsonify
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


## assume packages are installed, and the first step would be make a Flask object.
application = Flask(__name__)

## Essentially we want a simple server, meaning if we type/curl a url in a web browser/terminal, we want to see something return back.

## This would be simplest example for this type of communication, where
## application.route is a decorator that tells you where you could see designed content.
## in this case, if we run the whole script, you could see "why r you so beautiful" under http://127.0.0.1:5000/
@application.route('/',methods=['GET'])
def running():
    return "why r u so beautiful?"

# skip this
# @application.route('/query-example')
# def query_example():
# 	lang = request.args.get('language')
# 	if lang is None:
# 		return 'There is nothing'
# 	return lang


## second exmaple, POST request: while GET request is just you give a url and the server directly return what they have there
## A Post request will ask you turn apply a input, such that the server takes the input and return u the designed output.
## In our example, if you want to deploy a deep learning model, you definitely want a custimized input, thus we need this POST request
## Similar to GET request, we'll specify url component first in as /json-exmaple just to distinguish previous /
## we change methods to POST this time and try to read from as json format by request.get_json() or request.json
## Then we get fiels message and return it back
@application.route('/json-example',methods=['POST'])
def jsonexmaple():
	req_data = request.get_json(force=True)
	msg = req_data.get('message')

	out = {'output':msg}
	return jsonify(out)
## Once you run this server, you should be able to curl it in terminal via
## curl -X POST http://127.0.0.1:5000/json-example -d '{"message":"this is a message"}'

#### Third Example Using keras to run inference, this time we'll add keras exmaple
#### loading your keras model, in this case we need both tokenizer and model.
with open('./dumps/tokenizer.pkl','rb') as f:
	tokenizer = pickle.load(f)

# with open('./dumps/model_json.json','rb') as f:
# 	model_json = f.read()

def get_model():
	# global model
	model = load_model('./dumps/whole_model.h5')
	model._make_predict_function()
	return model

model = get_model()

 ## application is similar to last example, the only thing differs is we'll need a model inference part
@application.route('/keras-example',methods=['POST'])
def keras_exmample():
	req_data = request.json
	msg = req_data.get('message')

	## tokenizing and model inference
	seq = tokenizer.texts_to_sequences([msg])
	seq = pad_sequences(seq,maxlen=50,padding='post')
	pred = model.predict(seq).tolist()[0][0]
	out = {'prob':pred}

	return jsonify(out)

## run it via 
## curl -X POST http://127.0.0.1:5000/keras-example -d '{"message":"this is a message"}' -H "Content-Type: application/json"
## it'll return model inference probability back

if __name__ == "__main__":
    application.run(debug=True)
