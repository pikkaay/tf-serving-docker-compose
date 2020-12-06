import numpy as np
import json, requests
from io import BytesIO
from flask import Flask, request, jsonify


app = Flask(__name__)

img = np.random.random((1, 30,30,3))
imlst = img.tolist()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict')
def image_quality():

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': imlst}]
    }

    # Making POST request
    url = 'http://image-serving:8501/v1/models/img_cls:predict'
    data = json.dumps({"signature_name": "serving_default", "instances":imlst})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    out = np.array((json.loads(json_response.text)['predictions']))

    # Returning JSON response
    return jsonify({"status": 200, "message": out.shape})
