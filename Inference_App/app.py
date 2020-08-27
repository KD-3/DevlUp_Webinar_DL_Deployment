from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
from scipy.misc import imread, imresize
import re

app = Flask(__name__)

from load import *
global model
model = init()

def convertImage(imgData):
    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    img_data = request.get_data()
    convertImage(img_data)
    x = imread('output.png', mode='L')
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
        # convert the response to a string
    response = np.argmax(out, axis=1)
    return str(response[0])



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)