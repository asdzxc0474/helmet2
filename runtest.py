from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import base64
import numpy as np
import cv2
app = Flask(__name__)
CORS(app, supports_credentials=True)



@app.route('/', methods=['POST', 'GET'])
def getinput():
    img_str = request.form['image']
    img_byte = base64.b64decode(img_str)
    image = np.fromstring(img_byte, np.uint8)
    images = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    result = str('jimmy')

    return str(result)
if __name__ =='__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
