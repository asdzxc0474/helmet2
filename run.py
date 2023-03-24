from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import base64
import numpy as np
import cv2
from detect import detect
import time
app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/', methods=['POST', 'GET'])
def getinput():
    # img_str = request.form['image']
    gettxt = requests.post('http://192.168.1.114:5000/upload')
    img_str = gettxt.text
    img_byte = base64.b64decode(img_str)
    image = np.fromstring(img_byte, np.uint8)
    images = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    frame, pred = detector.predict(images)
    result = pred
    return str(result)
if __name__ =='__main__':
    detector = detect()
    app.run(host='192.168.1.114', port=5000, debug=True)