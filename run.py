from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from detect import detect
import time
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def getinput():
    t1 =time.time()
    detector = detect()
    img_str = request.form['image']
    img_byte = base64.b64decode(img_str)
    image = np.fromstring(img_byte, np.uint8)
    images = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    frame,pred = detector.predict(images)
    t2 =time.time()
    result = [pred, t2-t1]
    return str(result)
if __name__ =='__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)



