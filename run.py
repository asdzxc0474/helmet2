from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import detect
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def getinput():
    img_str = request.form['image']
    img_byte = base64.b64decode(img_str)
    image = np.fromstring(img_byte, np.uint8)
    frame,pred = detector.predict(image)
    return str(pred)
if __name__ =='__main__':
    detector = detect()
    app.run(host='127.0.0.1', port=500, debug=True)
