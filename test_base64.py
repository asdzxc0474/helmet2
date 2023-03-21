import base64
import os
import cv2
from detect import detect

class test_base64_func(base64_str):
    
    detector = detect()
    while True:
        img = base64.b64decode(base64_str)  
        frame = detector.predict(img)
        frame = cv2.resize(frame,(1280,720))
        cv2.imshow('test',frame)
        if cv2.waitKey(1)==ord('q'):
            break
            
if __name__ == '__main__':
    base64_str = '' #base64_path
    test_base64_func()
