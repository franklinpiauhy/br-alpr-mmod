# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:00:00 2019

@author: lucas
"""

import dlib
import cv2
import seg
import string
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

file = 'D:\\UFPR-ALPR dataset\\training_seg\\track0001\\track0001[01].png'

digit_model_path = 'classification\\digit_otsu.26-0.08.hdf5'
letter_model_path = 'classification\\letter_otsu.49-0.83.hdf5'

W, H = 300, 100

print("loading detector model...")
detector = dlib.fhog_object_detector("plate_detector.svm")
print("done!")

print("loading digit model...")
digit_model = load_model(digit_model_path)
print("done!")


print("loading letter model...")
letter_model = load_model(letter_model_path)
print("done!")

clr_img = cv2.imread(file)
img = cv2.cvtColor(clr_img, cv2.COLOR_BGR2GRAY)

dets = detector(img, 1)
print("Number of plates detected: {}".format(len(dets)))
for k, det in enumerate(dets):
    plate = img[det.top():det.bottom(), det.left():det.right()]
    plate = cv2.resize(plate, (W, H))
    cv2.rectangle(clr_img,(det.left(),det.top()),(det.right(),det.bottom()),(0,0,255),2)
    break
    _, boxes = seg.process(plate)
    if len(boxes) == 0:
        continue
    letters = [plate[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[:3]]
    digits = [plate[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[3:]]
    
    plate_text = []
    
    for l in letters:
        l = cv2.resize(l, (32,32))
        l = cv2.equalizeHist(l)
        _, l = cv2.threshold(l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        l = img_to_array(l)/255.5
        pred = letter_model.predict(np.array([l]), batch_size = 128, verbose = 0)
        pred = np.argmax(pred, axis=1)[0]
        plate_text.append(list(string.ascii_uppercase)[pred])

    for d in digits:
        d = cv2.resize(d, (32,32))
        d = cv2.equalizeHist(d)
        _, d = cv2.threshold(d,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        d = img_to_array(d)/255.5
        pred = digit_model.predict(np.array([d]), batch_size = 128, verbose = 0)
        pred = np.argmax(pred, axis=1)[0]
        plate_text.append(list(string.digits)[pred])
    
    plate_text = ''.join(plate_text)

    cv2.putText(clr_img, plate_text, (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imwrite('plate.png', clr_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()           