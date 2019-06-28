# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:38:06 2019

@author: lucas
"""

import dlib
import cv2
import seg
import string
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

video_file = 'C:\\Users\\lucas\\Videos\\moto.mp4'

digit_model_path = 'classification\\new_digit_otsu.61-0.11.hdf5'
letter_model_path = 'classification\\new_letter_otsu.28-0.75.hdf5'

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

cap = cv2.VideoCapture(video_file)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1920,1080))

k=0
while(cap.isOpened()):
    ret, frame = cap.read()
    #k+=1
    #if k <410: continue
    #if k > 3300: continue
    #print(k)
    
    if not ret:
        break
    #frame = cv2.resize(frame,None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dets = detector(frame, 1)
    print("Number of plates detected: {}".format(len(dets)))
    for k, det in enumerate(dets):
        plate = img[det.top():det.bottom(), det.left():det.right()]
        if plate.shape[0]==0 or plate.shape[1]==0: continue
        plate = cv2.resize(plate, (W, H))
        cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,255,0),2)
        plate, boxes = seg.process(plate)
        #boxes = get_boxes(chars)
        if len(boxes)==0: continue
        boxes = sorted(boxes, key=lambda k:k['x'])
        if len(boxes) == 0:
            continue
        letters = [plate[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[:3]]
        digits = [plate[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[3:]]
        
        plate_text = []
        for l in letters:
            l = cv2.resize(l, (28,28))
            l = cv2.equalizeHist(l)
            _, l = cv2.threshold(l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            l = img_to_array(l)/255.0
            pred = letter_model.predict(np.array([l]), batch_size = 128, verbose = 0)
            pred = np.argmax(pred, axis=1)[0]
            plate_text.append(list(string.ascii_uppercase)[pred])
    
        for d in digits:
            d = cv2.resize(d, (28,28))
            d = cv2.equalizeHist(d)
            _, d = cv2.threshold(d,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            d = img_to_array(d)/255.0
            pred = digit_model.predict(np.array([d]), batch_size = 128, verbose = 0)
            pred = np.argmax(pred, axis=1)[0]
            plate_text.append(list(string.digits)[pred])
        
        plate_text = ''.join(plate_text)
        print(plate_text)
        if len (plate_text)==7: cv2.putText(frame, plate_text, (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    #cv2.imshow('plate', cv2.resize(frame,None, fx=0.5, fy=0.5))
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()                    