
import dlib
import cv2
import seg
import string
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import glob
import os
from matplotlib import pyplot as plt
from difflib import SequenceMatcher 

DIR = 'D:\\UFPR-ALPR dataset\\testing_seg\\'

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

all_correct = 0
six_correct = 0
five_correct = 0
total = len(glob.glob(os.path.join(DIR,'**','*.png'), recursive=True))

def get_boxes(chars):
    boxes = []
    for [x,y,w,h] in chars:
        boxes.append({'x':x, 'y':y, 'w':w, 'h':h})
    return boxes

for f in glob.glob(os.path.join(DIR,'**','*.png'), recursive=True):
    print('file: {}'.format(f))
    metadata = f.split('.')[0] + '.txt'
    with open(metadata, 'r') as file:
        for i in range(0,2): file.readline()
        type_veic = file.readline().split(":")[-1].split()[0]
        if  type_veic != 'car':
            continue
        for i in range(3,6): file.readline()
        real_plate_text = file.readline().split(":")[-1].split()[0]
        real_plate_text = real_plate_text.replace('-', '')
        
        file.readline()
        chars = [[int(x) for x in file.readline().split(":")[-1].split()] for i in range(0,7)]
    
    clr_img = cv2.imread(f)
    img = cv2.cvtColor(clr_img, cv2.COLOR_BGR2GRAY)
    
    dets = detector(img, 1)
    print("Number of plates detected: {}".format(len(dets)))
    for k, det in enumerate(dets):
        if k>0: continue
        #plate = img[det.top():det.bottom(), det.left():det.right()]
        #if plate.shape[0]==0 or plate.shape[1]==0: continue
        #plate = cv2.resize(plate, (W, H))
        #cv2.rectangle(clr_img,(det.left(),det.top()),(det.right(),det.bottom()),(0,0,255),2)
        #plate, boxes = seg.process(plate)
        boxes = get_boxes(chars)
        if len(boxes)==0: continue
        boxes = sorted(boxes, key=lambda k:k['x'])
        #for box in boxes:
        #    cv2.rectangle(plate,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(0,0,255),2)
        #plt.imshow(plate, cmap='gray')
        #plt.show()
        if len(boxes) == 0:
            continue
        letters = [img[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[:3]]
        digits = [img[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] for b in boxes[3:]]
        
        plate_text = []
        
        for l in letters:
            l = cv2.resize(l, (28,28))
            l = cv2.equalizeHist(l)
            _, l = cv2.threshold(l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #plt.imshow(l, cmap='gray')
            #plt.title(plate_text[i])
            #plt.show()
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
        print(plate_text, real_plate_text)
        r = SequenceMatcher(None,plate_text,real_plate_text).ratio()
        if r>=5.0/7.0:
            five_correct += 1
        if r>=6.0/7.0:
            six_correct += 1
        if r==7.0/7.0:
            all_correct += 1
        #cv2.putText(clr_img, plate_text, (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    #cv2.imshow('plate', cv2.resize(clr_img,None, fx=0.5, fy=0.5))
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
#cv2.destroyAllWindows()

print("Accuracy for all correct chars: {} %".format(100*all_correct/total))
print("Accuracy for 6 correct chars: {} %".format(100*six_correct/total))           
print("Accuracy for 5 correct chars: {} %".format(100*five_correct/total))           