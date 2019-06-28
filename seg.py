# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:00:04 2019

@author: lucas
"""

import cv2
import numpy as np
import glob
import os
from math import atan, degrees
from sklearn.linear_model import LinearRegression
''''
DIR = 'D:\\UFPR-ALPR dataset\\training_seg\\'
SAVE_DIR = 'D:\\UFPR-ALPR dataset\\seg\\' '''
W, H = 300, 100

kernel_sharpening = 3*np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

'''total_chars = 0
total_plates = 0
total_imgs = len(glob.glob(os.path.join(DIR,'**','*.png'), recursive=True))'''


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def find_nonzero_runs(a):
        # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
        isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isnonzero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

def process(cimg):
    
    img = cv2.equalizeHist(cimg)
        
    img = cv2.filter2D(img, -1, kernel_sharpening)
    
    _, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
    
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    opening = closing
    
    contours, _ = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    boxes = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if H/h>1.2 and H/h<3.0:
            boxes.append({"x": x, "y": y,"w": w,"h": h, "contour": contour})
    if len(boxes) < 2:
        _, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 1)
        contours, _ = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        boxes = []
        for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            if H/h>1.2 and H/h<3.0:
                boxes.append({"x": x, "y": y,"w": w,"h": h, "contour": contour})
    if len(boxes) < 2:
        return opening, []
    while len(boxes)>7:
        regressor = LinearRegression()  
        regressor.fit([[boxes[i]['x']] for i in range(len(boxes))], [boxes[i]['y'] for i in range(len(boxes))])  
        b = (regressor.intercept_, regressor.coef_)
        error =abs(boxes[0]['y']-(b[0]+b[1]*boxes[0]['x']))
        outline = 0
        for i in range(1,len(boxes)):
            if abs(boxes[i]['y']-(b[0]+b[1]*boxes[i]['x'])) > error:
                error = abs(boxes[i]['y']-(b[0]+b[1]*boxes[i]['x']))
                outline = i
        del(boxes[outline])
        
    regressor = LinearRegression()  
    regressor.fit([[boxes[i]['x']] for i in range(len(boxes))], [boxes[i]['y'] for i in range(len(boxes))])  
    angle = atan(regressor.coef_) # radians
    rst_img = rotate_bound(opening, -degrees(angle))
    
    contours, _ = cv2.findContours(rst_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    boxes = []
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if H/h>1.2 and H/h<3.0:
            boxes.append({"x": x, "y": y,"w": w,"h": h, "contour": contour})
    while len(boxes)>7:
        regressor = LinearRegression()  
        regressor.fit([[boxes[i]['x']] for i in range(len(boxes))], [boxes[i]['y'] for i in range(len(boxes))])  
        b = (regressor.intercept_, regressor.coef_)
        error =abs(boxes[0]['y']-(b[0]+b[1]*boxes[0]['x']))
        outline = 0
        for i in range(1,len(boxes)):
            if abs(boxes[i]['y']-(b[0]+b[1]*boxes[i]['x'])) > error:
                error = abs(boxes[i]['y']-(b[0]+b[1]*boxes[i]['x']))
                outline = i
        del(boxes[outline])
    if len(boxes)>1:
        boxes_x = sorted(boxes, key=lambda k:k['x'])
        h = int(sum([box['h'] for box in boxes])/len(boxes))
        px, py, pw, ph = boxes_x[0]['x'], boxes_x[0]['y'], boxes_x[-1]['x'] + boxes_x[-1]['w']-boxes_x[0]['x'], h
        rst_img = rst_img[py:py+ph, px:px+pw]
    
        horizontal_hist = [sum(rst_img[:,i]) for i in range(rst_img.shape[1])]
        havg = int(np.average(horizontal_hist)/5)
        horizontal_hist = [max(x-havg, 0) for x in horizontal_hist]
        blocks = find_nonzero_runs(horizontal_hist)
        boxes = []
        for b in blocks:
            #cv2.rectangle(image2,(b[0],cut_v[0]),(b[1],cut_v[1]),(0,0,255),2)
            boxes.append({"x": px+b[0], "y": py,"w": b[1]-b[0],"h": h})
        boxes = sorted(boxes, key=lambda k:k['w'])
        while(len(boxes)>7):
            del(boxes[0])
            
    rst_img = rotate_bound(cimg, -degrees(angle))
    
    #show_img = cv2.cvtColor(rst_img, cv2.COLOR_GRAY2BGR)
    #for box in boxes:
    #    cv2.rectangle(show_img,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(0,0,255),2)
    return rst_img, boxes
'''
for f in glob.glob(os.path.join(DIR,'**','*.png'), recursive=True):
    metadata = f.split('.')[0] + '.txt'
    with open(metadata, 'r') as file:
        for i in range(0,2): file.readline()
        type_veic = file.readline().split(":")[-1].split()[0]
        if  type_veic != 'car':
            total_imgs -= 1
            continue
        for i in range(3,6): file.readline()
        plate_text = file.readline().split(":")[-1].split()[0]
        plate_text = plate_text.replace('-', '')
        pos = file.readline().split(":")[-1].split()
        [x,y,w,h] = [int(x) for x in pos]
        #print(plate_text, [x,y,w,h]) 
    print("file: " + f)    
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)[y:y+h,x:x+w]
    image = cv2.resize(image, (W, H))
    bin_img, boxes = process(image)
    for box in boxes:
        cv2.rectangle(bin_img,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(0,0,255),2)
    total_chars += len(boxes)
    if len(boxes)==7: 
        total_plates += 1
    name = os.path.join(SAVE_DIR, str(len(boxes)),f.split('\\')[-1])
    cv2.imwrite(name, bin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''