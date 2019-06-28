# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:51:55 2019

@author: lucas
"""

import glob 
import os
import xml.etree.ElementTree as ET  
import cv2 

load_dir = 'D:\\UFPR-ALPR dataset\\testing\\'
save_dir = 'D:\\UFPR-ALPR detection dataset\\testing\\'

dataset = ET.Element("dataset")
ET.SubElement(dataset, "name").text = "Validation plate features"
ET.SubElement(dataset, "comment").text = "These are images from the UFPR-ALPR dataset."
images = ET.SubElement(dataset, "images")


for f in glob.glob(os.path.join(load_dir,'**','*.png'), recursive=True):
    metadata = f.split('.')[0] + '.txt'
    save_f = save_dir + f.split('\\')[-1]
    img = cv2.imread(f)
    
    file = open(metadata, "r")
    for i in range(0,2): file.readline()
    type_veic = file.readline().split(":")[-1].split()[0]
    if  type_veic != 'car':
        continue
    for i in range(3,7): file.readline()
    
    [x,y,w,h] = [int(i) for i in file.readline().split(":")[-1].split()] # [x, y, w, h]
    sz = img.shape
    new_x, new_y = max(x-50, 0), max(y-50, 0)
    new_w, new_h = min(sz[1]-new_x-1, w+100), min(sz[0]-new_y-1, h+100)
    file.close()
    img = img[new_y:new_y+new_h, new_x:new_x+new_w]
    cv2.imwrite(save_f, img)    
    image = ET.SubElement(images, "image", file=save_f)
    box = ET.SubElement(image, "box", top=str(y-new_y), left=str(x-new_x), width=str(w), height=str(int(w/3)))
            
tree = ET.ElementTree(dataset)
tree.write(save_dir+"plate_testing.xml")
