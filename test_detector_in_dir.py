# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:01:46 2019

@author: lucas
"""

import os
import glob
import sys
import dlib
import cv2
import time

plates_folder = 'D:\\UFPR-ALPR dataset\\testing_seg\\'
save_folder = 'D:\\UFPR-ALPR detection dataset\\save tests 2\\'
# Now let's use it as you would in a normal application.  First we will load it
# from disk. We also need to load a face detector to provide the initial
# estimate of the facial location.
detector = dlib.simple_object_detector("plate_detector.svm") 

# Now let's run the detector and shape_predictor over the images in the faces
# folder and display the results.
#print("Showing detections and predictions on the images in the faces folder...")
#win = dlib.image_window()
start = time.time()
for f in glob.glob(os.path.join(plates_folder,'**', "*.png"), recursive=True):
    #print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    #img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #win.clear_overlay()
    #win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img)
    break
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print("Number of plates detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        img = cv2.rectangle(img, (d.left(),d.top()),(d.right(), d.bottom()),(0,255,0),1)
        cv2.imwrite(save_folder + f.split('\\')[-1], img)
        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)

    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()
end = time.time()

print(end-start)