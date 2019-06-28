# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:34:32 2019

@author: lucas
"""

import os

import dlib

plates_folder = 'D:\\UFPR-ALPR detection dataset\\testing\\'

testing_xml_path = os.path.join(plates_folder, "plate_testing.xml")

print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "plate_detector.svm")))