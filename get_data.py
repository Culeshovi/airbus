# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:12:28 2018

@author: subha
"""

import glob
import cv2
import numpy as np
import random

class DataLogger():
    IN = []
    OUT = []
    init = 0
    def __init__(self, DIR):
        self.IN = glob.glob(DIR+'/inputs/*.jpg')
        self.OUT = glob.glob(DIR+'/outputs/*.jpg')
        
    def process(self, img): # Pre-Processing Section for input images
        #img = cv2.resize(img, (320, 320)
        
        return img
    
    def get_batches(self, batch_size, shuffle=False): # Batching Section for inputs
        if(self.init>=len(self.IN)):
            self.init = 0
            if(shuffle):
                zipped = list(zip(self.IN, self.OUT))
                random.shuffle(zipped)
                self.IN, self.OUT = zip(*zipped)
        input_batch_path = self.IN[self.init:self.init+batch_size]
        output_batch_path = self.OUT[self.init:self.init+batch_size]
        IN_Batch = []
        OUT_Batch = []
        for i in range(0, len(input_batch_path)):
            temp = cv2.imread(input_batch_path[i]).astype('float32')
            temp = np.transpose(temp)
            temp = self.process(temp)
            temp = np.expand_dims(temp, axis=0)
            IN_Batch.append(temp)
            ##
            temp = cv2.imread(output_batch_path[i], 0)
            ret,thresh = cv2.threshold(temp,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            temp = thresh.astype('float32')
            temp = np.transpose(temp)
            temp = self.process(temp)
            temp = np.expand_dims(temp, axis=0)
            OUT_Batch.append(temp)
        IN_Batch = np.vstack(IN_Batch)
        OUT_Batch = np.vstack(OUT_Batch)
        self.init = self.init+batch_size
        return [IN_Batch, OUT_Batch]
              
        
        