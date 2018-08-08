# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:51:16 2018

@author: subha
"""

import pandas as pd
import cv2
import numpy as np

DataFrame = pd.read_csv("../content/kaggle/train_ship_segmentations.csv")
img_names = DataFrame['ImageId']
rle_encoded = DataFrame['EncodedPixels']

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)
#%%
with open('lost.txt', 'w') as f:
    for i in range(0, len(rle_encoded)):
        print(i)
        if(isinstance(rle_encoded[i], float)):
            img = np.zeros((768, 768))
        else:
            img = np.array(rle_decode(rle_encoded[i], (768, 768)))
            #cv2.imshow('I', img)
            #cv2.waitKey(0)
        val = cv2.imwrite("../content/kaggle/outputs/"+img_names[i], img)
        print(img.shape, val)
        if(val == False):
            f.write("../content/kaggle/outputs/"+img_names[i])
        
        