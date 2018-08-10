# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:51:16 2018

@author: subha
"""



import pandas as pd
import cv2
import numpy as np

DataFrame = pd.read_csv("train_ship_segmentations.csv")
img_names = DataFrame['ImageId']
rle_encoded = DataFrame['EncodedPixels']
img_names = pd.unique(img_names)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T
#%%
with open('lost.txt', 'w') as f:
    for i in range(0, len(img_names)):
        print(i)
        img_masks = DataFrame.loc[DataFrame['ImageId'] == img_names[i], 'EncodedPixels'].tolist()
        img = np.zeros((768, 768))
        for j in img_masks:
            if(isinstance(j, float)):
                img += np.zeros((768, 768))
            else:
                #img = np.zeros((768, 768))
                img += np.array(rle_decode(j, (768, 768)))
        #cv2.imshow('I', img)
        #cv2.waitKey(0)
        val = cv2.imwrite("/outputs/"+img_names[i], img)
        print(img.shape, val)
        if(val == False):
             f.write("/outputs/"+img_names[i])
        
        






