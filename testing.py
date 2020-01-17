import os
import sys
import numpy as np
import cv2
import string
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from skimage import feature
from functions import *
import pickle

def testing(img, model, lbpmean, lbpstddev, winshape, radius, neighborpts):

    height, width, rgb = winshape

    predictions = []

    img, facelocations, detectedfaces = detectfaces(img)

    if len(detectedfaces) != 0:
        
        for i in range(len(detectedfaces)):

            lbp = feature.local_binary_pattern(detectedfaces[i], neighborpts, radius, method="nri_uniform")
            hist = lbp_histogram(lbp, neighborpts)
            normallbp = lbp_norm_with_data(hist,lbpmean,lbpstddev)
            predictions.append(model.predict(normallbp.reshape(1, -1)))

        i = 0
        for (x, y, w, h) in facelocations:
            cv2.putText(img, predictions[i][0], (int(round(x)), int(round(y))), cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (255, 255, 0), 3)
            i = i+1
        
    return img, detectedfaces, predictions

