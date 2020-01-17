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
from testing import *
import time

trainingdir = "./trainingFaces"

requests = raw_input('linear model or KNN model \n')
if requests == 'linear':
    model_filename = 'LinearSVCfinalized_model.sav'
if requests == 'knn':
    model_filename = 'KNeigborsfinalized_model.sav'


model = pickle.load(open(model_filename, 'rb'))
with open ('lbpmean', 'rb') as fp:
    lbpmean = pickle.load(fp)
with open ('lbpstddev', 'rb') as fp:
    lbpstddev = pickle.load(fp)
with open ('neighboring points', 'rb') as fp:
    neighborpts = pickle.load(fp)
with open ('radius', 'rb') as fp:
    radius = pickle.load(fp)


winname = 'Cam Output:'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 100, 100)



def camera_loop():
    delay = input('Delay to add detected faces to training? Enter 1 or more \n')
    print("Press <Esc> or Q to exit.")
    if delay >= 1:
        print('If all faces are correct press space to save to respective training folder')
    ti = time.gmtime() 
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)

        if action == ord('q') or action == 27:
            break


        
        winshape = np.shape(frame)
        frame, detectedfaces, predictions = testing(frame, model, lbpmean, lbpstddev, winshape,radius,neighborpts)
            
        cv2.imshow(winname, frame)
        
        if delay >= 1:
            time.sleep(delay)
            if action == ord(' '):
                print('saving')
                for i in range(len(detectedfaces)):
                    folder = predictions[i][0]
                    print(folder)
                    filepath = os.path.join(trainingdir,folder)
                    print(filepath)
                    cv2.imwrite(os.path.join(filepath ,"%s %s.jpg" %(predictions[i], time.asctime(ti))), detectedfaces[i])
                    time.sleep(0.2)


    cap.release()


camera_loop()

