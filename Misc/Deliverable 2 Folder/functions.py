import os
import numpy as np
import cv2

def load_images_folder_and_directory(folder,directory):
    images = []
    filepath = os.path.join(directory,folder)
    for filename in os.listdir(filepath):
        img = cv2.imread(os.path.join(filepath,filename))
        if img is not None:
            images.append(img)
    return images

def load_images_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def detectface(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:    
        for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200), interpolation = cv2.INTER_AREA)
        
        return img, face
    except:
        print("no face")
        # cv2.imshow('no face',img)
        cv2.waitKey(0)
        return (), ()