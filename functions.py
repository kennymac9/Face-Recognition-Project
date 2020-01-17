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

def detectfaces(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)

        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        # eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

        cascadefaces = face_cascade.detectMultiScale(gray, 1.3, 5)
        detectedfaces = []

        #try:    
        for (x,y,w,h) in cascadefaces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_color = img[y:y+h, x:x+w]

                roi_gray = gray[y:y+h, x:x+w]
                cropped_face = cv2.resize(roi_gray, (200,200), interpolation = cv2.INTER_AREA)
                detectedfaces.append(cropped_face)

                # eyes = eye_cascade.detectMultiScale(roi_gray)
                # for (ex,ey,ew,eh) in eyes:
                #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


        #print('Number of faces: {}'.format(len(detectedfaces)))
        return img, cascadefaces, detectedfaces

def error_and_acc(model, prediction_error, dataset_size):
        model_error = float(prediction_error)/float(dataset_size)
        model_acc = 1 - model_error
        print("{} Accuracy: ".format(model) + " {:.2%}".format(model_acc))

def lbp_histogram(lbp, neighborpts):

        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))

        return hist

def lbp_normalize_values(lbp):
        row, col = np.shape(lbp)
        lbpcols = []
        lbpmean = []
        lbpstddev = []
        newlbp = np.empty((row,col), dtype=float, order='C')


        for j in range(col):
                lbpcols = []
                for i in range(row):
                        lbpcols.append(lbp[i][j])
                
                calcmean = np.mean(lbpcols)
                calcstddev = np.std(lbpcols)
                lbpmean.append(calcmean)
                lbpstddev.append(calcstddev)


        for j in range(col):
                for i in range(row):
                        lbpdata = float(lbp[i][j])
                        mean = float(lbpmean[j])
                        std = float(lbpstddev[j])
                        calc = (lbpdata-mean)/(std)
                        newlbp[i][j] = calc
                        

        return newlbp, lbpmean, lbpstddev, 
        
def lbp_norm_with_data(lbp, lbpmean, lbpstddev):
        bins = len(lbp)
        newlbp = np.empty((bins), dtype=float, order='C')

        for i in range(bins):
                newlbp[i] = ((lbp[i]-lbpmean[i])/(lbpstddev[i]))

        
        return newlbp

