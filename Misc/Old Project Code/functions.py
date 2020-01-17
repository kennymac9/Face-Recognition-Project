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
        
        #except:
                #print("no face")
                # cv2.imshow('no face',img)
                #cv2.waitKey(0)
                #return 0, 0, 0, 0

def error_and_acc(model, prediction_error, dataset_size):
        model_error = float(prediction_error)/float(dataset_size)
        model_acc = 1 - model_error
        print("{} Accuracy: ".format(model) + " {:.2%}".format(model_acc))

def lbp_histogram(lbp, neighborpts):

        n_bins = 58 #int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        #print(hist)

        # hist, _ = np.histogram(lbp.ravel(),
        # bins=np.arange(0, neighborpts + 3),
        # range=(0, neighborpts + 2))

        # hist = hist.astype("float")
        # hist /= (hist.sum() + 1e-7)

        return hist

def lbp_cvt_norm(lbp):
        row, col = np.shape(lbp)

        lbpcvt = np.empty((row, col), dtype=object)
        lbpcols = []
        lbpmean = []
        lbpstddev = []

        uniform = [0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57]
        

        for i in range(row):
                for j in range(col):
                        lbpcvtval = uniform[int(lbp[i][j])]
                        lbpcvt[i][j] = lbpcvtval



        for j in range(col):
                lbpcols = []
                
                for i in range(row):
                        uniformlbp = lbpcvt[i][j]
                        lbpcols.append(uniformlbp)
                
                calcmean = np.mean(lbpcols)
                calcstddev = np.std(lbpcols)
                lbpmean.append(calcmean)
                lbpstddev.append(calcstddev)

        lbpnorm = lbpcvt.copy()

        for j in range(col):
                lbpcols = []
                for i in range(row):
                      lbpnorm[i][j] = (lbpnorm[i][j]-lbpmean[j])/(lbpstddev[j])

        # print('space')
        # print(lbpcvt[0][0])
        # print(lbpmean[0])
        # print(lbpstddev[0])
        # print(lbpnorm[0][0])
        # print('space')
        # print(lbpcvt[10][30])
        # print(lbpmean[10])
        # print(lbpstddev[10])
        # print(lbpnorm[10][30])
        # print('space')
        # print(lbpcvt[110][10])
        # print(lbpmean[110])
        # print(lbpstddev[110])
        # print(lbpnorm[110][10])
        # print('space')
        # print(lbpcvt[30][10])
        # print(lbpmean[30])
        # print(lbpstddev[30])
        # print(lbpnorm[30][10])


        # print(len(lbpmean))
        # print(len(lbpstddev))
        # print(lbpmean[0])
        # print(lbpstddev[0])

        return lbpnorm, lbpmean, lbpstddev
        
