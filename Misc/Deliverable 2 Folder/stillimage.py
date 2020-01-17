import os
import sys
import numpy as np
import cv2
import string
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from functions import *

print('Training')

trainingdir = "./trainingFaces"
testingfolder = "./validationFaces"
validationdir = "./validationFaces"

winname = 'Haar Faces'
winname1 = 'Cropped Face'
winname2 = 'LBP'
winname3 = 'Prediction'

cv2.namedWindow(winname)
cv2.namedWindow(winname1)        
cv2.namedWindow(winname2)
cv2.namedWindow(winname3)

cv2.moveWindow(winname, 100, 100)
cv2.moveWindow(winname1, 600, 100)  
cv2.moveWindow(winname2, 1100, 100)
cv2.moveWindow(winname3, 600, 500)


identifiers = []
lbpdata = []
dataset_size = 0
prediction_error = 0
prediction2_error = 0

neighborpts = 24
radius = 8

#Array of Images
for folder in os.listdir(trainingdir):
    if not folder.startswith('.'):
        training_images = load_images_folder_and_directory(folder,trainingdir)
        identifier = folder
        print(identifier)
    
        #Detecting Face in each image and create LBP
        for img in training_images:

            #height, width = img.shape[:2]

            img, face = detectface(img)
            smallimg = cv2.resize(img, (384, 286), interpolation=cv2.INTER_AREA)

            #cv2.imshow(winname, smallimg)
            #cv2.imshow(winname1, face)

            lbp = feature.local_binary_pattern(face, neighborpts, 
            radius, method="default")

            (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, neighborpts + 3),
            range=(0, neighborpts + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            #cv2.imshow(winname2, lbp)

            identifiers.append(identifier)
            lbpdata.append(hist)
            #cv2.waitKey(0)
            

model = LinearSVC(C=100.0, max_iter=1000)
model.fit(lbpdata, identifiers)

model2 = KNeighborsClassifier(n_neighbors=15, leaf_size = 30)
model2.fit(lbpdata, identifiers)
print('Models Done')


print('Testing Training Set')

for folder in os.listdir(trainingdir):
    if not folder.startswith('.'):
        training_images = load_images_folder_and_directory(folder,trainingdir)
        identifier = folder
        dataset_size = len(training_images)+dataset_size

        for img in training_images:
            
            img, face = detectface(img)
            smallimg = cv2.resize(img, (384, 286), interpolation=cv2.INTER_AREA)
            smallimgOrignial = cv2.resize(img, (384, 286), interpolation=cv2.INTER_AREA)

            # cv2.putText(smallimgOrignial, identifier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            # 1.0, (0, 0, 255), 3)
            # cv2.imshow(winname, smallimgOrignial)
            # cv2.imshow(winname1, face)

            lbp = feature.local_binary_pattern(face, neighborpts, 
            radius, method="default")

            hist, _ = np.histogram(lbp.ravel(),
            bins=np.arange(0, neighborpts + 3),
            range=(0, neighborpts + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            # cv2.imshow(winname2, lbp)

            prediction = model.predict(hist.reshape(1, -1))
            prediction2 = model2.predict(hist.reshape(1, -1))

            # cv2.putText(smallimg, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            # 1.0, (255, 255, 0), 3)
            # cv2.putText(smallimg, prediction2[0], (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 
            # 1.0, (0, 255, 0), 3)
            # cv2.imshow(winname3, smallimg)

            if prediction[0] != identifier:
                prediction_error = prediction_error + 1
            if prediction2[0] != identifier:
                prediction2_error = prediction2_error + 1

            # cv2.waitKey(0)


model_error = float(prediction_error)/float(dataset_size)
model2_error = float(prediction2_error)/float(dataset_size)
model_acc = 1 - model_error
model2_acc = 1 - model2_error
print("Training Data set Size: " + "{}".format(dataset_size))
print("LinearSVC Accuracy: "+" {:.2%}".format(model_acc))
print("KNeighbor Accuracy: "+" {:.2%}".format(model2_acc))


dataset_size = 0
prediction_error = 0
prediction2_error = 0


print('Validating')
print('Use any key to step through')
cv2.waitKey(0)

for folder in os.listdir(validationdir):
    if not folder.startswith('.'):
        validation_images = load_images_folder_and_directory(folder,validationdir)
        identifier = folder
        dataset_size = len(validation_images)+dataset_size

        for img in validation_images:
            
            img, face = detectface(img)
            smallimg = cv2.resize(img, (384, 286), interpolation=cv2.INTER_AREA)
            smallimgOrignial = cv2.resize(img, (384, 286), interpolation=cv2.INTER_AREA)

            cv2.putText(smallimgOrignial, identifier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (0, 0, 255), 3)
            cv2.imshow(winname, smallimgOrignial)
            cv2.imshow(winname1, face)

            lbp = feature.local_binary_pattern(face, neighborpts, 
            radius, method="default")

            hist, _ = np.histogram(lbp.ravel(),
            bins=np.arange(0, neighborpts + 3),
            range=(0, neighborpts + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

            cv2.imshow(winname2, lbp)
            prediction = model.predict(hist.reshape(1, -1))
            prediction2 = model2.predict(hist.reshape(1, -1))
            cv2.putText(smallimg, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (255, 255, 0), 3)
            cv2.putText(smallimg, prediction2[0], (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (0, 255, 0), 3)
            cv2.imshow(winname3, smallimg)

            if prediction[0] != identifier:
                prediction_error = prediction_error + 1
            if prediction2[0] != identifier:
                prediction2_error = prediction2_error + 1

            cv2.waitKey(0)

cv2.destroyAllWindows()


model_error = float(prediction_error)/float(dataset_size)
model2_error = float(prediction2_error)/float(dataset_size)
model_acc = 1 - model_error
model2_acc = 1 - model2_error
print("Validation Data set Size: " + "{}".format(dataset_size))
print("LinearSVC Accuracy: "+" {:.2%}".format(model_acc))
print("KNeighbor Accuracy: "+" {:.2%}".format(model2_acc))


print('Complete')




