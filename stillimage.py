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



trainingdir = "./trainingFaces"
testingfolder = "./testingFaces"
validationdir = "./validationFaces"


model_filename1 = 'LinearSVCfinalized_model.sav'
model_filename2 = 'KNeigborsfinalized_model.sav'

winname = 'Orignial Image with Face Detection'
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
cv2.moveWindow(winname3, 100, 500)


identifiers = []
lbpdata = []
dataset_size = 0
prediction_error = 0
prediction2_error = 0

neighborpts = 16
radius = 1
penalty = 1
nneighbors = 1
leafsize = 30

#Inputs - train, test, validate, both, all
requests = raw_input('Train, Test or Validate? ')


if requests == 'both' or requests == 'train' or requests == 'all':

    print('Training')
        
    #Array of Images
    for folder in os.listdir(trainingdir):
        if not folder.startswith('.'):
            training_images = load_images_folder_and_directory(folder,trainingdir)
            identifier = folder
            print(identifier)
        
            #Detecting Face in each image and create LBP
            for img in training_images:

                img, facelocations, face = detectfaces(img)
                height, width = img.shape[:2]
                scaleH = float(286)/float(height)
                scaleW = float(384)/float(width)

                smallimg = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                cv2.imshow(winname, smallimg)
                
                if len(face) != 0:

                    cv2.imshow(winname1, face[0])

                    lbp = feature.local_binary_pattern(face[0], neighborpts, radius, method="nri_uniform")

                    hist = lbp_histogram(lbp, neighborpts)

                    [float(i) for i in hist]

                    # print(hist)
                    # print(len(hist))

                    cv2.imshow(winname2, lbp.astype("uint8"))

                    identifiers.append(identifier)
                    lbpdata.append(hist)
                    

                    
                    # cv2.waitKey(0)


    lbpnorm, lbpmean, lbpstddev = lbp_normalize_values(lbpdata)


    model = LinearSVC(C=penalty, max_iter=100000)
    model.fit(lbpnorm, identifiers)

    model2 = KNeighborsClassifier(n_neighbors=nneighbors, leaf_size = leafsize)
    model2.fit(lbpnorm, identifiers)

    savemodel = raw_input("Type save to save model or anything else not to: ")
    if savemodel == 'save':
        pickle.dump(model, open(model_filename1, 'wb'))
        pickle.dump(model2, open(model_filename2, 'wb'))
        with open('lbpmean', 'wb') as fp:
            pickle.dump(lbpmean, fp)
        with open('lbpstddev', 'wb') as fp:
            pickle.dump(lbpstddev, fp)
        with open('neighboring points', 'wb') as fp:
            pickle.dump(neighborpts, fp)
        with open('radius', 'wb') as fp:
            pickle.dump(radius, fp)
        print('Models Saved')
    

    print('Testing on Training Set')

    for folder in os.listdir(trainingdir):
        if not folder.startswith('.'):
            training_images = load_images_folder_and_directory(folder,trainingdir)
            identifier = folder
            dataset_size = len(training_images)+dataset_size

            for img in training_images:

                height, width = img.shape[:2]
                scaleH = float(286)/float(height)
                scaleW = float(384)/float(width)
                
                img, facelocations, face = detectfaces(img)
                smallimg = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                smallimgOrignial = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)

                cv2.putText(smallimgOrignial, identifier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 0, 255), 3)
                cv2.imshow(winname, smallimgOrignial)

                if len(face) != 0:
                    cv2.imshow(winname1, face[0])

                    lbp = feature.local_binary_pattern(face[0], neighborpts, 
                    radius, method="nri_uniform")

                    hist = lbp_histogram(lbp, neighborpts)
                    
                    normalhist = lbp_norm_with_data(hist,lbpmean,lbpstddev)
                    

                    cv2.imshow(winname2, lbp.astype("uint8"))

                    prediction = model.predict(normalhist.reshape(1, -1))
                    prediction2 = model2.predict(normalhist.reshape(1, -1))

                    x, y, w, h = facelocations[0]
                    cv2.putText(smallimg, prediction[0], (int(round(scaleW*x)), int(round(scaleH*y))), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (255, 255, 0), 3)
                    cv2.putText(smallimg, prediction2[0], (int(round(scaleW*x)), int(round(scaleH*(y+h)))), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 3)
                    
                    # cv2.putText(smallimg, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    # 1.0, (255, 255, 0), 3)
                    # cv2.putText(smallimg, prediction2[0], (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 
                    # 1.0, (0, 255, 0), 3)
                    
                    cv2.imshow(winname3, smallimg)


                    if prediction[0] != identifier:
                        prediction_error = prediction_error + 1
                    if prediction2[0] != identifier:
                        prediction2_error = prediction2_error + 1

                    # cv2.waitKey(0)


    print("Data set Size: " + "{}".format(dataset_size))
    error_and_acc('LinearSVC', prediction_error, dataset_size )
    error_and_acc('KNeighbor', prediction2_error, dataset_size )

if requests == 'both' or requests == 'validate' or requests == 'all':
        
    print('Validating')
    print('Use any key to step through')

    model = pickle.load(open(model_filename1, 'rb'))
    model2 = pickle.load(open(model_filename2, 'rb'))
    with open ('lbpmean', 'rb') as fp:
        lbpmean = pickle.load(fp)
    with open ('lbpstddev', 'rb') as fp:
        lbpstddev = pickle.load(fp)

    dataset_size = 0
    prediction_error = 0
    prediction2_error = 0
    
    for folder in os.listdir(validationdir):
        if not folder.startswith('.'):
            validation_images = load_images_folder_and_directory(folder,validationdir)
            identifier = folder
            if identifier != 'twofaces':
                dataset_size = len(validation_images)+dataset_size


            
            for img in validation_images:
                height, width = img.shape[:2]
                scaleH = float(286)/float(height)
                scaleW = float(384)/float(width)
                
                predictions = []
                predictions2 = []

                
                img, facelocations, detectedfaces = detectfaces(img)
                
                if len(detectedfaces) != 0:
                    

                
                    smallimg = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                    smallimgOrignial = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                    cv2.putText(smallimgOrignial, identifier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 255), 3)
                    cv2.imshow(winname, smallimgOrignial)
                    
                
                    for i in range(len(detectedfaces)):
                        cv2.imshow(winname1+" Face: {}".format(i), detectedfaces[i])
                        cv2.moveWindow(winname1+" Face: {}".format(i), 600+(500*i), 100)

                        lbp = feature.local_binary_pattern(detectedfaces[i], neighborpts, radius, method="nri_uniform")
                        
                        cv2.imshow(winname2+" Face: {}".format(i), lbp.astype("uint8"))
                        cv2.moveWindow(winname2+" Face: {}".format(i), 600+(500*i), 400)

                        hist = lbp_histogram(lbp, neighborpts)
                        normallbp = lbp_norm_with_data(hist,lbpmean,lbpstddev)

                        predictions.append(model.predict(normallbp.reshape(1, -1)))
                        predictions2.append(model2.predict(normallbp.reshape(1, -1)))
                        
                        if predictions[i] != identifier and identifier != 'twofaces':
                            prediction_error = prediction_error + 1
                            print('error')
                        if predictions2[i] != identifier and identifier != 'twofaces':
                            prediction2_error = prediction2_error + 1
                            print('error')



                    i = 0
                    for (x, y, w, h) in facelocations:
                        cv2.putText(smallimg, predictions[i][0], (int(round(scaleW*x)), int(round(scaleH*y))), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 0), 3)
                        cv2.putText(smallimg, predictions2[i][0], (int(round(scaleW*x)), int(round(scaleH*(y+h)))), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 3)
                        i = i+1
                    
                    cv2.imshow(winname3, smallimg)
                    cv2.waitKey(0)


    print("Data set Size: " + "{}".format(dataset_size))
    error_and_acc('LinearSVC', prediction_error, dataset_size )
    error_and_acc('KNeighbor', prediction2_error, dataset_size )

if requests == 'test' or requests == 'all':
        
    print('Testing')
    print('Use any key to step through')

    model = pickle.load(open(model_filename1, 'rb'))
    model2 = pickle.load(open(model_filename2, 'rb'))
    with open ('lbpmean', 'rb') as fp:
        lbpmean = pickle.load(fp)
    with open ('lbpstddev', 'rb') as fp:
        lbpstddev = pickle.load(fp)

    dataset_size = 0
    
    for folder in os.listdir(testingfolder):
        if not folder.startswith('.'):
            test_images = load_images_folder_and_directory(folder,testingfolder)
            identifier = folder
            dataset_size = len(test_images)+dataset_size


            
            for img in test_images:
                height, width = img.shape[:2]
                scaleH = float(286)/float(height)
                scaleW = float(384)/float(width)
                
                predictions = []
                predictions2 = []

                
                img, facelocations, detectedfaces = detectfaces(img)
                
                if len(detectedfaces) != 0:
                    

                
                    smallimg = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                    smallimgOrignial = cv2.resize(img,(0,0),fx = scaleW,fy = scaleH,interpolation=cv2.INTER_AREA)
                    cv2.putText(smallimgOrignial, identifier, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 255), 3)
                    cv2.imshow(winname, smallimgOrignial)
                    
                
                    for i in range(len(detectedfaces)):
                        cv2.imshow(winname1+" Face: {}".format(i), detectedfaces[i])
                        cv2.moveWindow(winname1+" Face: {}".format(i), 600+(500*i), 100)

                        lbp = feature.local_binary_pattern(detectedfaces[i], neighborpts, radius, method="nri_uniform")
                        
                        cv2.imshow(winname2+" Face: {}".format(i), lbp.astype("uint8"))
                        cv2.moveWindow(winname2+" Face: {}".format(i), 600+(500*i), 400)

                        hist = lbp_histogram(lbp, neighborpts)
                        normallbp = lbp_norm_with_data(hist,lbpmean,lbpstddev)

                        predictions.append(model.predict(normallbp.reshape(1, -1)))
                        predictions2.append(model2.predict(normallbp.reshape(1, -1)))

                    i = 0
                    for (x, y, w, h) in facelocations:
                        cv2.putText(smallimg, predictions[i][0], (int(round(scaleW*x)), int(round(scaleH*y))), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 0), 3)
                        cv2.putText(smallimg, predictions2[i][0], (int(round(scaleW*x)), int(round(scaleH*(y+h)))), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 3)
                        i = i+1
                    
                    cv2.imshow(winname3, smallimg)
                    cv2.waitKey(0)

cv2.destroyAllWindows()
print('Complete')




