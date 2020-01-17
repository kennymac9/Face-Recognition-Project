import numpy as np
import cv2



def camera_loop():
    print("Press <SPACE> to capture/classify an image, or <Esc> to exit.")
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)

        cv2.imshow('camera', frame)
        cv2.moveWindow('camera', 100, 100)

        if action == ord('q') or action == 27:
            break

        if action == ord(' '):
            # svm object detection
            return frame
            cv2.imshow('Cam Out', frame)

    cap.release()


img = camera_loop()

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.moveWindow('img', 100, 100)
cv2.waitKey(0)
cv2.destroyAllWindows()