import cv2
import numpy as np
import pickle
import os

videoCapture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('./Data/haarcascade_frontalface_default.xml')

facesData = []

i = 0
name = input("Enter the name: ")
while True:
    ret, videoFrame = videoCapture.read()

    gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        croppedImg = videoFrame[y:y+h, x:x+w,:]
        resizedimg = cv2.resize(croppedImg, (50,50))

        if len(facesData) <= 100 and i%5 == 0:
            facesData.append(resizedimg)
        i = i+1
        cv2.putText(videoFrame, str(len(facesData)), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,200,0), 2)
        cv2.rectangle(videoFrame, (x,y), (x+w, y+h), (0, 200, 0), 2)

    cv2.imshow("frame",videoFrame)

    if cv2.waitKey(1) == ord('q') or len(facesData)== 100:
        break

videoCapture.release()
cv2.destroyAllWindows()

facesData = np.asarray(facesData).reshape(100, - 1)

if 'names.pkl' not in os.listdir('Data/'):
    names = [name] * 100
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names  + [name] * 100
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
        
        
if 'faces_data.pkl' not in os.listdir('Data/'):
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(facesData, f)
else:
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, facesData, axis=0)
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)


