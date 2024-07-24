from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pickle
import os
import csv
import time
from datetime import datetime
import subprocess

videoCapture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('./Data/haarcascade_frontalface_default.xml')


with open('Data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('Data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)
    
def speak(text):    
    subprocess.run(["espeak", text])    


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbg = cv2.imread("background.png")
COL_NAMES = ['NAME', 'TIME']

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
        resizedimg = cv2.resize(croppedImg, (50,50)).flatten().reshape(1, - 1)
        output = knn.predict(resizedimg)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestap = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        file_exist = os.path.isfile("Attendance/Attedance_" + date + ".csv")
        
        cv2.rectangle(videoFrame, (x,y-40), (x+w,y), (0,200,0), -1)
        cv2.putText(videoFrame, str(output[0]), (x,y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(videoFrame, (x,y), (x+w, y+h), (0, 200, 0), 2)
        attendance = [str(output[0]), str(timestap)]

    imgbg[162:162+480, 55:55+640] = videoFrame
    cv2.imshow("frame",imgbg)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance taken of " + output[0])
        time.sleep(1)
        if file_exist:
            with open("Attendance/Attedance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
                csvfile.close()
        else:
            with open("Attendance/Attedance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                csvfile.close()

            
    if k == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
