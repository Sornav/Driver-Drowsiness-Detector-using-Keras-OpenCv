#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from tensorflow.keras.models import load_model
import numpy as np
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
model1 = load_model('./Models/Own_model_Left_Eye_Detection(Best).h5')
model2 = load_model('./Models/Own_model_Right_Eye_Detection(Best).h5')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_TRIPLEX
rpred=[99]
lpred=[99]
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(139,0,139),2)
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model2.predict_classes(r_eye)
        break
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model1.predict_classes(l_eye)
        break
    if(rpred[0]==0 and lpred[0]==0):
        cv2.putText(frame,"Eyes Closed",(10,frame.shape[:2][0]-20), font, 1,(0,0,255),1,cv2.LINE_AA)
    else:
        cv2.putText(frame,"Eyes Open",(10,frame.shape[:2][0]-20), font, 1,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow('My Face',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




