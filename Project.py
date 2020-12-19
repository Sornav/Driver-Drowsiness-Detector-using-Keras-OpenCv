#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
from pygame import mixer
import time
from tensorflow.keras.models import load_model
import numpy as np
import cv2
path = os.getcwd()
mixer.init()
sound = mixer.Sound('alarm.wav')
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
model1 = load_model('./Model/Own_model_Left_Eye_Detection(Best).h5')
model2 = load_model('./Model/Own_model_Right_Eye_Detection(Best).h5')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_TRIPLEX
rpred=[99]
lpred=[99]
count=0
score=0
thicc=2
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
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
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(34,139,34),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'  Score:'+str(score),(100,height-20), font, 1,(0,255,255),1,cv2.LINE_AA)
    if(score>15):
        
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            cv2.putText(frame,"Drowsy",(200,200),font,2,(0,0,255),1)       
        except: 
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[6]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')


# In[10]:


get_ipython().system('jupyter nbconvert --to script Project.ipynb')


# In[ ]:




