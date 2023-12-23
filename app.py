# from re import S
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration
# import av
# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# import time
# import streamlit as st
# import streamlit_webrtc as webrtc


# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
# reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# lbl = ['Close', 'Open']

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]
# classes_a = [99]
# classes_b = [99]


# class VideoProcessor:
#     # global c
#     # global s
#     def recv(self, frame):
#         mixer.init()
#         sound = mixer.Sound('alarm.wav')

#         face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#         leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
#         reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
#         eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#         lbl = ['Close', 'Open']

#         model = load_model('models/cnncat2.h5')
#         path = os.getcwd()
#         font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#         count = 15
#         score = 15
#         thicc = 2
#         rpred = [99]
#         lpred = [99]
#         classes_a = [99]
#         classes_b = [99]
#         frm = frame.to_ndarray(format="bgr24")
#         height,width = frm.shape[:2]

#         gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

#         faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
#         left_eye = leye.detectMultiScale(gray)
#         right_eye =  reye.detectMultiScale(gray)

#         for (x,y,w,h) in faces:
#             cv2.rectangle(frm, (x,y) , (x+w,y+h) , (255, 0, 0) , 1 )
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frm[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#         for (x,y,w,h) in right_eye:
#             r_eye=frm[y:y+h,x:x+w]
#             count=count+1
#             r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
#             r_eye = cv2.resize(r_eye,(24,24))
#             r_eye= r_eye/255
#             r_eye=  r_eye.reshape(24,24,-1)
#             r_eye = np.expand_dims(r_eye,axis=0)
#             rpred = model.predict(r_eye)
#             classes_a =np.argmax(rpred,axis=1)

#             if(classes_a[0]==1):
#                 lbl='Open'
#             if(classes_a[0]==0):
#                 lbl='Closed'
#             break

#         for (x,y,w,h) in left_eye:
#             l_eye=frm[y:y+h,x:x+w]
#             count=count+1
#             l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
#             l_eye = cv2.resize(l_eye,(24,24))
#             l_eye= l_eye/255
#             l_eye=l_eye.reshape(24,24,-1)
#             l_eye = np.expand_dims(l_eye,axis=0)
#             lpred = model.predict(l_eye)
#             classes_b =np.argmax(lpred,axis=1)
#             if(classes_b[0]==1):
#                 lbl='Open'
#             if(classes_b[0]==0):
#                 lbl='Closed'
#             break

#         if(classes_a[0]==0 and classes_b[0]==0):
#             score=score+1
#             cv2.putText(frm,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#         # if(rpred[0]==1 or lpred[0]==1):
#         else:
#             score=score-1
#             cv2.putText(frm,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


#         if(score<0):
#             score=0
#         cv2.putText(frm,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#         if(score>15):
#             # person is feeling sleepy so we beep the alarm
#             cv2.imwrite(os.path.join(path,'image.jpg'),frm)
#             try:
#                 sound.play()

#             except:  # isplaying = False
#                 pass
#             if(thicc<16):
#                 thicc= thicc+2
#             else:
#                 thicc=thicc-2
#                 if(thicc<2):
#                     thicc=2
#             cv2.rectangle(frm,(0,0),(width,height),(0,0,255),thicc)
#         return av.VideoFrame.from_ndarray(frm, format='bgr24')


# webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
# 				rtc_configuration=RTCConfiguration(
# 					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# 					)
# 	)

from re import S
import av
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration


mixer.init()
sound = mixer.Sound('untitled.mp3')

face = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_righteye_2splits.xml')
eye_cascade = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_eye.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
classes_a = [99]
classes_b = [99]


class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        height, width = frm.shape[:2]

        # Rest of your existing code for processing the video frame...

        return av.VideoFrame.from_ndarray(frm, format='bgr24')


webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
)
