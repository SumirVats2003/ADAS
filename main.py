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

face = cv2.CascadeClassifier(
    'haarCascadeFiles/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(
    'haarCascadeFiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    'haarCascadeFiles/haarcascade_righteye_2splits.xml')
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
