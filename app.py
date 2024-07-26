from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from pygame import mixer
from drowsiness_detection import perform_drowsiness_detection

app = Flask(__name__)

flag = True
score1 = 0


@app.route('/')
def index():
    return render_template('index.html', score=score1)


@app.route('/get_score')
def get_score():
    # Read the score from the temporary file
    with open('score_temp.txt', 'r') as file:
        score = int(file.read())

    # Determine the driver status based on the score

    if score == -1:
        score = 0
    driver_status = "Drowsy" if score > 5 else "Normal"

    return jsonify({'score': score, 'driver_status': driver_status})


@app.route('/video_feed')
def video_feed():
    return Response(perform_drowsiness_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')
