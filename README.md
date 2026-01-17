# handtracking
hand tracking with opencv, mediapipe, and tasks

Prerequisites
Before running the script, ensure you have the following installed:
-Python 3.12
-OpenCV (opencv-python)
-MediaPipe (mediapipe)
-A working webcam

Install the required packages:
-pip install opencv-python mediapipe

Model Setup (Required)
MediaPipe Tasks requires a hand landmark model file.
Create a models directory and download the model:

-mkdir -p models
-curl -L -o models/hand_landmarker.task \
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Usage
1.Clone or download this repository.

2.Navigate to the directory containing the script.

3.Run the script using Python:
- python handtracking.py


A window will open displaying your webcam feed with hand landmarks and finger count.

Press q to exit the program.
