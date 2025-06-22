
Face Detection using OpenCV
===============================

ABOUT THE PROGRAM
---------------------
This Python program performs face detection using OpenCV. It allows users to choose between:

1. Detecting faces from a static image (default or custom path)
2. Detecting faces in real-time using the webcam

You can also select one of the following face detection methods:
- DNN-SSD (Deep Neural Network using a pre-trained Caffe model) 
- Haar Cascade 
- LBP Cascade 

Detected faces are displayed with bounding boxes. The DNN method additionally shows the confidence scores.

N.B.
---------

- The accuracy of face detection for Haar and LBP can increase or decrease by altering the 'minNeighbors' value responsible for sensitivity.
- Make sure to handle privacy permissions for custom path directories as well as webcam access for Pycharm or any other IDE especially for Mac, for the code to run seamlessly. 
