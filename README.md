# Object-Detection-Education-Game
AR 24-game with object detection for math practice for young children.

# Description
This program was intended to be a tool to help children in Alabama education system practice math by playing an augmented reality video game. The game is based on the Chinese 24 card game, where the player will be presented with 4 random numbers, and they are expected to use 4 arithmetic operators to reach the number 24. In this game, the user picks the numbers and the operators with the motion of their hand. This project is currently in a very early stage, and is on hold. 

Game UI:

![Game UI](/images/1.png)
![Game UI](/images/9.png)


# Requirements
Python3
numpy
tkinter
tensorflow
matplotlib
cv2

# Instructions
After acquiring the required libraries, get tensorflow models repository (found here: [link](https//github.com/tensorflow/models)) and place the files on models/research/object_detection folder. Object detection model is not provided here. You may train your own detection model and use other "cursors" to play the game. After providing an object detection model, run GameTestV5.py with Python 3.
