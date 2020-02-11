##########################
#File Name:  GameTestV5.py
#Description:    The controller that manages the 24 game by using the helper
#                functions in game24aux to complete the task. Use Tensorflow
#                to perform hand detection. In each session objects are
#                selected by capturing the hand's location, a series of
#                arithmetic opeartions are performed until all objects/numbers
#                are picked. If the end result is 24, the player wins.
#
#Last Modified: 2/8/2020 - Minor Cleanup
##########################

# Import packages
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import sys
import time
import random
import math

#for display
import tkinter as tk
import PIL.Image, PIL.ImageTk

#Project specific imports
from game24Aux_csv1 import *

windowName = '24 Game!'

##This block of code can be moved to detection class

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_ssdlite_mobilenet_v2'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph_2.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_ssdlite_mobilenet_v2','labelmap.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 1
#Score thresh
MIN_SCORE_THRESH = 0.60
## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
##End dectection class


# Initialize webcam feed
End=False
def main():

    #window=tk.Tk()
    video = cv.VideoCapture(0)
    #set up the game
    #g=game(video,'localhost', 9000)
    #g=game(video,'199.80.10.204', 9000)
    g=game(video,'192.168.1.8', 9000)
    time.sleep(1)
    #g.sock.send(bytes("/Start", "utf-8"))
    gclientthread=threading.Thread(target=g.rec_procLoop)
    gclientthread.daemon = True
    gclientthread.start()
    width, height =g.getDim()
    frame_rate_calc = 1
    #talk to server and agree on the config
    #counter=0
    while(True):
        t1 = cv.getTickCount()
        ret, frame = video.read()
        frame = cv.flip(frame, 1)   #vertical flip.
        frame_expanded = np.expand_dims(frame, axis=0)
        ##This block of code should be move away to a standalone detection class
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, \
             detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        #Draw the results of the detection (aka 'visulaize the results')
        #getting hand center(s) - may want to use scores to pick the highest one
        boxes=np.squeeze(boxes)
        classes =np.squeeze(classes)
        scores = np.squeeze(scores)

        #print('*** scores *** ',scores)
        cls = classes.tolist()
        centers=[]
        ##'v==1' here is not really needed since 'h' is the only class
        max_score=0
        max_score_loc=None
        for i,v in enumerate(cls):
            if (v==1) and (scores[i]> max(MIN_SCORE_THRESH,max_score)):
                max_score=scores[i]
                max_score_loc=i
        if max_score_loc != None:
            ymin, xmin, ymax, xmax  = boxes[max_score_loc]
            y_min,y_max=int(ymin*height),int(ymax*height)
            x_min,x_max=int(xmin*width),int(xmax*width)

            # there is only one such center if exists
            center = (int((x_min+x_max)/2),int((y_min+y_max)/2))
            centers.append(center)
            cv.circle(frame, center,8 ,color['red'],-1)
        ##end detection class code

        #g.gameconfigdone=True
        #load frame
        g.loadFrame(frame)
        #identify all Blobs that are active in this loop
        if g.gameconfigdone==True:
            g.updateactiveBlobs()
            #Process Blobs, draw Blobs and prepare texts (for Blobs or not)
            #on this frame
            g.processactiveBlobs()
            #determine the hand-locked blob and actually put texts on this frame
            g.handlockedBlob(centers, frame_rate_calc)
        else:
            g.gameconfighandLock(centers, frame_rate_calc,frame)

        # Draw the results on the frame and display it 
        cv.putText(frame,"FPS: "+ str(frame_rate_calc),\
                   (10,26),tfont,0.7,color['pink'],2,cv.LINE_AA)

        # Displays the windoww
        cv.imshow(windowName, frame)

        # Calculate framerate
        t2 = cv.getTickCount()
        time1 = (t2-t1)/g.getFreq()
        frame_rate_calc = int(1/time1)

        # Press 'q' to quit
        if cv.waitKey(1) == ord('q') or g.getEndGame()==True:
            g.sock.send(bytes("/Exit", "utf-8"))
            print('in Gam24 - End game')
            break
    # Clean up
    video.release()
    cv.destroyAllWindows()
    sys.exit(0) # closes this thread


if __name__=='__main__':
    main()
