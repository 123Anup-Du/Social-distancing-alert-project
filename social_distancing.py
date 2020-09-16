# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:28:27 2020

@author: anup
"""

#loading cam/video
#load yolo, weights, labels

#Results centroids, confidence(80/90), probability(0/1/2)
#Fets Only Person value form Object Detection
#Start Calculating Euc dist
#Violation
#Draw Circles / Rect
#Output

from packages import social_distancing_configuration as config
from packages.object_detection import detect_people

from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os


labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])

configPath = os.path.sep.join([config.MODEL_PATH,"yolov3.cfg"])
print("[INFO] loading yolo from the device")
#coco 80 classes
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#check if we are going to use GPU
if config.USE_GPU:
    #set CUDA as the preferable backend and the target
    print("[INFO] setting perferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)
# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initilize the video stream and the pointer to output video file
print("[INFO] accessing video stream...")

vs = cv2.VideoCapture(r"pedestrians.mp3" if "pedestrians.mp3" else 0)
writer = None


#loop over the frames from the video stream
while True:
    #read the next frames from the file
    (grabbed, frame) = vs.read()
    #if the frame was not grabbed. then we have reached the end of the stream
    if not grabbed:
        break
    #resize the frame and then detect people (and only people) in
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,
                            personIdx=LABELS.index("person"))
    #initialize the set of indexes the violate the minimum social distance
    violate = set()

    #ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    if len(results) >= 2:
        #extract all centroids from the results and compute the
        #euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric= "euclidean")

        #loop over the upper triangular of the distancec metrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                #check to see if the distance between any two
                #centroid pairs is less than the configured number
                #of pixel
                if D[i, j]< config.MIN_DISTANCE:
                    #update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)

    #loop over the result
    for (i, (prob, bbox, centroid)) in enumerate(results):
        #extract the bounding box and centroid coordiantes, then
        #initilize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0) #BGR

        #if the index pair exists within the violation set, then
        #update the color
        if i in violate:
            color = (0, 0, 255)
        #draw (1) bounding box around the person and (2) the
        #centroid coordiantes of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
    #draw the total number of social distancing violation on the
    #output frame
    text = "Social Distancing violations : {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0]-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 3)

    #check to see if the output frame should be displayed to our screen
    #show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    #if the key 'q' key was pressed, break the loop
    if key == ord("q"):
        break
    #if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if r"social distance" != "" and writer is None:
        #initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.avi", fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)
    # if the video writer is not None, write the frame to the output
    #video file
    if writer is not None:
        writer.write(frame)
cv2.destroyAllWindows()
