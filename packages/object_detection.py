# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:28:27 2020

@author: anup
"""

from .social_distancing_configuration import MIN_CONF
from .social_distancing_configuration import NMS_THRESH
import numpy as np
import cv2
#Takes frames from social distancing
#Pre Process
#Frames, Give back to the model
#Get Outputs from the model
#compared - only persons returned
#Non Maxima Suppression
#Centroid, BBox Cord, Confidence

def detect_people(frame, net, ln, personIdx=0):
    #grab the dimension of the frame and initilize the list result
    (H,W) = frame.shape[:2]
    results =[]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB= True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    #initilize our list of detected bounding boxes,centoids,confidence
    boxes= []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            #for complete output frame

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONF:
                box= detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height)=box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y,int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #apply non maxima supression to suppress weak,overlapping, bounding boxes
    #this will find out the exact location of the person
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:
        #loop over the indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #update our results list to consist of the person
            #perdiction probability,bounding box coordinates,
            #and the centroid

            r = (confidences[i], (x,y,x+w,y+h), centroids[i])
            results.append(r)

    #return the list of results
    return results
