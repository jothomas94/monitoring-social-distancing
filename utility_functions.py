import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import scipy
import matplotlib.pyplot as plt

from random import randint
from scipy.spatial import distance
from collections import OrderedDict
from drawnow import drawnow

# Global variables
global xList, yList, y2List
xList=list()
yList=list()
y2List=list()


def display_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_boxes(img, boxes, confidences, classids, idxs, colours, labels, violated_index):
    
    count = 0
    j = 0

    for i in idxs.flatten():
        # Filters person from all detected categories
        if labels[classids[i]] == "person":
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            #default colour is green(BGR)
            colour = (0,255,0)

            if len(violated_index) != 0:
            # If violated set color as red 
                if count == violated_index[j]:
                    colour = (0,0,255)#red
                    if(j<(len(violated_index)-1)):
                        j +=1
            
            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), colour, 2)
            # Uncomment if the confidence and Id needs to be displayed
            #text = "{}{}: {:4f}".format(labels[classids[i]], i, confidences[i])
            #cv.putText(img, text, (x, y+h+2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            count +=1

    return img


def create_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            
            # Filter scores, classid, and confidence 
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Filter predictions above confidence level
            if confidence > tconf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Adding to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def process_image(net, layer_names, height, width, img, colours, labels, FLAGS, frame_count, ax, 
        boxes=None, confidences=None, classids=None, idxs=None, process=True):
    
    if process:

        # Creates a blob of image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Forward pass through YOLO

        net.setInput(blob)

        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] Inference time {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = create_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply NMS to remove irrelavant boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] No Boxes'

  
    person_boxes = [(0,0)]

    # Person boxes detection and identify transformed points
    person_boxes, trans_points = person_boxes_detection(boxes, classids, idxs, labels)

    #Identify violated indices
    violated_index = create_violated_index(trans_points)

    
    # On detections
    if len(idxs) > 0:
        # Draw boxes on the image
        img = draw_boxes(img, boxes, confidences, classids, idxs, colours, labels, violated_index)
 

    # To display transformed image
    # img = hom_trans(img)

    # Extra visual Aids: Scatter plot and Graphs
    # Current code only displays one at a time, Uncomment the one needed
    plot_scatter(trans_points, ax, violated_index, frame_count)
#   plot_line(frame_count, violated_index, len(person_boxes), ax)


    return img, boxes, confidences, classids, idxs



# Use this function if the transformed image needs to be displayed
def hom_trans(trans_img):

    # Source and Destination points, Cordinates are calibrated for sample image
    # Need to identify source and potential destination cordinates in case of
    # changing the image
    pts_src = np.array([[0,0], [1152, 0], [1422, 648], [-270,648]])
    pts_dst = np.array([[0,0], [1152, 0], [1152, 648], [0,648]])

    # Calculate Homography
    h, status = cv.findHomography(pts_src, pts_dst)

    # Warp the whole image based on H matrix calculated
    im_out = cv.warpPerspective(trans_img, h, (648,1152))

    return im_out


def create_violated_index(trans_points):


    violated_index = []
    max_people = len(trans_points)

    for i in range(max_people):

        for j in range((i+1), max_people):
            a = tuple(trans_points[i])
            b = tuple(trans_points[j])
            dst = distance.euclidean(a, b)

            # If the distance b/w bounding boxes less than 50 pixel units
            # Add to the violated index. This index could use to filter out
            # violated coordinates, Change 50 depending on your social-distancing
            # requirement
            if(dst<50):
                violated_index.append(i)
                violated_index.append(j)


    #Remove duplicates
    violated_index = sorted(set(violated_index))
    
    return violated_index


def resize_img(img):

    scale_percent = 60 # percent of original size
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    dim = (w, h)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    return resized

def add_border(src):

    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = 326
    right = 0  
    value = [randint(0, 255), randint(0, 255), randint(0, 255)]

    dst = cv.copyMakeBorder(src, top, bottom, left, right, cv.BORDER_CONSTANT, None)

    return dst


def transform_points(person_boxes):
    
    # Source and Destination points, Cordinates are calibrated for sample image
    # Need to identify source and potential destination cordinates in case of
    # changing the image
    pts_src = np.array([[0,0], [1152, 0], [1422, 648], [-270,648]])
    pts_dst = np.array([[0,0], [1152, 0], [1152, 648], [0,648]])


    # Calculate Homography
    h, status = cv.findHomography(pts_src, pts_dst)
    
    # Transformed points list
    trans_list = []
    
    # Calculating transformed x and y for all the bounding box center-bottom cordinates
    for i in range(len(person_boxes)):
        x,y = person_boxes[i]
        new_x = (x*h[0,0] + y*h[0,1] +  h[0,2])/(x*h[2,0] + y*h[2,1] + h[2,2])
        new_y = (x*h[1,0] + y*h[1,1] +  h[1,2])/(x*h[2,0] + y*h[2,1] + h[2,2])

        trans_list.append((int(new_x), int(new_y)))

    return trans_list


def person_boxes_detection(boxes, classids, idxs, labels):

    person_boxes = []
    count = 1

    # Weights are trained for COCO dataset. Detects all categories
    # Filters out person alone
    if len(idxs) != 0:  
        for i in idxs.flatten():
            if labels[classids[i]] == "person":
                # Get the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # Filtering only person boxes
                person_boxes.append([x+(w/2),y+h])

    # Calculate the corresponding cordinates in image after perspective transformation
    trans_points = transform_points(person_boxes)

    return person_boxes, trans_points

def plot_scatter(trans_points, ax, violated_index, frame_count):

    # Set axis for the plot
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_ylim(648,0)
    ax.set_xlim(0,1152)

    max_people = len(trans_points)
    
    # Filter out cordinates of violated people
    violated_points = [trans_points[i] for i in violated_index]

    if len(trans_points) != 0:
        x, y = zip(*trans_points)
        plt.scatter(x, y, color="g")

    # Draw red plots for violated people
    if len(violated_points) != 0:
        x, y = zip(*violated_points)
        plt.scatter(x, y, color="r")

    # Draw red line between  violated people
    for i in range(max_people):
        for j in range((i+1), max_people):
            a = tuple(trans_points[i])
            b = tuple(trans_points[j])

            dst = distance.euclidean(a, b)
            if(dst<50):
                x, y = zip(*[a,b])
                plt.plot(x,y, color="r")

    # Plot parameters
    plt.title("Real Time Scatter Plot")
    plt.xlabel("Pixel-xaxis")
    plt.ylabel("Pixel-yaxis")
    plt.grid()
    plt.pause(0.00000001)
    plt.clf()


def plot_line(frame_count, violated_index, totalPerson, ax):
 

    plt.title("Real Time Count per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    ax.legend(loc=0) 

    # List grows as frames progress
    xList.append(frame_count)
    yList.append(len(violated_index))
    y2List.append(totalPerson)

    plt.plot(xList,y2List,label = 'Total People')
    plt.plot(xList,yList,label = 'Violations')
    leg = plt.legend()
    plt.grid()
    plt.pause(0.00001)
    plt.clf()

