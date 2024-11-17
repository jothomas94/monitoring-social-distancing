# Monitoring Social Distancing

This project focuses on creating a solution to monitor whether individuals adhere to this social distancing
guideline using Deep Learning models and Visual Computing. During the 2020 pandemic one effective guideline put forward by
different governments was maintaining social distances. The proposed system uses YOLOv3
model to extract persons from each frame of video and apply Projective Transformations and
Geometry techniques to calculate the real-life distances between them. Later, the violations in the
frames are represented using real-time plots and other visual alerts. The result was a lightweight
solution written in Python which could detect people with a Mean Average Precision (mAP) value
of 0.43 and represent violations in real-time using scatter plots, violation count graphs and through
changing the bounding box colour. An attempt was also made to analyse the performance of the
model in detection and distance calculation. The entire software model is a portable design which
could be integrated to embedded solutions or to an existing surveillance system. In conclusion, this
project suggests a solution that could solve a novel problem we are currently facing by combining
areas of Deep learning detection methods and geometry in Computer Vision. 

## Stages of Execution:
![image](https://github.com/user-attachments/assets/22883233-d3f2-48f9-b388-fc0b648ab5a4)

* Person Detection: YOLOv3 would be used for this.
* Euclediean Distance Calculations: Projective Transformation in openCV
* Visual representation for Violations: Scatter plots, Real-time violation graph, Change in  bounding box colour

##Steps for execution

***Pre-Requsites:***
  Yolo weights are not uploaded in the repo. Please download weights by running the get_weights.sh script for the first time.

  ***To Run:***
  `python yolo.py`; To run the script using input from webcam
  `python yolo.py -i '/path/to/image/'` ; To run the script using image file
  `python yolo.py -v '/path/to/video/'` ; To run the script using video file

  A sample image and video uploaded in repository. Projective Transformations cordinates are calibrated based on the dimensions of sample image and video.

##Results

Snapshots of results from sample videos and thier corresponding scatter plots in pixel axis. Violations are highlighted in red.

![image](https://github.com/user-attachments/assets/bcc41287-5fa1-4819-b027-7e6d2e1272c4)

##Evaluation of Model

The accuracy in distance measurement is evaluated by comparing the software calculated distance with the real-life distance. Due to the limitations in setting up a test setup with people standing in known distances, the model was tested using images available online. The photo used was captured in Manchester Piccadilly station in April 2020, where a grid of floor stickers are placed to help people follow the social distancing. As mentioned in the article the stickers are kept at 2 meters apart, so it will be fair to assume the measurements are accurate. It can be observed in the figure that a group of train crew and station staff are standing on those stickers maintaining the 2-meter distance. Thus, these distances were measured through the model and the consistency in the values were evaluated.

![image](https://github.com/user-attachments/assets/e67283da-fdb2-4ebc-a4db-b9865c2791b3)

Among these boxes, both horizontally and vertically, there are twenty-four 2m measurements that we are interested in and the predicted values for these measurements are shown in the figure 19c When most of the measurements were within one standard deviation of two meter, only five stood out from the range and these are marked red. The measurements 1.18m, 1.36m and 0.56m were the lowest values and these distances were between the individuals standing in the back-left corner.

The model accurately measures when the subjects are closer to the camera and the error in measurements becomes evident when its farther from the field of view.


##Limitations

* Limited field of view: The detection and distance calculation results at the far end of the camera view were prone to more errors than the near end. Smaller subject size made it difficult for the model to identify them. A possible recovery step would be to use multiple cameras to cover the area we interested. If the software is ported to an embedded solution, placing one or two of such setups in the region of choice would help to assess the violations more accurately.
* Occlusion Handling: Though the model was able to perform with a fair detection accuracy, few false positives were observed for occluded persons in a frame. Few noted cases happened when the body of individuals overlap in a scene they were detected as one bounding box as shown in figure 20. Occasionally some errors were observed when vertical structures like traffic lights or trees appear closer to a person false positive were predicted. An example of oclusion seen:

![image](https://github.com/user-attachments/assets/93ae1f12-acbb-43a9-8454-d1292b2c6b87)


##Conlusions

A solution to monitor social distancing using surveillance videos was developed during this project. Different surveys and literatures on object detection using DNN was analysed to find a suitable method for person detection. The proposed solution is an algorithm that combines detection through YOLOv3 and projective techniques in geometry to predict the violations in social distancing. This potable software could be deployed in an embedded platform or existing surveillance solutions. The performance of the system in both detection and distance measurement were evaluated separately and the results were discussed. Using more advanced version YOLO would definitely improve the limitations discussed and improve the accuracy of the system. 
