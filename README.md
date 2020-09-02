# Social Distancing Detector

**This project focuses on creating a solution to monitor whether individuals adhere to this social distancing
guideline using Deep Learning models and Visual Computing.**

* Person Detection: YOLOv3 would be used for this.

* Euclediean Distance Calculations: Projective Transformation in openCV

* Visual representation for Violations: Scatter plots, Real-time violation graph, Change in  bounding box colour

**Steps for basic detection:**

Yolo weights are not uploadedin repo. Run the get-weights.sh script for the first time.

python yolo.py

Default Input is video from webcam. For using an input image or video:

python yolo.py -i '/path/to/image/'

python yolo.py -v '/path/to/video/'

A sample image and video uploaded in repository. Projective Transformations cordinates are calibrated based on the sample images and videos.



