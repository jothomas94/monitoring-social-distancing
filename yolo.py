import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import matplotlib.pyplot as plt

from utility_functions import process_image, display_image, resize_img

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
        type=str,
        default='./',
        help='The directory where the model weights and \
              configuration files are present')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./yolov3.weights',
        help='Path to the file which contains the weights \
                 for YOLOv3')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./yolov3.cfg',
        help='Path to the configuration file for the YOLOv3 model')

    parser.add_argument('-i', '--image-path',
        type=str,
        help='The path to the image file')

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output1.avi',
        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./coco-labels',
        help='Path to the file having the \
                    labels')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='Model will supress boxes with \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.5,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-weights',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
                help='Show the time taken to process each image.')

    FLAGS, unparsed = parser.parse_known_args()

    # Download the YOLOv3 published weights
    if FLAGS.download_weights:
        subprocess.call(['./get_weights.sh'])

    # Get labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colours 
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation 
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
    # Raise error if image and video files are both given 
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print ('Neither path to an image or path to video provided')
        print ('Starting Inference on Webcam')

    #Process the given image
    if FLAGS.image_path:
        # Read the image
        try:
            img = cv.imread(FLAGS.image_path)
            # Resized to a standard size
            img = cv.resize(img, (1152,648), interpolation = cv.INTER_AREA)
            print(img.shape)

            height, width = img.shape[:2]

        except:
            raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            img,  _, _, _, _ = process_image(net, layer_names, height, width, img, colours, labels, FLAGS, 0, 0)
                        
            #Write to a file
            cv.imwrite('./Results/Output-image.jpg', img)
            display_image(img)

    elif FLAGS.video_path:
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            #Plot initialisation
            x = 0
            y = 0
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Frame count for real-time plot
            frame_count = 0

            while True:
                grabbed, frame = vid.read()
                frame = resize_img(frame)
                height, width = frame.shape[:2]
                                            
                frame_count += 1 
                # Checking if the complete video is read
                if not grabbed:
                    break

                frame, _, _, _, _ = process_image(net, layer_names, height, width, frame, colours, labels, FLAGS, frame_count, ax)

                cv.imshow('image', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break


                if writer is None:
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 10, 
                                    (frame.shape[1], frame.shape[0]), True)

                writer.write(frame)

            print ("[INFO] Cleaning up...")
            writer.release()
            vid.release()


    else:
        # Process real-time on webcam
        count = 0

        vid = cv.VideoCapture(0)
        while True:
            _, frame = vid.read()
            height, width = frame.shape[:2]

            if count == 0:
                frame, boxes, confidences, classids, idxs = process_image(net, layer_names, \
                                    height, width, frame, colours, labels, FLAGS, 0, 0)
                count += 1
            else:
                frame, boxes, confidences, classids, idxs = process_image(net, layer_names, \
                                    height, width, frame, colours, labels, FLAGS, 0, 0, boxes, confidences, classids, idxs, process=False)
                count = (count + 1) % 6
            cv.imshow('webcam', frame)
    
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()

