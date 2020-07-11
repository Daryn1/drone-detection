#!/usr/bin/env python

"""
The pipeline of the proposed drone detection algorithm described in:
https://www.mdpi.com/1424-8220/20/14/3856
"""

import os
import sys
import numpy as np
import cv2
import pybgs as bgs
import time
import torch
import torchvision
import torchvision.transforms as transforms
from mobilenetv2 import *

# Without CUDA a video processing speed is decreased by 10 times
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the drone-bird-background classifier
mobilenet_model = mobilenet_v2(pretrained=True, device=device)
mobilenet_model = mobilenet_model.to(device)
mobilenet_model.eval()
class_names = ('background', 'bird', 'drone')

# Initialize transformations to be applied to images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# Initialize structuring elements for morphological operations
small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
medium_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
big_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

# Initialize background subtraction algorithm
background_subtraction_algorithm = bgs.TwoPoints()

def main():
    capture = cv2.VideoCapture(sys.argv[1])
    if not capture.isOpened():
        print("Unable to open video")
        return
    
    start_time = time.time()
    while True:
        # Grab a frame from the video
        flag, frame = capture.read()
        
        if flag:
            # Apply the background subtraction algorithm to the video frame
            binary_image = background_subtraction_algorithm.apply(frame)
            
            # Apply morphological operations on the binary image and fill regions with nearby pixels 
            filtered_binary_image = binary_image_filtering(binary_image)
            
            # Find bounding boxes of moving objects in the filtered binary image
            bounding_boxes = find_bounding_boxes(filtered_binary_image)
            
            # Extract the images of the moving objects from the frame using the found bounding boxes
            moving_object_images = extract_images(frame, bounding_boxes)
            
            # Classify the detected images of the moving objects
            predicted_classes = moving_object_images_classification(moving_object_images)
            
            # Show the bounding boxes of the detected drones
            for i in range(len(predicted_classes)):
                if class_names[predicted_classes[i]] == 'drone':
                    (x, y, w, h) = bounding_boxes[i]
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('frame', frame)
        else:
            break
        # Press ESC to exit
        if 0xFF & cv2.waitKey(1) == 27:
            break
    
    elapsed_time = time.time() - start_time
    print("Video processing time: ", format(elapsed_time, '.2f'), " seconds")
    print("Average FPS: ", format(capture.get(1)/elapsed_time, '.2f'))
    capture.release()
    cv2.destroyAllWindows()

# This function performs morphological operations on the given binary image and fills regions with nearby pixels.
def binary_image_filtering(binary_image):
    # Morphological operations with different kernels
    binary_image = cv2.dilate(binary_image, cross_kernel, iterations = 1)
    binary_image = cv2.erode(binary_image, cross_kernel, iterations = 1)
    binary_image = cv2.dilate(binary_image, small_kernel, iterations = 1)
    # Create a copy of the binary image and connect the pixels, the distance between which is less than 70.
    binary_image_copy = cv2.dilate(binary_image.copy(), big_kernel, iterations = 2)
    # Find countours of the resulting regions
    contours, _ = cv2.findContours(binary_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # If the contour size is less than 151, ignore it
        if cv2.contourArea(contour) < 23000: # 151*151
            continue
        # Fill the contours with a larger size in the original binary image
        binary_image = cv2.drawContours(binary_image, [contour], 0, 255, -1)
    binary_image = cv2.dilate(binary_image, medium_kernel, iterations=1)
    return binary_image

# This function returns the bounding boxes of moving objects extracted from the given binary image
def find_bounding_boxes(binary_image):
    cnts, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for c in cnts:
        # If the found contour size is less than 6 or greater than 141, ignore it
        if cv2.contourArea(c) < 36 or cv2.contourArea(c) > 20000: # 6x6 and 141x141
            continue
        # Get a bounding box from a contour
        (x, y, w, h) = cv2.boundingRect(c)
        # An additional filtering. In most cases, the width of the drone is greater than the height
        if w > 2*h:
            continue
        bounding_boxes.append([x, y, w, h])
    return bounding_boxes

# This function returns images of the moving objects extracted from a given frame using given bounding boxes
def extract_images(frame, bounding_boxes):
    moving_object_images = []
    for bounding_box in bounding_boxes:
        (x, y, w, h) = bounding_box
        # Extract the image of the moving object from the frame
        moving_object_image = frame[y:(y+ h), x:(x + w)]
        moving_object_image = cv2.resize(moving_object_image, (32,32), interpolation = cv2.INTER_AREA)
        # BGR to RGB conversion
        moving_object_image = moving_object_image[:, :, [2, 1, 0]]
        # Normalize the image and convert to tensor, then append it to the list
        with torch.no_grad():
            moving_object_images.append(transform(moving_object_image))
    return moving_object_images

# This function returns model predictions calculated from the given images.
def moving_object_images_classification(moving_object_images):
    if moving_object_images:
        stacked_tensors = torch.stack(moving_object_images, dim=0)
        with torch.no_grad():
            stacked_tensors = stacked_tensors.to(device)
            model_outputs = mobilenet_model(stacked_tensors)
            _, predicted_classes = torch.max(model_outputs, 1)
        return predicted_classes
    else:
        return []

if __name__ == '__main__':
    main()
