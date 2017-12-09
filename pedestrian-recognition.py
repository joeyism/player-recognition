from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from objects import Rect, Ellipse, PlayerImage

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def add_figures(image):
    global hog
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05) 
    additions= []
    for rect in rects:
        coord = Rect(rect)

        playerImage = PlayerImage(image, coord)
        additions.append((Ellipse(coord), playerImage))
        
    for addition in additions:
        ellipse = addition[0]
        playerImage = addition[1]
        cv2.ellipse( image, ellipse.center(), ellipse.axes(), 0, 0, 360, playerImage.colors[0], 2)

    return image


image = cv2.imread("liverpool-chelsea.jpeg")
cv2.imwrite("test.jpg", add_figures(image))

