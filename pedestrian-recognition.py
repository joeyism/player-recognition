from imutils.object_detection import non_max_suppression
from imutils import paths
from objects import *
import numpy as np
import argparse
import imutils
import cv2
import argparse
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to images directory")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
additions = ImageAdditions()

def to01(num):
    return 0 if num < 0 else 1


def add_figures(image):
    global hog
    global additions
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.01) 
    print("Found {} players".format(len(rects)))
    for i, rect in enumerate(rects):
        weight = weights[i][0]
        print("Player {} with weight {}".format(i, weight))
        coord = Rect(rect)
        playerImage = PlayerImage(image, coord, weight, debug=i)
        additions.append(ImageAddition(Ellipse(coord), playerImage))
        # cv2.imwrite("player" + str(i) + ".jpg", playerImage.image) # debug

    for addition in additions:
        cv2.ellipse( image, addition.ellipse.center(), addition.ellipse.axes(), 0, 0, 360, addition.playerImage.main_color, 2)

    corrs = additions.get_histogram_correlation()
    #for corr in corrs:
    #    if corr.score > 0:
    #        cv2.line(image, corr.pt1, corr.pt2, corr.color, 3)

    return image

image = cv2.imread(args["image"])
cv2.imwrite("player_processed.jpg", add_figures(image))

