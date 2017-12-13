from imutils.object_detection import non_max_suppression
import multiprocessing as mp
from functools import partial
from imutils import paths
from objects import *
import numpy as np
import argparse
import imutils
import cv2
import argparse
import math
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
additions = ImageAdditions()

def to01(num):
    return 0 if num < 0 else 1

def playerImage_process(playerImage):
    playerImage.process()
    return playerImage

def create_player_image(result, image):
    rect, weight = result
    coord = Rect(rect)
    playerImage = PlayerImage(image, coord, weight)
    return playerImage

def add_figures(image, winStride = (4, 4), padding = (8, 8), scale = 1.06):
    global hog
    global additions
    (rects, weights) = hog.detectMultiScale(image, winStride = winStride, padding = padding, scale = scale) 

    p = Pool(4)
    playerImages = p.map(partial(create_player_image, image=image), zip(rects, weights))

    playerImages = p.map(playerImage_process, playerImages)
   
    for playerImage in playerImages:
        addition = ImageAddition(Ellipse(playerImage.coord), playerImage)
        additions.append(addition)

        cv2.ellipse( image, addition.ellipse.center(), addition.ellipse.axes(), 0, 0, 360, addition.playerImage.main_color, 2)

    #corrs = additions.get_histogram_correlation()
    #for corr in corrs:
    #    if corr.score > 0:
    #        cv2.line(image, corr.pt1, corr.pt2, corr.color, 3)

    return image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to images directory")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    cv2.imwrite("player_processed.jpg", add_figures(image))

