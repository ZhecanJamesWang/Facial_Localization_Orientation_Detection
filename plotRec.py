import os
import cv2
import numpy as np
import random

rawDir = "competitionImageDataset/testset/semifrontal/"


def plotTarget(image, labels, ifGreen = False):
    img = np.copy(image)
    if ifGreen:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 255, 0), 3)
    else:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 0, 255), 3)
    return img

files = [
"12944.rec"
]
for fileName in files:
	fileHeader = fileName.split(".")[0]		
	imgName = fileHeader + ".jpg"
	img = cv2.imread(rawDir + imgName)


	fr = open(rawDir + fileName, 'r')
	lines = fr.readlines()
	xMin, yMin, xMax, yMax = lines[0].split(" ")
	xMin, yMin, xMax, yMax = float(xMin), float(yMin), float(xMax), float(yMax)
	plotImg = plotTarget(img, [xMin, yMin, xMax, yMax])
	cv2.imshow("img", plotImg)
	cv2.waitKey(0)
