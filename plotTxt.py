import os
import cv2
import numpy as np
import random
import utility as ut



files = [
"2430.rec"
]

ImgDir= 'data/preProcessedSemifrontal/img/'
labelDir= 'data/preProcessedSemifrontal/label/'

imgName = "2430.jpg"
fileHeader = imgName.split('.')[0]
print "self.ImgDir + imgName: ", ImgDir + imgName
img = cv2.imread(ImgDir + imgName)
print "self.labelDir + imgNameHeader + .txt: ", labelDir + fileHeader + ".txt"
label = np.loadtxt(labelDir + fileHeader + ".txt")

img = ut.plotTarget(img, label, ifSquareOnly = True,  ifGreen = True)

cv2.imshow("img", img)
cv2.waitKey(0)
