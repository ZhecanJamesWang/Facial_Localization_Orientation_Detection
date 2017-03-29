import os
import cv2
import numpy as np
import random

def plotTarget(image, labels, ifGreen = False):
    img = np.copy(image)
    if ifGreen:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 255, 0), 3)
    else:
        cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 0, 255), 3)
    return img

# rawDir = "competitionImageDataset/testset/profile/"
rawDir = "data/competitionImageDataset/testset/semifrontal/"

files = os.listdir(rawDir)
random. shuffle(files)

print len(files)
counter = 0
for fileName in files:
	if ".rec" in fileName:
		fileHeader = fileName.split(".")[0]		
		
		JSBBFile = fileHeader + ".JSBB_Select"
		fr = open(rawDir + JSBBFile, 'r')
		lines = fr.readlines()
		index = int(lines[0])
		if index == -1:
			print "fileName: ", fileName
			print lines

			imgName = fileHeader + ".jpg"
			img = cv2.imread(rawDir + imgName)

			if index < 0:
				JSBBFile = fileHeader + ".JSBB_Update"
			else:
				JSBBFile = fileHeader + ".JSBB"
			fr = open(rawDir + JSBBFile, 'r')
			lines = fr.readlines()
			boudingBoxInfo = lines[0]
			print "boudingBoxInfo: ", boudingBoxInfo


			fr = open(rawDir + fileName, 'r')
			lines = fr.readlines()
			xMin, yMin, xMax, yMax = lines[0].split(" ")
			xMin, yMin, xMax, yMax = float(xMin), float(yMin), float(xMax), float(yMax)
			print "writing into:", str(xMin) + " " + str(yMin) + " " + str(xMax) + " " + str(yMax)
			plotImg = plotTarget(img, [xMin, yMin, xMax, yMax])
			cv2.imshow("img", plotImg)
			cv2.waitKey(0)

		
# JSBBFile = fileHeader + ".JSBB_Update"
# fr = open(rawDir + JSBBFile, 'r')
# lines = fr.readlines()
# confidence, xMin, yMin, xMax, yMax = boudingBoxInfo.split("\t")
# confidence, xMin, yMin, xMax, yMax = float(confidence), float(xMin), float(yMin), float(xMax), float(yMax[:-1])




# fileName:  14002.rec
# fileName:  14085.rec

# fileName:  14275.rec

# fileName:  709.rec
# fileName:  10776.rec
# fileName:  12078.rec
# fileName:  8342.rec


# fileName:  9602.rec
# fileName:  2330.rec
# fileName:  14329.rec
# fileName:  12244.rec
# fileName:  9708.rec
# fileName:  15014.rec
# fileName:  6953.rec
# fileName:  6670.rec
# fileName:  2453.rec
# fileName:  14983.rec
# fileName:  9019.rec
# fileName:  9967.rec
# fileName:  1927.rec
# fileName:  7127.rec
# fileName:  14002.rec
# fileName:  14085.rec
# fileName:  14275.rec
# fileName:  709.rec
# fileName:  10776.rec
# fileName:  12078.rec
# fileName:  8342.rec
