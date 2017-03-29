import os
import cv2
import numpy as np
import utility as ut
from PIL import Image

debug = False

def plotTarget(image, labels, ifGreen = False):
	img = np.copy(image)
	if ifGreen:
		cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 255, 0), 3)
	else:
		cv2.rectangle(img,(int(labels[0]), int(labels[1])),(int(labels[2]), int(labels[3])),(0, 0, 255), 3)
	return img


# rawDir = "data/competitionImageDataset/testset/semifrontal/"
# imgOutputDir = "data/preProcessedSemifrontal/img/"
# labelOutputDir = "data/preProcessedSemifrontal/label/"
rawDir = "data/competitionImageDataset/testset/profile/"
imgOutputDir = "data/preProcessedProfile/img/"
labelOutputDir = "data/preProcessedProfile/label/"


files = os.listdir(rawDir)
print len(files)
counter = 0
for fileName in files:
	if ".rec" in fileName:
		fileHeader = fileName.split(".")[0]		
	
		# if fileHeader == "3433":
			# print "get the file"
		fr = open(rawDir + fileName, 'r')
		lines = fr.readlines()
		imgName = fileHeader + ".jpg"
		img = cv2.imread(rawDir + imgName)
		w, h, _ = img.shape
		if debug:
			cv2.imshow("original", img)
		xMin, yMin, xMax, yMax = lines[0].split(" ")
		xMin, yMin, xMax, yMax = float(xMin), float(yMin), float(xMax), float(yMax)
		
		if debug:
			plotImg = plotTarget(img, [xMin, yMin, xMax, yMax])
			cv2.imshow("img", plotImg)
			cv2.waitKey(0) 

		xMean = (xMax + xMin)/2.0
		yMean = (yMax + yMin)/2.0
		xEdge = xMax - xMin
		yEdge = yMax - yMin
		edge = max(xEdge, yEdge)

		newEdge = 1.3 * edge
		newXMin = int(xMean - newEdge/2.0)
		newXMax = int(xMean + newEdge/2.0)
		newYMin = int(yMean - newEdge/2.0)
		newYMax = int(yMean + newEdge/2.0)

		if debug:	
			cv2.circle(img,(int(xMean), int(yMean)), 2, (255, 0, 0), -1)
		
		xMean = xMean - newXMin
		yMean = yMean - newYMin
		
		img = Image.fromarray(img.astype(np.uint8))
		
		cropImg = img.crop((newXMin, newYMin, newXMax, newYMax))

		cropImg = np.array(cropImg)
		
		if debug:	
			cv2.circle(cropImg,(int(xMean), int(yMean)), 2, (0,0,255), -1)

		label = np.asarray([xMean, yMean, edge])

		# # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		if debug:
			img = plotTarget(img, [newXMin, newYMin, newXMax, newYMax])
			img = ut.plotTarget(img, [xMean, yMean, newEdge], ifSquareOnly = True)
			# img = ut.plotTarget(img, label, ifSquareOnly = True, ifGreen = True)
			# img = plotTarget(cropImg, [xMean - xEdge/2.0, yMean - yEdge/2.0, xMean + xEdge/2.0, yMean + yEdge/2.0], ifGreen = True)
			img = ut.plotTarget(cropImg, label, ifSquareOnly = True, ifGreen = True)
			img = ut.plotTarget(img, label, ifSquareOnly = True, ifGreen = True)

			cv2.imshow("img", img)
			cv2.imshow("cropped", cropImg)
			cv2.waitKey(0) 

		np.savetxt(labelOutputDir + fileHeader + '.txt', label)
		cv2.imwrite(imgOutputDir + fileHeader + '.jpg', cropImg)

		counter += 1

		if counter % 100 == 0:
			print "counter: ", counter

